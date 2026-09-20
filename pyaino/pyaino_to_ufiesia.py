#!/usr/bin/env python3
# pyaino_to_ufiesia.py
# pyaino -> ufiesia mechanical converter
#
# 方針
#   1. コメント・配置をできるだけ保存する（ASTでソースを再生成しない）
#   2. ASTは構造把握と監査だけに使う
#   3. 共通変換 + モジュール固有パッチ
#   4. nucleus固有要素が残ったら監査で検出する
#
# 実証済み主対象:
#   Config / Functions / Activators / Optimizers / Initializer
#   LossFunctions / Regularizers / Neuron / common_function
#
# nucleus / HDFunctions / safe_np は ufiesia には持ち込まない。
# Config は ufiesia0 相当の最小構成を生成する。
# `from ufiesia.Config import *` 後の `np = Config.np` は自動挿入しない。

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path


# pyaino の正式モジュールだが、ufiesia には持ち込まないもの。
# 変換器自身を pyaino に置く場合も CURRENT_MODULES に登録してよい。
UFIESIA_EXCLUDE = {
    "nucleus",
    "HDFunctions",
    "safe_np",
    "pyaino_to_ufiesia",
}

FUNCTION_BASES = {
    "Function",
    "CompositFunction",
    "nucleus.Function",
    "nucleus.CompositFunction",
}

# これらのmoduleから作ったclass instanceは forward() を持つものとして扱う。
FORWARD_MODULE_ALIASES = {
    "F", "Functions",
    "Activators",
    "lf", "LossFunctions",
    "Regularizers",
}

# 静的型推定しにくいが、今回の手変換で forward() と確定した属性。
FORCE_FORWARD_ATTRS = {
    "Functions": {"func", "primitive", "take", "take1", "take2"},
    "Regularizers": {"unit", "mean", "var", "take", "take_pair", "square_mean"},
    "Neuron": {
        "dot_linear", "proj", "sampling",
        "token_embedding", "position_embedding", "broadcast_to",
        "linear", "softmax", "attention",
    },
}

# Functions.py の __main__ で f() から生成されるもの。
FORCE_FORWARD_NAMES = {
    "Functions": {"func", "func1", "func2"},
}


@dataclass
class AuditResult:
    file: str
    compile_ok: bool = True
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.compile_ok and not self.issues


@dataclass
class ConvertResult:
    src: str
    dst: str
    action: str
    audit: AuditResult | None = None
    notes: list[str] = field(default_factory=list)
    write_state: str | None = None  # new / update / same


def read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def write_source(path: Path, text: str) -> str:
    """
    text を path に保存する。

    既存ファイルと内容が同じ場合は書き換えず、更新日時も保持する。
    戻り値は "new" / "update" / "same" のいずれか。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old_text = path.read_text(encoding="utf-8-sig")
        if old_text == text:
            return "same"
        state = "update"
    else:
        state = "new"

    path.write_text(text, encoding="utf-8", newline="\n")
    return state


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted_name(node.value)
        if left:
            return left + "." + node.attr
    return None


def iter_class_defs(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield node


def class_index(tree: ast.AST):
    info = {}
    for cls in iter_class_defs(tree):
        bases = [dotted_name(b) or "" for b in cls.bases]
        methods = {
            n.name: n for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        info[cls.name] = {"node": cls, "bases": bases, "methods": methods}
    return info


def compute_forwardable_classes(info) -> set[str]:
    fwd = set()
    for name, meta in info.items():
        bases = set(meta["bases"])
        methods = set(meta["methods"])
        if bases & FUNCTION_BASES or "forward" in methods or "__forward__" in methods:
            fwd.add(name)

    changed = True
    while changed:
        changed = False
        for name, meta in info.items():
            if name in fwd:
                continue
            base_names = {b.split(".")[-1] for b in meta["bases"]}
            if base_names & fwd:
                fwd.add(name)
                changed = True
    return fwd


def parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[c] = p
    return parents


def enclosing(node: ast.AST, parents, typ):
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, typ):
            return cur
    return None


def enclosing_scope(node: ast.AST, parents):
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            return cur
    return None


def class_ref_from_call(call: ast.Call) -> tuple[str | None, str | None]:
    f = call.func
    if isinstance(f, ast.Name):
        return None, f.id
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f.value.id, f.attr
    return None, None


def is_forward_constructor(call: ast.Call, local_forwardable: set[str]) -> bool:
    mod, name = class_ref_from_call(call)
    if name is None:
        return False
    if mod is None:
        return name in local_forwardable
    return mod in FORWARD_MODULE_ALIASES


def byte_col_to_char(line: str, byte_col: int) -> int:
    raw = line.encode("utf-8")
    return len(raw[:byte_col].decode("utf-8", errors="ignore"))


def insert_forward_calls(source: str, module_name: str, tree: ast.AST) -> str:
    """Function.__call__ 依存の呼出しに .forward を挿入する。"""
    info = class_index(tree)
    local_forwardable = compute_forwardable_classes(info)
    parents = parent_map(tree)

    class_attrs: dict[str, set[str]] = {name: set() for name in info}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        cls = enclosing(node, parents, ast.ClassDef)
        if cls is None or not is_forward_constructor(node.value, local_forwardable):
            continue
        for t in node.targets:
            if (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                class_attrs.setdefault(cls.name, set()).add(t.attr)

    class_attrs.setdefault("*", set()).update(FORCE_FORWARD_ATTRS.get(module_name, set()))

    scope_names: dict[int, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not is_forward_constructor(node.value, local_forwardable):
            continue
        scope = enclosing_scope(node, parents)
        if scope is None:
            continue
        for t in node.targets:
            if isinstance(t, ast.Name):
                scope_names.setdefault(id(scope), set()).add(t.id)

    scope_names.setdefault(id(tree), set()).update(FORCE_FORWARD_NAMES.get(module_name, set()))

    src_lines = source.splitlines()
    inserts: dict[int, list[tuple[int, str]]] = {}

    def add_insert(lineno: int, col_bytes: int):
        if not (1 <= lineno <= len(src_lines)):
            return
        col = byte_col_to_char(src_lines[lineno - 1], col_bytes)
        inserts.setdefault(lineno, []).append((col, ".forward"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Class(...)(...) -> Class(...).forward(...)
        if isinstance(node.func, ast.Call):
            inner = node.func
            if is_forward_constructor(inner, local_forwardable):
                add_insert(inner.end_lineno, inner.end_col_offset)
                continue

        # self.attr(...) -> self.attr.forward(...)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            cls = enclosing(node, parents, ast.ClassDef)
            attrs = set(class_attrs.get("*", set()))
            if cls is not None:
                attrs |= class_attrs.get(cls.name, set())
            if node.func.attr in attrs:
                add_insert(node.func.end_lineno, node.func.end_col_offset)
                continue

        # local(...) -> local.forward(...)
        if isinstance(node.func, ast.Name):
            scope = enclosing_scope(node, parents)
            names = set(scope_names.get(id(tree), set()))
            if scope is not None:
                names |= scope_names.get(id(scope), set())
            if node.func.id in names:
                add_insert(node.func.end_lineno, node.func.end_col_offset)

    if not inserts:
        return source

    lines = source.splitlines(keepends=True)
    for lineno, edits in inserts.items():
        line = lines[lineno - 1]
        for col, token in sorted(set(edits), reverse=True):
            if line[col:col + len(token)] == token:
                continue
            line = line[:col] + token + line[col:]
        lines[lineno - 1] = line
    return "".join(lines)


def rewrite_class_header(line: str) -> str:
    m = re.match(
        r"^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_]\w*)\((?P<bases>[^)]*)\)(?P<tail>\s*:.*)$",
        line,
    )
    if not m:
        return line

    bases = [x.strip() for x in m.group("bases").split(",") if x.strip()]
    filtered = [b for b in bases if b not in FUNCTION_BASES]
    if filtered == bases:
        return line

    prefix = f'{m.group("indent")}class {m.group("name")}'
    if filtered:
        return prefix + "(" + ", ".join(filtered) + ")" + m.group("tail")
    return prefix + m.group("tail")


def special_ranges(original: str, module_name: str, tree: ast.AST):
    """start_line -> (end_line, replacement_lines)"""
    ranges = {}
    notes = []
    info = class_index(tree)

    if module_name == "Activators":
        cls = info.get("ActivatorBase")
        if cls and "forward" in cls["methods"]:
            m = cls["methods"]["forward"]
            ranges[m.lineno] = (m.end_lineno, [])
            notes.append("ActivatorBase.forward bridge removed")

    if module_name == "Neuron":
        # BaseLayer派生クラスはpyaino側ですでに _forward/_backward を持つ。
        # 変換器側でのメソッド名の挿げ替えは行わない。
        aliases = {
            "KullbackLeiblerDivergenceNormal2": [
                "class KullbackLeiblerDivergenceNormal2(KullbackLeiblerDivergenceNormalBasic):",
                "    pass",
            ],
            "MutualInformationLoss2": [
                "class MutualInformationLoss2(MutualInformationLoss):",
                "    pass",
            ],
        }
        for name, repl in aliases.items():
            meta = info.get(name)
            if meta:
                clsnode = meta["node"]
                ranges[clsnode.lineno] = (clsnode.end_lineno, repl)
                notes.append(f"{name} -> explicit implementation alias")

    return ranges, notes


def structural_transform(source: str, module_name: str) -> tuple[str, list[str]]:
    original = source
    tree = ast.parse(original)
    info = class_index(tree)

    # 改行数を変える前に、ASTの位置情報で .forward を挿入する。
    source = insert_forward_calls(source, module_name, tree)

    ranges, notes = special_ranges(original, module_name, tree)

    method_rename: dict[int, str] = {}
    method_add_kwargs: set[int] = set()

    # Activators では、pyaino の ActivatorBase.forward(x, **kwargs) が担っていた
    # 「共通インターフェース由来の余分な kwargs を吸収する」規律を、
    # ufiesia 側では各 Activator の forward(x, **kwargs) に展開する。
    # 直接派生だけでなく、中間基底クラスを挟んだ派生も対象にする。
    activator_classes: set[str] = set()
    if module_name == "Activators" and "ActivatorBase" in info:
        activator_classes.add("ActivatorBase")
        changed = True
        while changed:
            changed = False
            for cls_name, meta in info.items():
                if cls_name in activator_classes:
                    continue
                base_names = {b.split(".")[-1] for b in meta["bases"]}
                if base_names & activator_classes:
                    activator_classes.add(cls_name)
                    changed = True

    # 同名classが複数あっても落とさないよう、辞書ではなく全ClassDefを走査する。
    # line_direct_function[n] は、その行が「Functionだけを直接の親に持つclass」内かを示す。
    line_direct_function: dict[int, bool] = {}
    class_nodes = list(iter_class_defs(tree))

    for cls in class_nodes:
        name = cls.name
        bases = [dotted_name(b) or "" for b in cls.bases]
        meaningful = [b for b in bases if b not in FUNCTION_BASES]
        direct_only = bool(bases and not meaningful and any(b in FUNCTION_BASES for b in bases))

        for n in range(cls.lineno, cls.end_lineno + 1):
            line_direct_function[n] = direct_only

        for meth in cls.body:
            if not isinstance(meth, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            meth_name = meth.name

            if (
                module_name == "Activators"
                and name in activator_classes
                and name != "ActivatorBase"
                and meth_name == "__forward__"
                and meth.args.kwarg is None
            ):
                method_add_kwargs.add(meth.lineno)

            new_name = None
            if meth_name == "__forward__":
                new_name = "forward"
            elif meth_name == "__backward__":
                new_name = "backward"
            elif meth_name.startswith("__forward__"):
                new_name = "forward" + meth_name[len("__forward__"):]
            elif meth_name.startswith("__backward__"):
                new_name = "backward" + meth_name[len("__backward__"):]
            if new_name:
                method_rename[meth.lineno] = new_name

    lines = source.splitlines()
    out = []
    i = 1
    while i <= len(lines):
        if i in ranges:
            end, repl = ranges[i]
            out.extend(repl)
            i = end + 1
            continue

        line = lines[i - 1]

        # pyaino固有importを除去。
        if re.match(r"^\s*from\s+pyaino\.nucleus\s+import\b", line):
            i += 1
            continue
        if re.match(r"^\s*from\s+pyaino\s+import\s+nucleus\b", line):
            i += 1
            continue
        if re.match(r"^\s*from\s+pyaino\s+import\s+safe_np\s+as\s+snp\b", line):
            i += 1
            continue

        line = line.replace("from pyaino.", "from ufiesia.")
        line = line.replace("from pyaino import", "from ufiesia import")
        line = line.replace("import pyaino.", "import ufiesia.")

        line = rewrite_class_header(line)

        if i in method_rename:
            new_name = method_rename[i]
            line = re.sub(r"(\bdef\s+)[A-Za-z_]\w*(\s*\()", rf"\1{new_name}\2", line, count=1)

        if i in method_add_kwargs:
            # 現行 Activators の __forward__ は1行シグネチャ。
            # 既存の引数はそのままに、末尾へ **kwargs だけを追加する。
            close = line.rfind(")")
            if close < 0:
                raise RuntimeError(
                    f"multiline Activator __forward__ signature is not supported at line {i}"
                )
            line = line[:close] + ", **kwargs" + line[close:]

        # Function 継承を外した後の初期化呼出しは実行しない。
        # 行そのものを削除すると、それが唯一の文だった __init__ や if 節が
        # 空になって SyntaxError になるため、同じインデントの pass に置換する。
        if line_direct_function.get(i, False) and "super().__init__(" in line:
            indent = re.match(r"^(\s*)", line).group(1)
            line = indent + "pass  # Function.__init__ is not needed in ufiesia"

        # Function 継承を外した後に残る明示的な初期化も同様に pass 化する。
        if re.match(
            r"^\s*nucleus\.(?:Function|CompositFunction)\.__init__\(self\)\s*$",
            line,
        ):
            indent = re.match(r"^(\s*)", line).group(1)
            line = indent + "pass  # Function.__init__ is not needed in ufiesia"

        line = line.replace("super().__forward__(", "super().forward(")
        line = line.replace("super().__backward__(", "super().backward(")
        line = line.replace("snp.", "np.")
        # safe_np.add_at は NumPy/CuPy の ufunc.at に対応する。
        line = line.replace("np.add_at(", "np.add.at(")

        out.append(line)
        i += 1

    if method_add_kwargs:
        notes.append(
            f"Activator forward **kwargs compatibility added: {len(method_add_kwargs)}"
        )

    return "\n".join(out) + "\n", notes


def replace_class_block(text: str, class_name: str, replacement: str) -> str:
    tree = ast.parse(text)
    for cls in iter_class_defs(tree):
        if cls.name == class_name:
            lines = text.splitlines()
            a, b = cls.lineno - 1, cls.end_lineno
            repl = replacement.rstrip("\n").splitlines()
            return "\n".join(lines[:a] + repl + lines[b:]) + "\n"
    return text



def transform_config(text: str, notes: list[str]) -> str:
    """
    ufiesia の Config は define-and-run 用の最小構成に固定する。

    pyaino 側の Config から nucleus / define-by-run / automatic-differentiation
    関連設定を選別して残すのではなく、ufiesia0.Config と同等の内容を生成する。
    """
    notes.append("Config replaced with minimal ufiesia0-compatible definition")

    return """class Config:
    np    = None
    dtype = 'f4'
    seed  = None


def set_dtype(value):
    #print('old_value =', getattr(Config, 'dtype'))
    setattr(Config, 'dtype', value)
    print('Config.dtype is set to', Config.dtype)


def set_seed(value):
    #print('old_value =', getattr(Config, 'seed'))
    setattr(Config, 'seed', value)
    np.random.seed(seed=Config.seed)
    print('random.seed', Config.seed, 'is set for', np.__name__)


def set_np(value=None):
    global np
    #print('Config.np old_value =', getattr(Config, 'np'))

    if value is None:
        try:
            import cupy as np
        except:
            import numpy as np
    elif value == 'numpy':
        import numpy as np
    elif value == 'cupy':
        import cupy as np
    else:
        raise Exception("Invalid library specified. Specify either 'numpy' or 'cupy'.")

    if np.__name__ == 'numpy':
        np.seterr(divide='raise') # 割算例外でnanで続行せずに例外処理させる
        #np.seterr(over='raise')

    setattr(Config, 'np', np)


set_np()

print(np.__name__, 'is running in', __file__, np.random.rand(1))
print('Config.dtype =', Config.dtype)
print('Config.seed =', Config.seed)
print("If you want to change np, run 'set_np('numpy' or 'cupy'); np = Config.np.'")
print("If you want to change Config.dtype, run 'set_dtype('value')'")
print("If you want to set seed for np.random, run set_seed(number)")
"""

def transform_functions(text: str, notes: list[str]) -> str:
    erf = '''class Erf:
    """ 誤差関数(ガウスの誤差関数) """
    def __init__(self):
        if np.__name__ == 'cupy':
            from cupyx.scipy.special import erf
            self.erf = erf
        else:
            try:
                from scipy.special import erf
                self.erf = erf
            except ImportError:
                self.erf = self.AbramowitzStegun

    @staticmethod
    def AbramowitzStegun(x):
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p  = 0.3275911
        sign = np.sign(x)
        ax = np.abs(x)
        t = 1.0 / (1.0 + p * ax)
        poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
        y = 1.0 - poly * np.exp(-ax * ax)
        return sign * y

    def forward(self, x):
        self.x = x
        return self.erf(x)

    def backward(self, gy):
        x = self.x
        return gy * (2.0 / np.sqrt(np.pi)) * np.exp(-x * x)
'''
    new = replace_class_block(text, "Erf", erf)
    if new != text:
        text = new
        notes.append("Erf backend selection normalized")

    # 既知のsource typo。変換規則とは別扱い。
    text = text.replace("return SquareRoot().forward(x)", "return Sqrt().forward(x)")

    # 型が引数から渡されるためASTだけでは確定しない箇所。
    text = text.replace("self.func(*xs)", "self.func.forward(*xs)")
    text = text.replace("self.func(tuple(xs[0]))", "self.func.forward(*tuple(xs[0]))")
    text = text.replace("return f(x)", "return f.forward(x)")
    text = text.replace("self.primitive(x)", "self.primitive.forward(x)")

    # nucleus固有のsmoke testだけ取り除く。
    lines = text.splitlines()

    def cut_between(lines, start_pat, end_pat, replacement):
        try:
            s = next(i for i, x in enumerate(lines) if start_pat in x)
            e = next(i for i, x in enumerate(lines[s + 1:], s + 1) if end_pat in x)
        except StopIteration:
            return lines, False
        indent = re.match(r"^(\s*)", lines[s]).group(1)
        return lines[:s] + [indent + replacement] + lines[e:], True

    lines, ok = cut_between(
        lines,
        "基本関数の組み合わせのテスト2 backtrace",
        "そのほかの関数のテスト",
        "# define-by-run / nucleus 固有の backtrace テストは ufiesia では行わない。",
    )
    if ok:
        notes.append("define-by-run backtrace smoke test removed")

    lines, ok = cut_between(
        lines,
        "合成関数の検証",
        "テンソル操作の関数のテスト",
        "# CompositFunction / HDArray のテストは ufiesia では行わない。",
    )
    if ok:
        notes.append("CompositFunction/HDArray smoke tests removed")

    # HDArrayそのものに演算子を差し込む節はufiesiaでは不要。
    try:
        s = next(i for i, x in enumerate(lines) if "# OperatorOverload" in x)
        e = next(i for i, x in enumerate(lines[s + 1:], s + 1) if x.startswith("if __name__"))
        # 区切り線も含めて簡潔な注記に置換。
        start = max(0, s - 1)
        lines = lines[:start] + [
            "#######################################################",
            "# HDArray OperatorOverload は nucleus 固有のため ufiesia では持たない",
            "#######################################################",
            "",
        ] + lines[e:]
        notes.append("HDArray OperatorOverload section removed")
    except StopIteration:
        pass

    text = "\n".join(lines) + "\n"

    # smoke testでも上流勾配を明示する。
    # func.backward() はすべて直前の y に対する上流勾配1を与える。
    text = text.replace("func.backward()", "func.backward(np.ones_like(y))")
    text = re.sub(
        r"(?m)^(\s*)gx\s*=\s*func\.backward\(np\.ones_like\(y\)\)\s*$",
        r"\1gx = func.backward(np.ones_like(y))",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*)gx0,\s*gx1\s*=\s*func\.backward\(\)\s*$",
        r"\1gx0, gx1 = func.backward(np.ones_like(y))",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*)gxs\s*=\s*func\.backward\(\)\s*$",
        r"\1gxs = func.backward(np.ones_like(y))",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*)gx\s*=\s*func1\.backward\(\)\s*$",
        r"\1gys_in = tuple(np.ones_like(y) for y in ys)\n\1gx = func1.backward(*gys_in)",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*)gys\s*=\s*func2\.backward\(\)\s*$",
        r"\1gys = func2.backward(np.ones_like(z))",
        text,
    )
    return text


def transform_activators(text: str, notes: list[str]) -> str:
    # generic snp->np 後は np.erf になるため、backend helperへ戻す。
    text = text.replace("np.erf(", "_erf(")
    text = text.replace("cf.convert_one_hot(", "_convert_one_hot(")

    helper = '''
# backendに応じたerf。CuPyではcupyx、NumPyではSciPyまたは近似式を使う
def _erf(x):
    if np.__name__ == 'cupy':
        from cupyx.scipy.special import erf
        return erf(x)
    try:
        from scipy.special import erf
        return erf(x)
    except ImportError:
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p  = 0.3275911
        sign = np.sign(x)
        ax = np.abs(x)
        t = 1.0 / (1.0 + p * ax)
        poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
        y = 1.0 - poly * np.exp(-ax * ax)
        return sign * y


def _convert_one_hot(t, size):
    t = np.asarray(t, dtype=int)
    y = np.zeros(t.shape + (size,), dtype=Config.dtype)
    np.put_along_axis(y, t[..., None], 1.0, axis=-1)
    return y
'''
    marker = "#### 活性化関数"
    prefix = text.split(marker, 1)[0]
    if "def _convert_one_hot" not in prefix:
        pos = text.find(marker)
        if pos >= 0:
            text = text[:pos] + helper + "\n" + text[pos:]
            notes.append("backend erf / one-hot helper injected")
    return text



def transform_optimizers(text: str, notes: list[str]) -> str:
    # Optimizers末尾の f(x)=x^2 smoke test。ufiesiaでは上流勾配を明示する。
    text = text.replace("gx = func.backward()", "gx = func.backward(1)")
    return text

def transform_regularizers(text: str, notes: list[str]) -> str:
    text = text.replace("divergence(a)", "divergence.forward(a)")
    text = text.replace("regularize(result)", "regularize.forward(result)")
    return text


def transform_neuron(text: str, notes: list[str]) -> str:
    text = text.replace(
        "mi_loss = F.mean(log_qz_cond_x - log_pz, axis=-1)\n        return mi_loss",
        "return np.mean(log_qz_cond_x - log_pz, axis=-1)",
    )
    return text


def transform_common_function(text: str, notes: list[str]) -> str:
    lines = text.splitlines()
    out = []
    i = 0
    removed = 0
    while i < len(lines):
        line = lines[i]
        if "if type(obj) in (nucleus.HDArray, nucleus.XArray):" in line:
            if out and ("HDArray" in out[-1] or "XArray" in out[-1]):
                out.pop()
            i += 1
            if i < len(lines) and re.match(r"^\s*return(?:\s+None)?\s*$", lines[i]):
                i += 1
            removed += 1
            continue
        out.append(line)
        i += 1

    if removed:
        notes.append(f"HDArray/XArray traversal guards removed: {removed}")
    return "\n".join(out) + "\n"


MODULE_HOOKS = {
    "Config": transform_config,
    "Functions": transform_functions,
    "Activators": transform_activators,
    "Optimizers": transform_optimizers,
    "Regularizers": transform_regularizers,
    "Neuron": transform_neuron,
    "common_function": transform_common_function,
}


def convert_text(source: str, module_name: str) -> tuple[str, list[str]]:
    notes = []
    text, more = structural_transform(source, module_name)
    notes.extend(more)

    hook = MODULE_HOOKS.get(module_name)
    if hook:
        text = hook(text, notes)

    return text, notes


def audit_source(text: str, file_name: str) -> AuditResult:
    result = AuditResult(file=file_name)
    try:
        tree = ast.parse(text)
        compile(text, file_name, "exec")
    except SyntaxError as e:
        result.compile_ok = False
        result.issues.append(f"SyntaxError: {e}")
        return result

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "pyaino" or mod.startswith("pyaino."):
                result.issues.append(f"pyaino import remains at line {node.lineno}: {mod}")
            if mod == "ufiesia.nucleus" or mod.endswith(".nucleus"):
                result.issues.append(f"nucleus import remains at line {node.lineno}")

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pyaino" or alias.name.startswith("pyaino."):
                    result.issues.append(f"pyaino import remains at line {node.lineno}: {alias.name}")

        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = dotted_name(base)
                if name in FUNCTION_BASES:
                    result.issues.append(
                        f"Function/CompositFunction base remains: {node.name} line {node.lineno}"
                    )

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in {"__forward__", "__backward__"}:
                result.issues.append(f"{node.name} remains at line {node.lineno}")

        if isinstance(node, ast.Name) and node.id in {
            "nucleus", "HDArray", "XArray", "CompositFunction", "Function", "snp",
        }:
            result.issues.append(f"pyaino-specific name '{node.id}' remains at line {node.lineno}")

        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "nucleus"
        ):
            result.issues.append(f"nucleus.{node.attr} remains at line {node.lineno}")

    result.issues = list(dict.fromkeys(result.issues))
    return result


def _literal_assignment(init_text: str, name: str, init_file: str):
    """__init__.py のリテラル代入を package import なしで取得する。"""
    tree = ast.parse(init_text, filename=init_file)
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                value = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                value = node.value
        if value is not None:
            try:
                return ast.literal_eval(value)
            except Exception as e:
                raise RuntimeError(
                    f"{name} in {init_file} must be a literal value."
                ) from e
    raise RuntimeError(f"{name} is not defined in {init_file}.")


def read_current_modules(src_dir: Path) -> tuple[str, ...]:
    """pyaino/__init__.py の CURRENT_MODULES を安全に読む。"""
    init_file = src_dir / "__init__.py"
    if not init_file.exists():
        raise RuntimeError(f"pyaino __init__.py not found: {init_file}")

    init_text = read_source(init_file)
    modules = _literal_assignment(init_text, "CURRENT_MODULES", str(init_file))
    if not isinstance(modules, (tuple, list)):
        raise RuntimeError("CURRENT_MODULES must be a tuple or list.")

    result = []
    seen = set()
    for module in modules:
        if not isinstance(module, str) or not module.strip():
            raise RuntimeError(f"Invalid CURRENT_MODULES entry: {module!r}")
        module = module.strip()
        if module.endswith(".py"):
            module = module[:-3]
        if module in seen:
            raise RuntimeError(f"Duplicate CURRENT_MODULES entry: {module}")
        seen.add(module)
        result.append(module)
    return tuple(result)


def module_source(src_dir: Path, module: str) -> Path | None:
    rel = Path(*module.split("."))
    py_file = src_dir / rel.with_suffix(".py")
    if py_file.exists():
        return py_file
    package_init = src_dir / rel / "__init__.py"
    if package_init.exists():
        return package_init
    return None


def module_destination(dst_dir: Path, module: str, source: Path) -> Path:
    rel = Path(*module.split("."))
    if source.name == "__init__.py":
        return dst_dir / rel / "__init__.py"
    return dst_dir / rel.with_suffix(".py")


def generated_init_text(src_dir: Path, converted_modules: list[str]) -> str:
    """生成 ufiesia の実際の構成に合わせた __init__.py を作る。"""
    src_init = src_dir / "__init__.py"
    src_text = read_source(src_init)

    try:
        version = _literal_assignment(src_text, "__version__", str(src_init))
    except RuntimeError:
        version = None
    try:
        public = _literal_assignment(src_text, "__all__", str(src_init))
    except RuntimeError:
        public = ()

    converted_top = {m.split(".")[0] for m in converted_modules}
    public = tuple(
        name for name in public
        if isinstance(name, str) and name in converted_top
    )

    lines = [
        '"""',
        "ufiesia: define-and-run reference framework generated from pyaino.",
        '"""',
        "",
    ]
    if version is not None:
        lines += [f"__version__ = {version!r}", ""]

    lines.append("__all__ = (")
    lines += [f"    {name!r}," for name in public]
    lines += [")", ""]

    lines.append("CURRENT_MODULES = (")
    lines += [f"    {module!r}," for module in converted_modules]
    lines += [")", ""]
    return "\n".join(lines)


def default_pyaino_dir() -> Path:
    """変換器自身が pyaino 内にあれば、その package を変換元にする。"""
    here = Path(__file__).resolve().parent
    if here.name == "pyaino" and (here / "__init__.py").exists():
        return here
    return discover_pyaino()


def discover_pyaino() -> Path:
    spec = importlib.util.find_spec("pyaino")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("pyaino package not found. Specify source directory explicitly.")
    return Path(next(iter(spec.submodule_search_locations)))


def convert_file(src: Path, dst: Path, module_name: str | None = None) -> ConvertResult:
    stem = (module_name or src.stem).split(".")[-1]
    if stem in UFIESIA_EXCLUDE:
        return ConvertResult(str(src), str(dst), "skip")

    source = read_source(src)
    text, notes = convert_text(source, stem)
    write_state = write_source(dst, text)
    audit = audit_source(text, str(dst))
    return ConvertResult(
        str(src), str(dst), "convert", audit, notes, write_state=write_state
    )


def convert_package(src_dir: Path, dst_dir: Path, clean=False) -> list[ConvertResult]:
    """CURRENT_MODULES に列挙された正式モジュールだけを変換する。"""
    modules = read_current_modules(src_dir)

    if clean and dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    results = []
    converted_modules = []

    for module in modules:
        source = module_source(src_dir, module)
        if source is None:
            results.append(ConvertResult(
                str(src_dir / (module + ".py")),
                "",
                "missing",
                notes=[f"CURRENT_MODULES entry not found: {module}"],
            ))
            continue

        stem = module.split(".")[-1]
        if stem in UFIESIA_EXCLUDE:
            results.append(ConvertResult(
                str(source),
                "",
                "skip",
                notes=[f"excluded from ufiesia: {module}"],
            ))
            continue

        destination = module_destination(dst_dir, module, source)
        result = convert_file(source, destination, module_name=stem)
        results.append(result)
        if result.action == "convert":
            converted_modules.append(module)

    init_dst = dst_dir / "__init__.py"
    init_text = generated_init_text(src_dir, converted_modules)
    init_state = write_source(init_dst, init_text)
    init_audit = audit_source(init_text, str(init_dst))
    results.append(ConvertResult(
        str(src_dir / "__init__.py"),
        str(init_dst),
        "generated",
        init_audit,
        notes=["generated package initializer"],
        write_state=init_state,
    ))
    return results


def print_report(results: list[ConvertResult]) -> bool:
    ok = True
    print("\n=== pyaino -> ufiesia conversion report ===")
    for r in results:
        if r.action == "skip":
            print(f"SKIP  {Path(r.src).name}")
            for note in r.notes:
                print("      -", note)
            continue

        if r.action == "missing":
            ok = False
            print(f"MISSING {Path(r.src).name}")
            for note in r.notes:
                print("      !", note)
            continue

        if r.audit and r.audit.ok:
            status = {
                "new": "NEW",
                "update": "UPDATE",
                "same": "SAME",
            }.get(r.write_state, "OK")
        else:
            status = "CHECK"

        print(f"{status:6s} {Path(r.dst).name}")
        for note in r.notes:
            print("       -", note)
        if r.audit and r.audit.issues:
            ok = False
            for issue in r.audit.issues:
                print("       !", issue)
    print("=== end ===")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Convert pyaino CURRENT_MODULES into a define-and-run ufiesia tree."
    )
    parser.add_argument(
        "src", nargs="?",
        help="pyaino package directory. If omitted, use this script's pyaino package or installed pyaino.",
    )
    parser.add_argument(
        "dst", nargs="?", default=None,
        help=("staging root. If omitted, create "
              "<pyaino parent>/ufiesia_generated/ufiesia"),
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--strict", action="store_true",
        help="exit non-zero if audit finds unresolved pyaino-specific code",
    )
    args = parser.parse_args()

    src_dir = Path(args.src).resolve() if args.src else default_pyaino_dir()

    if args.dst is None:
        staging_root = src_dir.parent / "ufiesia_generated"
    else:
        staging_root = Path(args.dst).resolve()

    dst_dir = staging_root / "ufiesia"

    print("source      :", src_dir)
    print("staging root:", staging_root)
    print("package     :", dst_dir)

    # --clean removes the whole staging root, not only the package directory.
    if args.clean and staging_root.exists():
        shutil.rmtree(staging_root)

    results = convert_package(src_dir, dst_dir, clean=False)
    ok = print_report(results)

    if args.strict and not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
