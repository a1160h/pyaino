"""pyaino.numpy_min

Build a NumPy-only, inference-oriented minimum pyaino package for one target program.

The reducer combines static AST dependency analysis with an optional runtime trace.
The public API is intentionally small:

    trace_usage(...)
        Run the target once and record the pyaino classes/functions actually used.

    build_numpy_min(...)
        Extract only the required pyaino definitions, strip training-only methods,
        and copy the target plus explicitly listed runtime resources.

This module contains no application-specific settings.  A concrete application
should import these functions from a separate driver program.
"""

from pathlib import Path
from collections import defaultdict, deque
import argparse
import ast
import importlib.util
import shutil
import tempfile
import json
import runpy
import sys
import os


# Training-side methods removed from an inference-only package by default.
TRAINING_METHODS = {
    'backward', '__backward__', 'update', 'step',
    'cleargrad', 'cleargrads', 'clear_grad', 'clear_grads',
    'zerograd', 'zero_grad', 'zero_grads',
    'set_grad', 'set_grads', 'get_grad', 'get_grads', 'fix_grads',
    'generate_dot_graph', '_dot_var', 'backtrace',
    'backup', 'recover', 'set_gradient', 'flush_gradient', 'accommodate',
}


# Config is small and performs backend initialization at import time.
# Keeping it intact is safer than slicing it in this experimental version.
FULL_COPY_MODULES = {'Config'}
MODULE_ONLY = '@module'

# safe_np contains backend-conditional definitions, which are deliberately
# replaced by a tiny NumPy-only compatibility layer in the minimized pack.
SAFE_NP_MODULE = 'safe_np'
SAFE_NP_SPECIAL = {
    'add_at', 'erf',
}


def installed_pyaino_dir():
    spec = importlib.util.find_spec('pyaino')
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def current_pyaino_dir():
    """Return the pyaino package directory that contains this module."""
    return Path(__file__).resolve().parent


def module_source(source_dir, module):
    if module == '__init__':
        p = source_dir / '__init__.py'
        return p if p.exists() else None
    rel = Path(*module.split('.'))
    p = source_dir / rel.with_suffix('.py')
    if p.exists():
        return p
    p = source_dir / rel / '__init__.py'
    return p if p.exists() else None


def module_destination(package_dir, module, src):
    if module == '__init__':
        return package_dir / '__init__.py'
    rel = Path(*module.split('.'))
    if src.name == '__init__.py':
        return package_dir / rel / '__init__.py'
    return package_dir / rel.with_suffix('.py')


def assigned_names(node):
    names = []
    def add_target(t):
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)
    if isinstance(node, ast.Assign):
        for t in node.targets:
            add_target(t)
    elif isinstance(node, ast.AnnAssign):
        add_target(node.target)
    elif isinstance(node, ast.AugAssign):
        add_target(node.target)
    return names


def is_main_guard(node):
    if not isinstance(node, ast.If):
        return False
    t = node.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == '__name__'
            and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq) and len(t.comparators) == 1
            and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == '__main__')


class ModuleInfo:
    def __init__(self, module, path, source_dir):
        self.module = module
        self.path = path
        self.source_dir = source_dir
        self.source = path.read_text(encoding='utf-8-sig')
        self.tree = ast.parse(self.source, filename=str(path))
        self.defs = {}
        self.assignments = {}
        self.module_aliases = {}   # local name -> pyaino module
        self.symbol_aliases = {}   # local name -> (pyaino module, symbol)
        self.star_modules = []
        self.pyaino_import_nodes = []
        self.non_pyaino_import_nodes = []
        self.preserved_top = []
        self._index()

    def _index(self):
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.defs[node.name] = node
                continue

            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for name in assigned_names(node):
                    self.assignments[name] = node
                self.preserved_top.append(node)
                continue

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if self._index_pyaino_import(node):
                    self.pyaino_import_nodes.append(node)
                else:
                    self.non_pyaino_import_nodes.append(node)
                continue

            if is_main_guard(node):
                continue
            self.preserved_top.append(node)

    def _index_pyaino_import(self, node):
        if isinstance(node, ast.Import):
            hit = False
            for a in node.names:
                if a.name == 'pyaino':
                    local = a.asname or 'pyaino'
                    self.module_aliases[local] = '__init__'
                    hit = True
                elif a.name.startswith('pyaino.'):
                    module = a.name[len('pyaino.'):]
                    local = a.asname or a.name.split('.')[-1]
                    self.module_aliases[local] = module
                    hit = True
            return hit

        mod = node.module or ''
        if mod == 'pyaino':
            for a in node.names:
                if a.name == '*':
                    self.star_modules.append('__init__')
                    continue
                # In "from pyaino import Neuron", Neuron is normally a submodule.
                if module_source(self.source_dir, a.name) is not None:
                    self.module_aliases[a.asname or a.name] = a.name
                else:
                    self.symbol_aliases[a.asname or a.name] = ('__init__', a.name)
            return True

        if mod.startswith('pyaino.'):
            module = mod[len('pyaino.'):]
            for a in node.names:
                if a.name == '*':
                    self.star_modules.append(module)
                else:
                    self.symbol_aliases[a.asname or a.name] = (module, a.name)
            return True
        return False

    @property
    def local_names(self):
        return set(self.defs) | set(self.assignments)


class DependencyVisitor(ast.NodeVisitor):
    def __init__(self, info):
        self.info = info
        self.local = set()
        self.external = set()  # (module, symbol), symbol='*' means module-wide
        self.used_names = set()

    def visit_Attribute(self, node):
        root, attrs = self._chain(node)
        if root in self.info.module_aliases and attrs:
            self.external.add((self.info.module_aliases[root], attrs[0]))
            # still visit index/call arguments elsewhere through parent traversal;
            # there is nothing else inside this Attribute except the root chain.
            return
        self.generic_visit(node)

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            return
        name = node.id
        self.used_names.add(name)
        if name in self.info.module_aliases:
            self.external.add((self.info.module_aliases[name], '*'))
        elif name in self.info.symbol_aliases:
            self.external.add(self.info.symbol_aliases[name])
        elif name in self.info.local_names:
            self.local.add(name)
        else:
            # Resolve names supplied by a star import when possible.
            for module in self.info.star_modules:
                src = module_source(self.info.source_dir, module)
                if src is None:
                    continue
                try:
                    mi = get_info(module, self.info.source_dir)
                except Exception:
                    continue
                if name in mi.local_names:
                    self.external.add((module, name))
                    break

    def visit_Import(self, node):
        for a in node.names:
            if a.name == 'pyaino':
                self.external.add(('__init__', MODULE_ONLY))
            elif a.name.startswith('pyaino.'):
                self.external.add((a.name[len('pyaino.'):], MODULE_ONLY))

    def visit_ImportFrom(self, node):
        mod = node.module or ''
        if mod == 'pyaino':
            for a in node.names:
                if a.name == '*':
                    self.external.add(('__init__', MODULE_ONLY))
                elif module_source(self.info.source_dir, a.name) is not None:
                    self.external.add((a.name, MODULE_ONLY))
                else:
                    self.external.add(('__init__', a.name))
        elif mod.startswith('pyaino.'):
            module = mod[len('pyaino.'):]
            for a in node.names:
                self.external.add((module, '*' if a.name == '*' else a.name))

    @staticmethod
    def _chain(node):
        attrs = []
        cur = node
        while isinstance(cur, ast.Attribute):
            attrs.append(cur.attr)
            cur = cur.value
        attrs.reverse()
        return (cur.id if isinstance(cur, ast.Name) else None), attrs


_INFO_CACHE = {}


def get_info(module, source_dir):
    key = (str(Path(source_dir).resolve()), module)
    if key in _INFO_CACHE:
        return _INFO_CACHE[key]
    src = module_source(Path(source_dir), module)
    if src is None:
        raise FileNotFoundError(module)
    info = ModuleInfo(module, src, Path(source_dir))
    _INFO_CACHE[key] = info
    return info


def make_forward_function_class(node):
    """Replace nucleus.Function by an inference-only dispatcher."""
    src = '''
class Function:
    def __init__(self, log=False, log_file='log_file.txt', preserve_attr=False):
        self.inputs = None
        self.outputs = None
        self.generation = 0
        if not hasattr(self, 'config'):
            self.config = None
        self.preserve_attr = preserve_attr

    def forward(self, *inputs, **kwargs):
        return self.__forward__(*inputs, **kwargs)

    def __forward__(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
'''
    return ast.parse(src).body[0]


def make_forward_composit_class(node):
    src = '''
class CompositFunction:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward(self, *inputs, **kwargs):
        return self._forward(*inputs, **kwargs)

    def _forward(self, *inputs, **kwargs):
        raise NotImplementedError()

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)
'''
    return ast.parse(src).body[0]




def contains_name(node, names):
    return any(isinstance(x, ast.Name) and x.id in names for x in ast.walk(node))


def assignment_attr_names(node):
    out = []
    targets = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    for t in targets:
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self':
            out.append(t.attr)
    return out


class InferenceInitTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        node = self.generic_visit(node)
        attrs = assignment_attr_names(node)
        if attrs and any(a.startswith('optimizer_') or a.startswith('OF') for a in attrs):
            if contains_name(node, {'Optimizers'}) or contains_name(node, {'cf'}):
                return None
        return node

    def visit_AnnAssign(self, node):
        node = self.generic_visit(node)
        attrs = assignment_attr_names(node)
        if attrs and any(a.startswith('optimizer_') or a.startswith('OF') for a in attrs):
            if contains_name(node, {'Optimizers'}) or contains_name(node, {'cf'}):
                return None
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        node.body = [x for x in node.body if x is not None]
        node.orelse = [x for x in node.orelse if x is not None]
        if not node.body and not node.orelse:
            return None
        if not node.body:
            node.body = [ast.Pass()]
        return node


def strip_training_initialization(method, class_name):
    """Remove constructor state used only by optimizer/regularizer machinery."""
    if method.name != '__init__':
        return method

    method = InferenceInitTransformer().visit(method)
    ast.fix_missing_locations(method)

    if class_name not in {'AttentionUnit', 'AttentionUnit_bkup'}:
        return method

    body = []
    skip_regularizer_if = False
    for stmt in method.body:
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'regularizer' for t in stmt.targets):
            body.append(ast.Assign(
                targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='regularizer', ctx=ast.Store())],
                value=ast.Constant(value=None)))
            skip_regularizer_if = True
            continue
        if skip_regularizer_if and isinstance(stmt, ast.If):
            names = {x.id for x in ast.walk(stmt) if isinstance(x, ast.Name)}
            if 'regularizer' in names:
                skip_regularizer_if = False
                continue
        skip_regularizer_if = False
        body.append(stmt)

    method.body = body or [ast.Pass()]
    ast.fix_missing_locations(method)
    return method


def make_update_marker():
    # common_function.import_parameters_recursive() historically uses
    # hasattr(obj, 'update') as the test for a parameter-bearing layer.
    # Keep that structural marker without retaining any training machinery.
    return ast.parse('''
def update(self, *args, **kwargs):
    pass
''').body[0]


def prune_class(module, node, keep_methods):
    if module == 'nucleus' and node.name == 'Function':
        return make_forward_function_class(node)
    if module == 'nucleus' and node.name == 'CompositFunction':
        return make_forward_composit_class(node)

    original_methods = {
        item.name for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    body = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (item.name in TRAINING_METHODS or item.name.endswith('_bkup')) and item.name not in keep_methods:
                continue
            if isinstance(item, ast.FunctionDef):
                item = strip_training_initialization(item, node.name)
        body.append(item)

    # Parameter loading needs the existence of update() as a layer marker.
    # A no-op stub preserves that convention while the pack remains inference-only.
    if 'update' in original_methods and 'update' not in keep_methods:
        body.append(make_update_marker())

    if not body:
        body = [ast.Pass()]
    new = ast.ClassDef(name=node.name, bases=node.bases, keywords=node.keywords,
                       body=body, decorator_list=node.decorator_list)
    if hasattr(node, 'type_params'):
        new.type_params = node.type_params
    return ast.copy_location(new, node)


def transformed_def(module, node, keep_methods):
    if isinstance(node, ast.ClassDef):
        return prune_class(module, node, keep_methods)
    return node


def scan_node(node, info):
    v = DependencyVisitor(info)
    v.visit(node)
    return v


def roots_from_target(target, source_dir):
    source = target.read_text(encoding='utf-8-sig')
    tree = ast.parse(source, filename=str(target))
    # Build import bindings by parsing the target as if it were a module.
    pseudo_path = target
    info = ModuleInfo('<target>', pseudo_path, source_dir)
    v = scan_node(tree, info)
    return v.external


def collect_symbols(target, source_dir, keep_methods, traced_defs=None):
    traced_defs = traced_defs or {}
    required = defaultdict(set)
    selected_defs = defaultdict(set)
    selected_assignments = defaultdict(set)
    required_by = defaultdict(set)
    missing = defaultdict(set)
    activated = set()
    pending = deque()

    def require(module, symbol, parent):
        key = (module, symbol)

        # A wildcard may later be narrowed by runtime trace.  Therefore it must
        # not suppress an explicit dependency discovered from a selected symbol.
        # Example:
        #   Activators:*  --trace--> {Mish, Softmax}
        #   Mish -> ActivatorBase
        # The explicit ActivatorBase requirement still has to enter the queue.
        if symbol == '*':
            if '*' in required[module]:
                required_by[key].add(parent)
                return
            required[module].add('*')
        else:
            if symbol in required[module]:
                required_by[key].add(parent)
                return
            required[module].add(symbol)

        required_by[key].add(parent)
        pending.append((module, symbol, parent))

    for module, symbol in roots_from_target(target, source_dir):
        require(module, symbol, '<target>')

    while pending:
        module, symbol, parent = pending.popleft()
        src = module_source(source_dir, module)
        if src is None:
            missing[(module, symbol)].add(parent)
            continue
        info = get_info(module, source_dir)

        if module not in activated:
            activated.add(module)

        # safe_np is emitted as a NumPy-only shim. Its public functions are
        # conditionally defined in the original source, so ordinary top-level
        # symbol indexing is intentionally bypassed here.
        if module == SAFE_NP_MODULE:
            continue

        if module in FULL_COPY_MODULES:
            # Its own source is copied intact; only discover pyaino deps if any.
            v = scan_node(info.tree, info)
            for depmod, depsym in v.external:
                if depmod != module:
                    require(depmod, depsym, module + ':*')
            continue

        if symbol == MODULE_ONLY:
            continue

        if symbol == '*':
            traced = set(traced_defs.get(module, ()))
            available = set(info.defs) | set(info.assignments)
            # Runtime trace resolves dynamic module lookups such as
            # eval_in_module(name, Activators).  If no trace exists for a
            # wildcard module, retain the old conservative behaviour.
            symbols = sorted((traced & available) or {n for n in available if not n.endswith('_bkup')})
        else:
            symbols = [symbol]
        for name in symbols:
            if name in info.defs:
                if name in selected_defs[module]:
                    continue
                selected_defs[module].add(name)
                node = transformed_def(module, info.defs[name], keep_methods)
                v = scan_node(node, info)
                for local in v.local:
                    require(module, local, f'{module}.{name}')
                for depmod, depsym in v.external:
                    require(depmod, depsym, f'{module}.{name}')

            elif name in info.assignments:
                if name in selected_assignments[module]:
                    continue
                selected_assignments[module].add(name)
                node = info.assignments[name]
                v = scan_node(node, info)
                for local in v.local:
                    require(module, local, f'{module}.{name}')
                for depmod, depsym in v.external:
                    require(depmod, depsym, f'{module}.{name}')

            elif name in info.symbol_aliases:
                depmod, depsym = info.symbol_aliases[name]
                require(depmod, depsym, f'{module}.{name}')

            elif name in info.module_aliases:
                require(info.module_aliases[name], '*', f'{module}.{name}')

            else:
                # It may be provided by a star import/re-export.
                resolved = False
                for smod in info.star_modules:
                    ssrc = module_source(source_dir, smod)
                    if ssrc is None:
                        continue
                    sinfo = get_info(smod, source_dir)
                    if name in sinfo.local_names:
                        require(smod, name, f'{module}.{name}')
                        resolved = True
                        break
                if not resolved:
                    missing[(module, name)].add(parent)

    return selected_defs, selected_assignments, required, missing, required_by, activated


def names_loaded(nodes):
    names = set()
    for node in nodes:
        for x in ast.walk(node):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                names.add(x.id)
    return names


def rewrite_pyaino_import(node, used_names, source_dir):
    if isinstance(node, ast.Import):
        kept = []
        for a in node.names:
            if a.name == 'pyaino' or a.name.startswith('pyaino.'):
                local = a.asname or ('pyaino' if a.name == 'pyaino' else a.name.split('.')[-1])
                if local in used_names:
                    kept.append(a)
            else:
                kept.append(a)
        return ast.Import(names=kept) if kept else None

    mod = node.module or ''
    if not (mod == 'pyaino' or mod.startswith('pyaino.')):
        return node
    if any(a.name == '*' for a in node.names):
        return node
    kept = []
    for a in node.names:
        local = a.asname or a.name
        if local in used_names:
            kept.append(a)
    if not kept:
        return None
    return ast.ImportFrom(module=node.module, names=kept, level=node.level)


def import_bound_names(node):
    """Local names bound by an import statement ('*' for a star import)."""
    names = []
    for a in node.names:
        if a.name == '*':
            names.append('*')
        elif isinstance(node, ast.Import):
            # "import os.path" binds "os"; "import x.y as z" binds "z".
            names.append(a.asname or a.name.split('.')[0])
        else:
            names.append(a.asname or a.name)
    return names


def rewrite_plain_import(node, used_names):
    """Drop non-pyaino imports no longer referenced by the reduced module.

    Stripping training/plotting definitions can leave their imports behind,
    e.g. matplotlib in BigramLanguageModel once graph_plus() is removed.
    """
    bound = import_bound_names(node)
    if '*' in bound:
        # A star import may provide names this module still relies on.
        return node
    kept = [a for a, local in zip(node.names, bound) if local in used_names]
    if not kept:
        return None
    if len(kept) == len(node.names):
        return node
    if isinstance(node, ast.Import):
        return ast.Import(names=kept)
    return ast.ImportFrom(module=node.module, names=kept, level=node.level)


def build_reduced_tree(module, info, selected_defs, selected_assignments, keep_methods):
    if module in FULL_COPY_MODULES:
        return None

    kept_defs = selected_defs.get(module, set())
    kept_assign = selected_assignments.get(module, set())
    selected_nodes = []

    for node in info.tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in kept_defs:
            selected_nodes.append(transformed_def(module, node, keep_methods))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if any(n in kept_assign for n in assigned_names(node)):
                selected_nodes.append(node)

    used = names_loaded(selected_nodes)
    body = []

    # Keep a module docstring first.
    if info.tree.body and isinstance(info.tree.body[0], ast.Expr) and isinstance(info.tree.body[0].value, ast.Constant) \
            and isinstance(info.tree.body[0].value.value, str):
        body.append(info.tree.body[0])

    # Both ordinary and pyaino imports are filtered down to what the reduced
    # module actually references.
    for node in info.non_pyaino_import_nodes:
        r = rewrite_plain_import(node, used)
        if r is not None:
            body.append(r)
    for node in info.pyaino_import_nodes:
        r = rewrite_pyaino_import(node, used, info.source_dir)
        if r is not None:
            body.append(r)

    # Add selected setup/definitions in original order, excluding imports/docstring/main guard.
    for node in info.tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or is_main_guard(node):
            continue
        if body and node is body[0]:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in kept_defs:
                body.append(transformed_def(module, node, keep_methods))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if any(n in kept_assign for n in assigned_names(node)):
                body.append(node)

    if not body:
        body = [ast.Pass()]
    tree = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(tree)
    return tree


def write_safe_np(dst, requested):
    """Emit only the safe_np names required by this NumPy-only inference pack."""
    names = sorted(n for n in requested if n not in {'*', MODULE_ONLY})
    lines = [
        '# generated NumPy-only safe_np subset',
        'import numpy as np',
        'import math as _math',
        '',
    ]
    for name in names:
        if name == 'add_at':
            lines.append('add_at = np.add.at')
        elif name == 'erf':
            lines += [
                'def erf(x):',
                "    return np.vectorize(_math.erf, otypes=[float])(x)",
            ]
        else:
            lines.append(f'{name} = np.{name}')
    if not names:
        lines.append('pass')
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_module(module, info, dst, selected_defs, selected_assignments, keep_methods, required=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if module == SAFE_NP_MODULE:
        write_safe_np(dst, (required or {}).get(module, set()))
        return info.path.stat().st_size, dst.stat().st_size
    if module in FULL_COPY_MODULES:
        shutil.copy2(info.path, dst)
        return info.path.stat().st_size, dst.stat().st_size
    tree = build_reduced_tree(module, info, selected_defs, selected_assignments, keep_methods)
    text = ast.unparse(tree) + '\n'
    dst.write_text(text, encoding='utf-8')
    return info.path.stat().st_size, dst.stat().st_size


def dir_py_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*.py'))


def _pack(target, source_dir, output_dir, keep_methods=(), traced_defs=None):
    target = Path(target).resolve()
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    package_dir = output_dir / 'pyaino'
    keep_methods = set(keep_methods)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    package_dir.mkdir(parents=True)

    selected_defs, selected_assignments, required, missing, required_by, activated = \
        collect_symbols(target, source_dir, keep_methods, traced_defs)

    shutil.copy2(target, output_dir / target.name)

    # A deliberately empty package initializer avoids importing unrelated pyaino modules.
    (package_dir / '__init__.py').write_text(
        '# generated inference-only pyaino subset; no installed pyaino required\n', encoding='utf-8')

    sizes = []
    for module in sorted(activated):
        if module == '__init__':
            continue
        src = module_source(source_dir, module)
        if src is None:
            continue
        info = get_info(module, source_dir)
        dst = module_destination(package_dir, module, src)
        before, after = write_module(module, info, dst, selected_defs, selected_assignments, keep_methods, required)
        sizes.append((module, before, after))

    lines = [
        f'target: {target}',
        f'source: {source_dir}',
        '',
        'reduced modules:',
    ]
    for module, before, after in sizes:
        pct = 100.0 * after / before if before else 0.0
        defs = ', '.join(sorted(selected_defs.get(module, ()))) or '-'
        lines.append(f'  {module}: {before:,} -> {after:,} bytes ({pct:5.1f}%)')
        if module not in FULL_COPY_MODULES:
            lines.append(f'    defs: {defs}')

    before_total = sum(x[1] for x in sizes)
    after_total = dir_py_size(package_dir)
    lines += [
        '',
        f'pyaino source total: {before_total:,} -> {after_total:,} bytes',
        f'reduction: {(1-after_total/before_total)*100:.1f}%' if before_total else 'reduction: n/a',
    ]

    if missing:
        lines += ['', 'unresolved symbols/modules:']
        for (module, symbol), parents in sorted(missing.items()):
            lines.append(f'  {module}:{symbol} <- {", ".join(sorted(parents))}')
    else:
        lines += ['', 'all selected dependencies resolved.']

    if traced_defs:
        lines += ['', 'runtime trace assisted: yes']
        for module in sorted(traced_defs):
            lines.append(f'  {module}: {", ".join(sorted(traced_defs[module]))}')
    else:
        lines += ['', 'runtime trace assisted: no']

    lines += ['', 'forward-only policy:']
    lines.append('  stripped methods: ' + ', '.join(sorted(TRAINING_METHODS - keep_methods)))
    if keep_methods:
        lines.append('  explicitly kept: ' + ', '.join(sorted(keep_methods)))
    lines.append('  update() is retained only as a no-op structural marker for parameter loading.')
    lines.append('  safe_np is replaced by a NumPy-only shim containing only referenced names.')
    lines.append('  nucleus.Function / CompositFunction replaced by inference-only dispatchers.')

    manifest = '\n'.join(lines) + '\n'
    (output_dir / 'PACKING_MIN.txt').write_text(manifest, encoding='utf-8')
    print(manifest, end='')
    return missing



def _resolve_resources(resources, target_dir):
    """Resolve and validate runtime resource specifications.

    Returns ``[(source_absolute_path, destination_relative_path), ...]``.
    Resolution happens *before* the output directory is replaced, so resources
    may safely be taken from a previous generated package as well.
    """
    resolved = []
    target_dir = Path(target_dir).resolve()

    for item in resources or ():
        if isinstance(item, (tuple, list)):
            if len(item) != 2:
                raise ValueError('resource tuple/list must be (source, destination_relative_path)')
            raw_src, raw_dst = item
            raw_src = Path(raw_src)
            src = raw_src if raw_src.is_absolute() else target_dir / raw_src
            dst_rel = Path(raw_dst)
        else:
            raw_src = Path(item)
            src = raw_src if raw_src.is_absolute() else target_dir / raw_src
            if raw_src.is_absolute():
                try:
                    dst_rel = src.resolve().relative_to(target_dir)
                except ValueError:
                    dst_rel = Path(src.name)
            else:
                dst_rel = raw_src

        src = src.resolve()
        if not src.exists():
            raise FileNotFoundError(f'runtime resource not found: {src}')
        if dst_rel.is_absolute() or '..' in dst_rel.parts:
            raise ValueError(f'resource destination must stay inside output_dir: {dst_rel}')
        resolved.append((src, dst_rel))

    return resolved


def _path_is_inside(path, directory):
    path = Path(path).resolve()
    directory = Path(directory).resolve()
    return path == directory or directory in path.parents


def _stage_overlapping_resources(resolved_resources, output_dir, stage_dir):
    """Stage resources that live inside ``output_dir`` before it is replaced.

    This matters when a previous ``numpy_package_min`` is used as the source of
    tokenizer/checkpoint files for rebuilding itself.  Without staging, the
    reducer would delete the source files together with the old output tree.
    """
    output_dir = Path(output_dir).resolve()
    stage_dir = Path(stage_dir).resolve()
    staged = []

    for i, (src, dst_rel) in enumerate(resolved_resources):
        if _path_is_inside(src, output_dir):
            safe_name = f'{i:03d}_{src.name}'
            tmp = stage_dir / safe_name
            if src.is_dir():
                shutil.copytree(src, tmp)
            else:
                tmp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, tmp)
            staged.append((tmp, dst_rel))
        else:
            staged.append((src, dst_rel))

    return staged


def _copy_resolved_resources(resolved_resources, output_dir):
    """Copy already-resolved runtime resources into the generated package."""
    copied = []
    output_dir = Path(output_dir).resolve()

    for src, dst_rel in resolved_resources:
        src = Path(src).resolve()
        dst_rel = Path(dst_rel)
        dst = output_dir / dst_rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied.append((src, dst_rel))

    return copied


def build_numpy_min(target, output_dir='numpy_package_min', source_dir=None,
                    trace=None, keep_methods=(), resources=()):
    """Build a minimum NumPy-oriented pyaino inference package.

    Parameters
    ----------
    target : path-like
        One executable Python program that uses pyaino.
    output_dir : path-like
        Destination directory.  An existing directory is replaced.
    source_dir : path-like or None
        Source pyaino package.  ``None`` means the pyaino package containing
        this ``numpy_min`` module.
    trace : path-like, mapping, or None
        ``PYAINO_TRACE.json`` created by :func:`trace_usage`, or an already
        loaded ``{module: set(symbols)}`` mapping.  ``None`` uses static
        analysis only.
    keep_methods : iterable[str]
        Training-side method names that must exceptionally be kept.
    resources : iterable[path-like | (source, destination)]
        Runtime data files/directories to copy.  Relative paths are resolved
        from the target program directory and keep the same relative path.

    Returns
    -------
    dict-like
        Unresolved symbol/module report returned by the reducer.
    """
    target = Path(target).resolve()
    output_dir = Path(output_dir).resolve()
    source_dir = current_pyaino_dir() if source_dir is None else Path(source_dir).resolve()

    if not target.exists():
        raise FileNotFoundError(f'target program not found: {target}')
    if not source_dir.exists():
        raise FileNotFoundError(f'pyaino source directory not found: {source_dir}')

    if trace is None:
        traced_defs = None
    elif isinstance(trace, (str, Path)):
        traced_defs = load_trace(trace)
    else:
        traced_defs = {m: set(v) for m, v in trace.items()}

    # Resolve resource sources before replacing output_dir.  Some workflows use
    # a previous numpy_package_min itself as the source of runtime data.
    resolved_resources = _resolve_resources(resources, target.parent)

    with tempfile.TemporaryDirectory(prefix='pyaino_numpy_min_') as tmpdir:
        prepared_resources = _stage_overlapping_resources(
            resolved_resources, output_dir, Path(tmpdir))
        missing = _pack(target, source_dir, output_dir, keep_methods, traced_defs)
        copied = _copy_resolved_resources(prepared_resources, output_dir)

    manifest_path = output_dir / 'PACKING_MIN.txt'
    with manifest_path.open('a', encoding='utf-8') as f:
        f.write('\nruntime resources:\n')
        if copied:
            for src, dst_rel in copied:
                f.write(f'  {src} -> {dst_rel}\n')
        else:
            f.write('  none\n')

    if copied:
        print('runtime resources:')
        for src, dst_rel in copied:
            print(f'  {src} -> {dst_rel}')

    return missing



def trace_usage(target, trace_output='PYAINO_TRACE.json', source_dir=None, working_dir=None):
    """Run an inference program and record actually used pyaino symbols.

    ``source_dir=None`` traces the same pyaino package that contains this module.
    Pass ``source_dir`` only when intentionally tracing another pyaino source tree.

    By default, the target is executed with its own directory as the current
    working directory.  ``working_dir`` may be supplied when the program's
    runtime data lives under another execution root.  This is useful for legacy
    programs that open paths such as ``corpus_params/model.pkl`` relative to the
    process working directory.
    """
    target = Path(target).resolve()
    trace_output = Path(trace_output).resolve()
    working_dir = target.parent if working_dir is None else Path(working_dir).resolve()
    pyaino_root = current_pyaino_dir() if source_dir is None else Path(source_dir).resolve()
    if not pyaino_root.exists():
        raise RuntimeError(f'pyaino package not found: {pyaino_root}')
    if not working_dir.exists():
        raise FileNotFoundError(f'working directory not found: {working_dir}')

    used = defaultdict(set)

    # sys.setprofile() is invoked for every Python call in the process.
    # Keep the hot path string-only; filesystem/AST work is cached once per file.
    root_norm = os.path.normcase(os.path.abspath(str(pyaino_root)))
    root_prefix = root_norm + os.sep
    inside_cache = {}
    defs_cache = {}

    def inside_pyaino(filename):
        hit = inside_cache.get(filename)
        if hit is not None:
            return hit
        f = os.path.normcase(os.path.abspath(filename))
        hit = (f == root_norm) or f.startswith(root_prefix)
        inside_cache[filename] = hit
        return hit

    def normalize_module(module_name):
        if module_name == 'pyaino':
            return '__init__'
        if module_name.startswith('pyaino.'):
            return module_name[len('pyaino.'):]
        return None

    def add(module_name, symbol):
        module = normalize_module(module_name)
        if module is None:
            return
        if symbol and symbol not in {'<module>', '<lambda>'}:
            used[module].add(symbol)

    def top_defs(filename):
        """Return (classes, functions) defined at module top level."""
        hit = defs_cache.get(filename)
        if hit is not None:
            return hit
        classes, functions = set(), set()
        try:
            source = Path(filename).read_text(encoding='utf-8-sig')
            tree = ast.parse(source, filename=filename)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes.add(node.name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.add(node.name)
        except Exception:
            pass
        hit = (classes, functions)
        defs_cache[filename] = hit
        return hit

    def profiler(frame, event, arg):
        if event != 'call':
            return
        filename = frame.f_code.co_filename
        if not inside_pyaino(filename):
            return

        module_name = frame.f_globals.get('__name__', '')
        if normalize_module(module_name) is None:
            return

        # Instance methods, including inherited methods: record the concrete
        # runtime class rather than the defining base class.
        self_obj = frame.f_locals.get('self')
        if self_obj is not None:
            cls = self_obj.__class__
            add(getattr(cls, '__module__', ''), getattr(cls, '__name__', ''))
            return

        # Class methods.
        cls_obj = frame.f_locals.get('cls')
        if isinstance(cls_obj, type):
            add(getattr(cls_obj, '__module__', ''), getattr(cls_obj, '__name__', ''))
            return

        classes, functions = top_defs(filename)
        name = frame.f_code.co_name
        qualname = getattr(frame.f_code, 'co_qualname', name)

        # Import-time execution of "class Foo: ..." has co_name/co_qualname
        # equal to Foo.  It is definition work, not runtime use.
        if name in classes and qualname == name:
            return

        # staticmethod: Class.method (there is no self/cls local).
        head = qualname.split('.', 1)[0]
        if head in classes and '.' in qualname:
            add(module_name, head)
            return

        # Nested function called from a top-level function: retaining the outer
        # top-level function is sufficient because the nested def lives inside it.
        if '.<locals>.' in qualname:
            outer = qualname.split('.<locals>.', 1)[0].split('.', 1)[0]
            if outer in functions:
                add(module_name, outer)
            elif outer in classes:
                add(module_name, outer)
            return

        # Ordinary module-level function.
        if name in functions:
            add(module_name, name)

    # Make execution resemble running the target directly:
    #   * relative files resolve from working_dir
    #   * target-local helper modules are importable
    #   * the explicitly selected pyaino tree wins over another installation
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    old_pyaino_modules = {
        name: module for name, module in sys.modules.items()
        if name == 'pyaino' or name.startswith('pyaino.')
    }
    for name in list(old_pyaino_modules):
        del sys.modules[name]
    sys.path[:] = [str(pyaino_root.parent), str(target.parent)] + [
        x for x in old_path
        if x not in {str(pyaino_root.parent), str(target.parent)}
    ]

    exc = None
    try:
        os.chdir(working_dir)
        sys.setprofile(profiler)
        runpy.run_path(str(target), run_name='__main__')
    except BaseException as e:
        exc = e
    finally:
        sys.setprofile(None)
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name == 'pyaino' or name.startswith('pyaino.'):
                del sys.modules[name]
        sys.modules.update(old_pyaino_modules)

        data = {
            'trace_format': 1,
            'completed': exc is None,
            'target': str(target),
            'working_dir': str(working_dir),
            'pyaino_root': str(pyaino_root),
            'modules': {m: sorted(v) for m, v in sorted(used.items())},
        }
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        trace_output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if exc is None:
            print(f'trace written: {trace_output}')
        else:
            print(f'incomplete trace written: {trace_output}')
        for module, symbols in data['modules'].items():
            print(f'  {module}: {", ".join(symbols)}')

    if exc is not None:
        raise exc


def load_trace(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if 'completed' in data and not data['completed']:
        raise RuntimeError(f'incomplete runtime trace: {Path(path).resolve()}')
    return {m: set(v) for m, v in data.get('modules', {}).items()}




__all__ = [
    'trace_usage',
    'build_numpy_min',
    'load_trace',
    'current_pyaino_dir',
    'installed_pyaino_dir',
]


def main(argv=None):
    """Generic command-line interface; application-specific drivers are preferred."""
    parser = argparse.ArgumentParser(
        description='Trace and build a runtime-assisted minimum pyaino inference package.')
    sub = parser.add_subparsers(dest='command', required=True)

    p_trace = sub.add_parser('trace', help='run a target and record used pyaino symbols')
    p_trace.add_argument('target', type=Path, help='Python inference program using pyaino')
    p_trace.add_argument('--source', type=Path, default=None,
                         help='pyaino package directory (default: the package containing pyaino.numpy_min)')
    p_trace.add_argument('--trace-output', type=Path, default=Path('PYAINO_TRACE.json'))
    p_trace.add_argument('--working-dir', type=Path, default=None,
                         help='current working directory used while running the target (default: target directory)')

    p_build = sub.add_parser('build', help='build the minimized NumPy-oriented package')
    p_build.add_argument('target', type=Path, help='Python inference program using pyaino')
    p_build.add_argument('--source', type=Path, default=None,
                         help='source pyaino package directory (default: the package containing pyaino.numpy_min)')
    p_build.add_argument('--output', type=Path, default=Path('numpy_package_min'))
    p_build.add_argument('--trace', type=Path, default=None,
                         help='PYAINO_TRACE.json created by the trace command')
    p_build.add_argument('--keep-method', action='append', default=[],
                         help='normally stripped method that this target still needs; repeatable')
    p_build.add_argument('--resource', action='append', default=[],
                         help='runtime file/directory relative to the target; repeatable')

    args = parser.parse_args(argv)
    if args.command == 'trace':
        trace_usage(args.target, args.trace_output, args.source, args.working_dir)
        return

    build_numpy_min(
        args.target,
        output_dir=args.output,
        source_dir=args.source,
        trace=args.trace,
        keep_methods=args.keep_method,
        resources=args.resource,
    )


if __name__ == '__main__':
    main()
