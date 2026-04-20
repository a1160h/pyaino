# safe_np.py
# 20260419 A.Inoue

from pyaino.Config import *
if np.__name__ == "cupy":
    import cupy
else:
    cupy = None

try:
    import cupyx
except ImportError:
    cupyx = None
    
import numpy
from functools import lru_cache


"""
CuPy の高レベル API（cp.reshape, cp.transposeなど）は
内部で余計な引数を付けるので使いたくない
ndarray メソッド（gy.reshape(), gy.transpose()）は
pyaino の自動微分に捕まるので使えない
NumPy の API も使わない
だから CuPy の低レベル実装（core.core）を使う
NumPy のときは高レベル API (普通のもの np.reshape, np.transposeなど)
をそのまま使うが、逆にpyainoの自動微分を避ける（安全策）ため、ここで定義
（本来pyainoの自動微分は変数がHDArrayかどうかを見ているのでこれは安全策）

"""

def _resolve_attr(obj, path):
    """
    'a.b.c' のような属性パスを辿って取得する。
    見つからなければ None を返す。
    """
    for attr in path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj

def _get_cupy12_modules():
    modules = []
    for path in ("_core.core", "core.core"):
        mod = _resolve_attr(cupy, path)
        if mod is not None:
            modules.append(mod)
    return modules

def _get_cupy13_modules():
    modules = []
    core = getattr(cupy, "_core", None)
    if core is not None:
        for attr in dir(core):
            if attr.startswith("_routines_"):
                mod = getattr(core, attr, None)
                if mod is not None:
                    modules.append(mod)
    return modules


@lru_cache(None)
def _load_cupy_func(name):
    """
    CuPy の内部関数を探索して返す（キャッシュ付き）。
    見つからなければ cupy.<name> を返す。

    - "reshape" → "reshape", "_reshape" の両方を探索
    - CuPy 12 / 13+ に対応
    """
    if cupy is None:
        raise RuntimeError(
            f"_load_cupy_func('{name}') called but CuPy is not available")

    # 候補名
    candidate_names = (name, f"_{name}")

    # 探索対象モジュール
    modules = []
    modules.extend(_get_cupy12_modules())
    modules.extend(_get_cupy13_modules())

    # 探索
    for mod in modules:
        for cand in candidate_names:
            func = getattr(mod, cand, None)
            if callable(func):
                return func
        
    # フォールバック
    func = getattr(cupy, name, None)
    if callable(func):
        return func

    raise AttributeError(f"{name} not found in CuPy internals or public API.")

def _cupy_reduction_bkup(a, method_name, axis, dtype, out, keepdims):
    """
    cupyの関数をndarrayメソッドに落として実行
    不足するkeepdimsは中で付加、outはサポートせず例外にする

    """
    method = getattr(a, method_name, None)
    if method is None:
        raise AttributeError(f"CuPy ndarray has no method '{method_name}'")

    # CuPy ndarray メソッドは out を受け取れない
    if out is not None:
        raise NotImplementedError(f"CuPy ndarray.{method_name} does not support 'out'")

    # CuPy ndarray メソッドは (axis, dtype) のみ
    res = method(axis, dtype)

    # keepdims を自前で再現
    if keepdims and axis is not None:
        if not isinstance(axis, tuple):
            axes = (axis,)
        else:
            axes = axis
        for ax in sorted(axes):
            res = cupy.expand_dims(res, ax)

    return res

def _cupy_reduction(a, method_name, axis, dtype, out, keepdims):
    """
    cupyの関数をndarrayメソッドに落として実行
    不足するkeepdimsは中で付加、outはサポートせず例外にする

    """
    method = getattr(a, method_name, None)
    if method is None:
        raise AttributeError(f"CuPy ndarray has no method '{method_name}'")

    # CuPy ndarray メソッドは out を受け取れない
    if out is not None:
        raise NotImplementedError(f"CuPy ndarray.{method_name} does not support 'out'")

    # CuPy ndarray メソッドは (axis, dtype) のみ
    res = method(axis, dtype)

    # --- keepdims の再現 ---
    if keepdims:
        if axis is None:
            # NumPy と同じ：全軸縮約 → (1,1,1,...)
            return res.reshape((1,) * a.ndim)
        else:
            # axis が単数か複数かを正規化
            axes = axis if isinstance(axis, tuple) else (axis,)
            for ax in sorted(axes):
                res = cupy.expand_dims(res, ax)

    return res



# ------------------------------------------------------------
# fallbackの用意
# ------------------------------------------------------------
def _fallback_add_at(x, indices, values):
    for idx, val in zip(indices, values):
        x[idx] += val
    return None # x の破壊的更新   

def _fallback_erf(x):
    # numpyベース fallback（cupy配列なら一度CPUに戻る）
    if cupy is not None and isinstance(x, cupy.ndarray):
        return cupy.asarray(np.erf(cupy.asnumpy(x)))
    else:
        return np.erf(x)

# ------------------------------------------------------------
# NumPy / CuPy の切り替え
# ------------------------------------------------------------
if np.__name__ == "cupy":
    reshape      = _load_cupy_func("reshape")
    broadcast_to = _load_cupy_func("broadcast_to")
    moveaxis     = _load_cupy_func("moveaxis")
    ascontiguousarray = _load_cupy_func("ascontiguousarray")
    transpose    = _load_cupy_func("transpose")

    # ここからは高レベル API を使う
    concatenate  = cupy.concatenate
    stack        = cupy.stack
    hstack       = cupy.hstack
    vstack       = cupy.vstack
    dstack       = cupy.dstack
    split        = cupy.split
    tile         = cupy.tile
    repeat       = cupy.repeat
    take         = cupy.take
    put          = cupy.put

    def sum(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "sum", axis, dtype, out, keepdims)

    def mean(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "mean", axis, dtype, out, keepdims)

    def std(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "std", axis, dtype, out, keepdims)

    def var(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "var", axis, dtype, out, keepdims)

    def prod(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "prod", axis, dtype, out, keepdims)

    def max(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "max", axis, dtype, out, keepdims)

    def min(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "min", axis, dtype, out, keepdims)

    def argmax(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "argmax", axis, dtype, out, keepdims)

    def argmin(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "argmin", axis, dtype, out, keepdims)

    def cumsum(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "cumsum", axis, dtype, out, keepdims)

    def cumprod(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "cumprod", axis, dtype, out, keepdims)
  
    #mean         = cupy.mean
    #std          = cupy.std
    #var          = cupy.var
    #prod         = cupy.prod
    #max          = cupy.max
    #min          = cupy.min
    #argmax       = cupy.argmax
    #argmin       = cupy.argmin
    #cumsum       = cupy.cumsum
    #cumprod      = cupy.cumprod
    einsum       = cupy.einsum
    
    sort         = cupy.sort
    argsort      = cupy.argsort

    where        = cupy.where
    clip         = cupy.clip
    expand_dims  = cupy.expand_dims
    squeeze      = cupy.squeeze

    triu_indices = cupy.triu_indices
    tril_indices = cupy.tril_indices

    # --------------------------------------------------------
    # add.at のフォールバック処理
    # --------------------------------------------------------
    """
    try:
        add_at = cupy.add.at
    except AttributeError:
        try:
            add_at = cupy.scatter_add
        except AttributeError:
            try:
                import cupyx
                add_at = cupyx.scatter_add
            except AttributeError:
                add_at = _fall_back_add_at
    """

    if cupyx is not None and hasattr(cupyx, "scatter_add"):
        add_at = cupyx.scatter_add
    elif cupy is not None and hasattr(cupy, "add") and hasattr(cupy.add, "at"):
        add_at = cupy.add.at
    else:
        add_at = _fallback_add_at                


    if cupyx is not None and hasattr(cupyx, "scipy") and hasattr(cupyx.scipy, "special"):
        erf = cupyx.scipy.special.erf
    elif cupy is not None and hasattr(cupy, "erf"):
        # 将来版 or 一部環境対策
        erf = cupy.erf
    else:
        erf = _fallback_erf


else:
    # NumPy の場合は全部そのまま
    reshape      = numpy.reshape
    broadcast_to = numpy.broadcast_to
    moveaxis     = numpy.moveaxis
    ascontiguousarray = numpy.ascontiguousarray
    transpose    = numpy.transpose
    
    concatenate  = numpy.concatenate
    stack        = numpy.stack
    hstack       = numpy.hstack
    vstack       = numpy.vstack
    dstack       = numpy.dstack
    split        = numpy.split
    tile         = numpy.tile
    repeat       = numpy.repeat
    take         = numpy.take
    put          = numpy.put

    sum          = numpy.sum
    mean         = numpy.mean
    std          = numpy.std
    var          = numpy.var
    prod         = numpy.prod
    max          = numpy.max
    min          = numpy.min
    argmax       = numpy.argmax
    argmin       = numpy.argmin
    cumsum       = numpy.cumsum
    cumprod      = numpy.cumprod
    einsum       = numpy.einsum

    sort         = numpy.sort
    argsort      = numpy.argsort

    where        = numpy.where
    clip         = numpy.clip
    expand_dims  = numpy.expand_dims
    squeeze      = numpy.squeeze
    
    triu_indices = numpy.triu_indices
    tril_indices = numpy.tril_indices

    add_at = numpy.add.at
    erf    = numpy.erf




