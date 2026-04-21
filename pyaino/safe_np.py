# safe_np.py
# 20260421 A.Inoue

# safe_np 使用ガイドライン
#
# 本モジュールは、NumPy / CuPy 間の差異や版数依存、未実装関数などを吸収し、
# 実装の安定性を高めるための「互換レイヤ」である。
# 自動微分のための必須要素ではない点に注意すること。
#
# 【基本方針】
# - safe_np は「必要な箇所に限定して使用する」
# - すべての np 呼び出しを置き換えることは目的としない
#
# 【使用すべき箇所（推奨・必須）】
# - NumPy / CuPy 間で挙動差・未実装・版数差がある関数
#   例：add.at, 特殊関数（erf 等）, cupyx 依存機能 など
# - 将来的に互換性問題が再発する可能性が高い箇所
#
# 【使用が望ましい箇所（任意）】
# - reshape, transpose, broadcast_to, expand_dims などの基本操作
# - concatenate / split / where / reduction 系など頻出処理
# → 統一的に扱いたい場合のみ使用
#
# 【使用しなくてよい箇所】
# - 明確に NumPy / CuPy 固有処理を行う場合
# - 外部ライブラリ制約により直接呼び出しが必要な場合
# - 挙動が完全に安定しており、ラップの必要がない場合
#
# 【設計原則】
# - safe_np は「すべてを包む層」ではなく、「差異を吸収する層」とする
# - 問題が起きた際に、原因（backend / safe_np / 上位層）を切り分けやすく保つ
#
# 必要最小限にとどめることで、保守性と可読性のバランスを維持すること。
#
# transpose の扱い
#
# - x.T はそのまま使用してよい（NumPy / CuPy 間で安定）
# - np.transpose(x) は x.T に置き換える（axes指定なしの場合）
# - 軸指定あり transpose は safe_np 経由を推奨
#     snp.transpose(x, axes=...)
#
# - 可読性・保守性の観点から、
#   可能であれば「軸の意味」に基づくラッパ関数の導入も検討する
#
# reduction 系（sum, mean など）は CuPy / NumPy 間で API 差異があり、
# backward 内での安定動作のため safe_np で統一する

from pyaino import Config
np = Config.np

if np.__name__ == "cupy":
    import cupy as _cupy
else:
    _cupy = None

try:
    import cupyx as _cupyx
except ImportError:
    _cupyx = None

import numpy as _numpy
import math as _math
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
        mod = _resolve_attr(_cupy, path)
        if mod is not None:
            modules.append(mod)
    return modules


def _get_cupy13_modules():
    modules = []
    core = getattr(_cupy, "_core", None)
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
    if _cupy is None:
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

                #print(f'###_load_cupy_func {func}')
                
                return func

    # フォールバック
    func = getattr(_cupy, name, None)
    if callable(func):
        return func

    raise AttributeError(f"{name} not found in CuPy internals or public API.")


# ------------------------------------------------------------
# CuPy の参照を import 時に固定
# ------------------------------------------------------------
if _cupy is not None:
    _cupy_ndarray = _cupy.ndarray

    _cupy_expand_dims = _cupy.expand_dims
    _cupy_asarray = _cupy.asarray
    _cupy_asnumpy = _cupy.asnumpy

    _cupy_sum = _cupy_ndarray.sum
    _cupy_mean = _cupy_ndarray.mean
    _cupy_std = _cupy_ndarray.std
    _cupy_var = _cupy_ndarray.var
    _cupy_prod = _cupy_ndarray.prod
    _cupy_max = _cupy_ndarray.max
    _cupy_min = _cupy_ndarray.min
    _cupy_argmax = _cupy_ndarray.argmax
    _cupy_argmin = _cupy_ndarray.argmin
    _cupy_cumsum = _cupy_ndarray.cumsum
    _cupy_cumprod = _cupy_ndarray.cumprod

    _cupy_reduction_methods = {
        "sum": _cupy_sum,
        "mean": _cupy_mean,
        "std": _cupy_std,
        "var": _cupy_var,
        "prod": _cupy_prod,
        "max": _cupy_max,
        "min": _cupy_min,
        "argmax": _cupy_argmax,
        "argmin": _cupy_argmin,
        "cumsum": _cupy_cumsum,
        "cumprod": _cupy_cumprod,
    }
else:
    _cupy_ndarray = None
    _cupy_expand_dims = None
    _cupy_asarray = None
    _cupy_asnumpy = None
    _cupy_reduction_methods = {}


def _cupy_reduction(a, method_name, axis, dtype, out, keepdims):
    """
    CuPy の reduction を ndarray メソッド経由で安全に実行する。

    safe_np では reduction 系の呼び出し形をある程度そろえるため、
    一部メソッドでも共通引数（axis, dtype, out, keepdims）を受ける。
    ただし CuPy ndarray メソッドで未対応の機能は以下のように扱う。

    - out はサポートしない（指定時は例外）
    - keepdims は必要に応じて safe_np 側で再現する
    - dtype を受けないメソッドでは無視または非使用とする

    """
    method = _cupy_reduction_methods.get(method_name, None)
    if method is None:
        raise AttributeError(f"CuPy ndarray method '{method_name}' is not available")

    # CuPy ndarray メソッドは out を受け取れない
    if out is not None:
        raise NotImplementedError(f"CuPy ndarray.{method_name} does not support 'out'")

    #print(f'###_cupy_reduction {method} {type(a)}')

    # ---- CuPy ndarray メソッド流儀で呼ぶ ----
    if method_name in ("sum", "mean", "std", "var", "prod"):
        # 元の実装と同様、位置引数で呼ぶ
        res = method(a, axis, dtype)

    elif method_name in ("max", "min"):
        # CuPy の max/min は dtype を受けない
        res = method(a, axis)

    elif method_name in ("argmax", "argmin"):
        # dtype も out も keepdims も元メソッドでは未対応
        res = method(a, axis)

    elif method_name in ("cumsum", "cumprod"):
        # axis, dtype は環境差が出やすいので分岐
        if axis is None and dtype is None:
            res = method(a)
        elif axis is None:
            res = method(a, None, dtype)
        elif dtype is None:
            res = method(a, axis)
        else:
            res = method(a, axis, dtype)

    else:
        raise ValueError(f"Unsupported reduction method: {method_name}")

    # --- keepdims の再現 ---
    if keepdims:
        if axis is None:
            # NumPy と同じ：全軸縮約 → (1,1,1,...)
            return res.reshape((1,) * a.ndim)
        else:
            # axis が単数か複数かを正規化
            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(ax if ax >= 0 else ax + a.ndim for ax in axes)
            for ax in sorted(axes):
                res = _cupy_expand_dims(res, ax)

    return res


# ------------------------------------------------------------
# fallbackの用意
# ------------------------------------------------------------
def _fallback_add_at(x, indices, values):
    for idx, val in zip(indices, values):
        x[idx] += val
    return None  # x の破壊的更新


def _numpy_erf(x):
    """
    NumPy 配列/スカラーに対して math.erf をベクトル化して適用する。
    NumPy トップレベルに erf が無い環境でも動作するようにする。
    """
    vec_erf = _numpy.vectorize(_math.erf, otypes=[float])
    return vec_erf(x)


def _fallback_erf(x):
    """
    NumPy/CuPy 共通 fallback。
    CuPy 配列なら一度 CPU に戻してから math.erf ベースで処理し、
    最後に元の backend に戻す。
    """
    if _cupy is not None and isinstance(x, _cupy.ndarray):
        return _cupy_asarray(_numpy_erf(_cupy_asnumpy(x)))
    else:
        return _numpy_erf(x)



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
    concatenate  = _cupy.concatenate
    stack        = _cupy.stack
    hstack       = _cupy.hstack
    vstack       = _cupy.vstack
    dstack       = _cupy.dstack
    split        = _cupy.split
    tile         = _cupy.tile
    repeat       = _cupy.repeat
    take         = _cupy.take
    put          = _cupy.put

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

    einsum       = _cupy.einsum

    sort         = _cupy.sort
    argsort      = _cupy.argsort

    where        = _cupy.where
    clip         = _cupy.clip
    expand_dims  = _cupy_expand_dims
    squeeze      = _cupy.squeeze

    triu_indices = _cupy.triu_indices
    tril_indices = _cupy.tril_indices

    # --------------------------------------------------------
    # add.at のフォールバック処理
    # --------------------------------------------------------
    if _cupyx is not None and hasattr(_cupyx, "scatter_add"):
        add_at = _cupyx.scatter_add
    elif _cupy is not None and hasattr(_cupy, "add") and hasattr(_cupy.add, "at"):
        add_at = _cupy.add.at
    else:
        add_at = _fallback_add_at

    if _cupyx is not None and hasattr(_cupyx, "scipy") and hasattr(_cupyx.scipy, "special"):
        erf = _cupyx.scipy.special.erf
    elif _cupy is not None and hasattr(_cupy, "erf"):
        # 将来版 or 一部環境対策
        erf = _cupy.erf
    else:
        erf = _fallback_erf

else:
    # NumPy の場合は全部そのまま
    reshape      = _numpy.reshape
    broadcast_to = _numpy.broadcast_to
    moveaxis     = _numpy.moveaxis
    ascontiguousarray = _numpy.ascontiguousarray
    transpose    = _numpy.transpose

    concatenate  = _numpy.concatenate
    stack        = _numpy.stack
    hstack       = _numpy.hstack
    vstack       = _numpy.vstack
    dstack       = _numpy.dstack
    split        = _numpy.split
    tile         = _numpy.tile
    repeat       = _numpy.repeat
    take         = _numpy.take
    put          = _numpy.put

    sum          = _numpy.sum
    mean         = _numpy.mean
    std          = _numpy.std
    var          = _numpy.var
    prod         = _numpy.prod
    max          = _numpy.max
    min          = _numpy.min
    argmax       = _numpy.argmax
    argmin       = _numpy.argmin
    cumsum       = _numpy.cumsum
    cumprod      = _numpy.cumprod
    einsum       = _numpy.einsum

    sort         = _numpy.sort
    argsort      = _numpy.argsort

    where        = _numpy.where
    clip         = _numpy.clip
    expand_dims  = _numpy.expand_dims
    squeeze      = _numpy.squeeze

    triu_indices = _numpy.triu_indices
    tril_indices = _numpy.tril_indices

    add_at       = _numpy.add.at

    if hasattr(_numpy, "erf"):
        erf = _numpy.erf
    else:
        erf = _numpy_erf

