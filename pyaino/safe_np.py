# safe_np.py
# 20260418 A.Inoue

from pyaino.Config import *

"""
CuPy の高レベル API（cp.reshape, cp.transposeなど）は内部で余計な引数を付けるので使いたくない
ndarray メソッド（gy.reshape(), gy.transpose()）は Pyaino の自動微分に捕まるので使えない
NumPy の API も使わない
だから CuPy の低レベル実装（core.core）を使う
NumPy のときは普通のもの(np.reshape, np.transposeなど)を使う

"""
# NumPy / CuPy のどちらかで動作する安全ラッパー

# safe_np.py

import numpy as np

if np.__name__ == 'cupy':
    # CuPy の低レベル API を取得（12系/13系 両対応）
    try:
        core = np.core.core
    except Exception:
        core = np._core.core

    # CuPy の低レベル関数（Pyaino に捕まらず、余計な引数も付かない）
    reshape      = core.reshape
    transpose    = core.transpose
    broadcast_to = core.broadcast_to
    broadcast_arrays  = core.broadcast_arrays
    moveaxis     = core.moveaxis
    sum          = core.sum
    mean         = core.mean
    std          = core.std
    var          = core.var
    max          = core.max
    min          = core.min
    prod         = core.prod

    # 追加分
    expand_dims  = core.expand_dims
    squeeze      = core.squeeze
    clip         = core.clip
    where        = core.where
    cumsum       = core.cumsum
    cumprod      = core.cumprod
    split        = core.split
    concatenate  = core.concatenate 
    stack        = core.stack
    hstack       = core.hstack
    vstack       = core.vstack
    dstack       = core.dstack

    tile         = core.tile
    repeat       = core.repeat
    take         = core.take
    put          = core.put
    argmax       = core.argmax
    argmin       = core.argmin
    sort         = core.sort
    argsort      = core.argsort
    ascontiguousarray = core.ascontiguousarray
    triu_indices = core.triu_indices
    tril_indices = core.tril_indices

else:
    # NumPy の通常 API（こちらは安全）
    reshape      = np.reshape
    transpose    = np.transpose
    broadcast_to = np.broadcast_to
    broadcast_arrays  = np.broadcast_arrays
    moveaxis     = np.moveaxis
    sum          = np.sum
    mean         = np.mean
    std          = np.std
    var          = np.var
    max          = np.max
    min          = np.min
    prod         = np.prod

    # 追加分
    expand_dims  = np.expand_dims
    squeeze      = np.squeeze
    clip         = np.clip
    where        = np.where
    cumsum       = np.cumsum
    cumprod      = np.cumprod
    split        = np.split
    concatenate  = np.concatenate

    stack        = np.stack
    hstack       = np.hstack
    vstack       = np.vstack
    dstack       = np.dstack

    tile         = np.tile
    repeat       = np.repeat
    take         = np.take
    put          = np.put
    argmax       = np.argmax
    argmin       = np.argmin
    sort         = np.sort
    argsort      = np.argsort
    ascontiguousarray = np.ascontiguousarray
    triu_indices = np.triu_indices
    tril_indices = np.tril_indices

# 環境に応じた関数の選択
try:
    add_at = np.add.at
except:
    try:
        add_at = np.scatter_add
    except:
        try:
            add_at = np._cupyx.scatter_add
        except:
            def add_at(x, y, z): # xのyの位置にzを加算する
                for idx, val in zip(y, z):
                    x[idx] += val

    
