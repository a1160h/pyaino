# test_ffn_swiglu.py

from pyaino.Config import *
from pyaino.stems_blocks_heads import FeedForward, SwiGLU


# ------------------------------------------------------------
# 基本設定
# ------------------------------------------------------------
set_seed(200)
np = Config.np


def as_float(x):
    """numpy / cupy 共通で scalar -> Python float"""
    return float(x.item())


# ------------------------------------------------------------
# 1. 遅延初期化と shape
# ------------------------------------------------------------
def test_lazy_configuration():

    print('\n--- test_lazy_configuration ---')

    B, T, D = 2, 3, 8
    x = np.random.randn(B, T, D).astype(Config.dtype)

    # FeedForward
    ffn = FeedForward(
        emb_dim=None,
        intermediate=None,
        expansion=2,
    )

    y = ffn.forward(x, dropout=0.0)

    print('FeedForward')
    print('  x.shape =', x.shape)
    print('  y.shape =', y.shape)
    print('  config  =', ffn.config)

    assert y.shape == x.shape
    assert ffn.config == (D, D * 2, 2)

    # SwiGLU
    swiglu = SwiGLU(
        emb_dim=None,
        intermediate=None,
        expansion=2,
    )

    y = swiglu.forward(x, dropout=0.0)

    print('SwiGLU')
    print('  x.shape =', x.shape)
    print('  y.shape =', y.shape)
    print('  config  =', swiglu.config)

    assert y.shape == x.shape
    assert swiglu.config == (D, D * 2, 2)

    print('test_lazy_configuration : OK')


# ------------------------------------------------------------
# 2. 明示 intermediate
# ------------------------------------------------------------
def test_explicit_intermediate():

    print('\n--- test_explicit_intermediate ---')

    B, T, D = 2, 3, 8
    M = 11

    x = np.random.randn(B, T, D).astype(Config.dtype)

    swiglu = SwiGLU(
        emb_dim=None,
        intermediate=M,
    )

    y = swiglu.forward(x, dropout=0.0)

    print('config =', swiglu.config)

    assert y.shape == x.shape
    assert swiglu.config[0] == D
    assert swiglu.config[1] == M

    print('test_explicit_intermediate : OK')


# ------------------------------------------------------------
# 3. backward shape
# ------------------------------------------------------------
def test_backward_shape():

    print('\n--- test_backward_shape ---')

    B, T, D = 2, 3, 8
    M = 10

    x = np.random.randn(B, T, D).astype(Config.dtype)

    for cls in (FeedForward, SwiGLU):

        np.random.seed(200)

        model = cls(
            emb_dim=D,
            intermediate=M,
        )

        y = model.forward(x, dropout=0.0)

        gy = np.random.randn(*y.shape).astype(Config.dtype)
        gx = model.backward(gy)

        print(cls.__name__)
        print('  x ', x.shape)
        print('  y ', y.shape)
        print('  gy', gy.shape)
        print('  gx', gx.shape)

        assert gx.shape == x.shape

    print('test_backward_shape : OK')


# ------------------------------------------------------------
# 4. 数値微分
#
# L = sum(y * gy)
#
# dL/dx を backward と finite difference で比較する。
# SwiGLU では特に
#
#       gate ----+
#                *
#       up ------+
#
# の分岐と合流が正しいことを確認できる。
# ------------------------------------------------------------
def numerical_gradient_x(model, x, gy, h=1e-3):

    gx_num = np.zeros_like(x)

    x_flat = x.reshape(-1)
    gx_flat = gx_num.reshape(-1)

    for i in range(x_flat.size):

        xp = x.copy()
        xm = x.copy()

        xp.reshape(-1)[i] += h
        xm.reshape(-1)[i] -= h

        yp = model.forward(xp, dropout=0.0)
        ym = model.forward(xm, dropout=0.0)

        lp = np.sum(yp * gy)
        lm = np.sum(ym * gy)

        gx_flat[i] = (lp - lm) / (2 * h)

    return gx_num


def check_gradient(cls, D=4, M=6):

    # 小さくして数値微分を軽くする
    B, T = 1, 2

    np.random.seed(200)

    model = cls(
        emb_dim=D,
        intermediate=M,
    )

    x = np.random.randn(B, T, D).astype(Config.dtype)

    # analytical gradient
    y = model.forward(x, dropout=0.0)
    gy = np.random.randn(*y.shape).astype(Config.dtype)

    gx = model.backward(gy)

    # analytical backward が終わってから数値微分する。
    # numerical forward により model 内部状態が更新されても
    # gx はすでに確定している。
    gx_num = numerical_gradient_x(model, x, gy)

    diff = np.abs(gx - gx_num)

    max_abs = as_float(np.max(diff))

    scale = np.maximum(
        np.maximum(np.abs(gx), np.abs(gx_num)),
        1e-5
    )

    max_rel = as_float(np.max(diff / scale))

    print('\n', cls.__name__)
    print('gx =')
    print(gx)
    print('gx numerical =')
    print(gx_num)
    print('max abs error =', max_abs)
    print('max rel error =', max_rel)

    # float32 の central difference なので少し余裕を持たせる
    assert max_abs < 5e-3

    return max_abs, max_rel


def test_gradient():

    print('\n--- test_gradient ---')

    check_gradient(FeedForward)
    check_gradient(SwiGLU)

    print('test_gradient : OK')


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
if __name__ == '__main__':

    test_lazy_configuration()
    test_explicit_intermediate()
    test_backward_shape()
    test_gradient()

    print('\n==============================')
    print('ALL TESTS PASSED')
    print('==============================')
