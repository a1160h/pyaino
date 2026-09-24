# test_rope_attentionunit.py
# RoPE integrated AttentionUnit smoke / numerical-gradient test

from pyaino.Config import *

# 数値勾配テストは NumPy で行う方が簡潔かつ安定
#set_np('numpy')
#set_dtype('float32')
#np = Config.np

from pyaino.Neuron import AttentionUnit, RoPE


ATOL_FORWARD = 1e-6
ATOL_GRAD = 3e-3
EPS = 1e-3


def scalar(x):
    """numpy/cupy scalar -> Python float"""
    if hasattr(x, 'get'):
        x = x.get()
    return float(x)


def max_abs(x):
    return scalar(np.max(np.abs(x)))


def assert_close(name, a, b, atol):
    err = max_abs(a - b)
    print(f'{name:28s} max error = {err:.8e}')
    if err > atol:
        raise AssertionError(f'{name}: error {err} > {atol}')


def make_data(B=1, T=3, C=4):
    np.random.seed(123)
    q = np.random.randn(B, T, C).astype(Config.dtype)
    k = np.random.randn(B, T, C).astype(Config.dtype)
    v = np.random.randn(B, T, C).astype(Config.dtype)
    gy = np.random.randn(B, T, C).astype(Config.dtype)
    return q, k, v, gy


def test_rope_flag():
    print('\n--- test_rope_flag ---')
    off = AttentionUnit(head=2, rope=False)
    on  = AttentionUnit(head=2, rope=True)

    assert off.rope is None
    assert isinstance(on.rope, RoPE)
    print('rope=False -> None : OK')
    print('rope=True  -> RoPE : OK')


def test_position0_identity():
    """
    T=1 なら position=0 しかないので RoPE は恒等変換。
    AttentionUnit の出力は rope=False/True で一致するはず。
    """
    print('\n--- test_position0_identity ---')
    q, k, v, _ = make_data(B=1, T=1, C=4)

    off = AttentionUnit(head=2, rope=False, causality=False, scale=True)
    on  = AttentionUnit(head=2, rope=True,  causality=False, scale=True)

    y0 = off.forward(q.copy(), k.copy(), v.copy(), dropout=0.0)
    y1 = on.forward(q.copy(),  k.copy(), v.copy(), dropout=0.0)

    assert_close('T=1 rope off/on', y0, y1, ATOL_FORWARD)


def test_rope_changes_attention():
    """
    T>1 では相対位置回転が入るので、通常は出力が変化する。
    「RoPE が実際に AttentionUnit 内を通っている」ことの簡易確認。
    """
    print('\n--- test_rope_changes_attention ---')
    q, k, v, _ = make_data(B=1, T=4, C=4)

    off = AttentionUnit(head=2, rope=False, causality=False, scale=True)
    on  = AttentionUnit(head=2, rope=True,  causality=False, scale=True)

    y0 = off.forward(q.copy(), k.copy(), v.copy(), dropout=0.0)
    y1 = on.forward(q.copy(),  k.copy(), v.copy(), dropout=0.0)

    diff = max_abs(y1 - y0)
    print(f'rope off/on difference       = {diff:.8e}')
    if diff <= ATOL_FORWARD:
        raise AssertionError('RoPE enabled output did not change for T>1')


def numerical_grad(unit, q, k, v, gy, target, eps=EPS):
    """
    L = sum(AttentionUnit(q,k,v) * gy) の中心差分。
    target: 'q', 'k', 'v'
    """
    src = {'q': q, 'k': k, 'v': v}[target]
    g = np.zeros_like(src)

    for idx in np.ndindex(src.shape):
        qp, kp, vp = q.copy(), k.copy(), v.copy()
        qm, km, vm = q.copy(), k.copy(), v.copy()

        plus  = {'q': qp, 'k': kp, 'v': vp}[target]
        minus = {'q': qm, 'k': km, 'v': vm}[target]

        plus[idx] += eps
        minus[idx] -= eps

        yp = unit.forward(qp, kp, vp, dropout=0.0)
        ym = unit.forward(qm, km, vm, dropout=0.0)

        lp = np.sum(yp * gy)
        lm = np.sum(ym * gy)
        g[idx] = (lp - lm) / (2.0 * eps)

    return g


def test_backward_numerical_gradient():
    """
    RoPE=True の AttentionUnit 全体について q/k/v の解析勾配を
    中心差分と比較する。

    dropout=0, mask=None, causality=False とし、
    RoPE の forward/backward 差し込みそのものを検証する。
    """
    print('\n--- test_backward_numerical_gradient ---')
    q, k, v, gy = make_data(B=1, T=3, C=4)

    unit = AttentionUnit(
        head=2,
        rope=True,
        causality=False,
        scale=True,
    )

    # 解析勾配
    unit.forward(q.copy(), k.copy(), v.copy(), dropout=0.0)
    gq, gk, gv = unit.backward(gy.copy())

    # 数値勾配用。forward 状態は毎回上書きされるので同一 unit でよい。
    num = AttentionUnit(
        head=2,
        rope=True,
        causality=False,
        scale=True,
    )

    ngq = numerical_grad(num, q, k, v, gy, 'q')
    ngk = numerical_grad(num, q, k, v, gy, 'k')
    ngv = numerical_grad(num, q, k, v, gy, 'v')

    assert_close('grad q', gq, ngq, ATOL_GRAD)
    assert_close('grad k', gk, ngk, ATOL_GRAD)
    assert_close('grad v', gv, ngv, ATOL_GRAD)


if __name__ == '__main__':
    test_rope_flag()
    test_position0_identity()
    test_rope_changes_attention()
    test_backward_numerical_gradient()

    print('\nAll AttentionUnit + RoPE tests passed.')
