# -*- coding: utf-8 -*-
"""pyaino.LinearLayerCrossEntropy の KD 経路テスト（整理後の新版）。

Neuron.py の統合版を直接 import する。Neuron.py は書き換えない。
旧版 (_bkup) や単独 KD クラスは使わない。

    python test_LinearLayerCrossEntropy_KD_再検証.py

数学的な基準：
    L = tau**2 * mean( KL(q_tau || softmax(z / tau)) )
    dL/dz = tau/N * (softmax(z / tau) - q_tau)
q は「教師の温度 tau での確率」であり、教師 logits ではない。
"""
from pyaino.Config import *
from pyaino.Neuron import LinearLayerCrossEntropy

DTYPE = Config.dtype
RTOL = 6.e-5
ATOL = 5.e-5
passed = 0


def same(actual, expected, label, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=label)


def same_id(actual, expected, label):
    np.testing.assert_array_equal(actual, expected, err_msg=label)


def softmax(z):
    u = z - z.max(axis=-1, keepdims=True)
    e = np.exp(u)
    return e / e.sum(axis=-1, keepdims=True)


def kd_reference(x, w, b, q, tau):
    """全語彙一括計算による独立した NumPy/CuPy 基準値。scale=False。"""
    z = x @ w
    if b is not None:
        z = z + b
    u = z / tau
    u = u - u.max(axis=-1, keepdims=True)
    logp = u - np.log(np.exp(u).sum(axis=-1, keepdims=True))
    p = np.exp(logp)
    logq = np.log(np.maximum(q, np.asarray(1.e-30, dtype=DTYPE)))
    loss = np.mean(np.sum(q * (logq - logp), axis=-1)) * tau**2
    N = x.size // x.shape[-1]
    gz = (p - q) * (tau / N)
    gx = gz @ w.T
    gw = x.reshape(-1, x.shape[-1]).T @ gz.reshape(-1, w.shape[-1])
    gb = gz.reshape(-1, w.shape[-1]).sum(axis=0) if b is not None else None
    return z.argmax(axis=-1), loss, gx, gw, gb


def make_layer(d, v, tile, bias=True, scale=False, w=None, b=None, gamma=None):
    layer = LinearLayerCrossEntropy(d, v, matmul=True, bias=bias,
                                   scale=scale, tile_size=tile, optimize='SGD')
    p = layer.parameters
    _ = p()                    # パラメータを初期化
    if w is not None:
        p.w[...] = w
    if bias and b is not None:
        p.b[...] = b
    if scale and gamma is not None:
        p.gamma[...] = gamma
    return layer


def parameters(d, v, bias=True):
    w = np.random.normal(0.0, 0.28, size=(d, v)).astype(DTYPE)
    b = (np.random.normal(0.0, 0.2, size=(v,)).astype(DTYPE)
         if bias else None)
    return w, b


def passed_case(label):
    global passed
    passed += 1
    print(' PASS', passed, '-', label)


print('Backend:', np.__name__)
print('Layer:', LinearLayerCrossEntropy.__module__ + '.' +
      LinearLayerCrossEntropy.__name__)

# 1: one-hot KD(tau=1) は旧形式 CE と全く同じ目的関数・勾配。
B, S, D, V = 2, 3, 4, 11
x = np.random.normal(0.0, 1.0, (B, S, D)).astype(DTYPE)
t = np.array([[0, 3, V-1], [3, 0, V-1]], dtype=np.int32)
q = np.eye(V, dtype=DTYPE)[t]
w, b = parameters(D, V)
a = make_layer(D, V, 3, w=w, b=b)
c = make_layer(D, V, 3, w=w, b=b)
y_ce, l_ce = a.forward(x, t)
y_kd, l_kd = c.forward(x, q=q, tau=1.0)
same_id(y_kd, y_ce, 'one-hot prediction')
same(l_kd, l_ce, 'one-hot CE=KL loss')
gx_ce, gx_kd = a.backward(), c.backward()
same(gx_kd, gx_ce, 'one-hot gx')
same(c.parameters.grad_w, a.parameters.grad_w, 'one-hot gw')
same(c.parameters.grad_b, a.parameters.grad_b, 'one-hot gb')
a.update(eta=0.001)
c.update(eta=0.001)
same(c.parameters.w, a.parameters.w, 'one-hot updated w')
same(c.parameters.b, a.parameters.b, 'one-hot updated b')
passed_case('one-hot KD (tau=1) == CE: loss/grad/update')

# 2: soft targets の KL、解析的勾配。形状・温度・タイル幅・bias有無を網羅。
cases = [
    ('3D V=11 tile=1 T=1',     (2, 3, 4), 11, 1,    1.0, True),
    ('3D V=11 tile=3 T=2',     (2, 3, 4), 11, 3,    2.0, True),
    ('3D V=11 tile=V T=0.7',   (2, 3, 4), 11, 11,   0.7, True),
    ('3D V=11 tile>V T=4',     (2, 3, 4), 11, 15,   4.0, True),
    ('3D V=11 tile=None T=2',  (2, 3, 4), 11, None, 2.0, True),
    ('2D V=17 tile=4 T=2',     (5, 4),    17, 4,    2.0, True),
    ('2D V=17 tile=V T=1',     (5, 4),    17, 17,   1.0, True),
    ('3D bias=False T=3',      (2, 3, 4), 19, 2,    3.0, False),
    ('2D bias=False T=0.5',    (5, 4),    7, 1,     0.5, False),
]
for label, shape, v, tile, tau, bias in cases:
    d = shape[-1]
    x = np.random.normal(0.0, 0.8, size=shape).astype(DTYPE)
    teacher_logits = np.random.normal(0.0, 1.0, size=shape[:-1] + (v,)).astype(DTYPE)
    q = softmax(teacher_logits / tau)
    w, b = parameters(d, v, bias)
    layer = make_layer(d, v, tile, bias=bias, w=w, b=b)
    expected = kd_reference(x, w, b, q, tau)
    y, loss = layer.forward(x, q=q, tau=tau)
    same_id(y, expected[0], label + ' predicted IDs')
    same(loss, expected[1], label + ' KL loss')
    gx = layer.backward()
    same(gx, expected[2], label + ' gx')
    same(layer.parameters.grad_w, expected[3], label + ' gw')
    if bias:
        same(layer.parameters.grad_b, expected[4], label + ' gb')
    passed_case(label + ': dense reference loss/gradient')

# 3: q配列と q(start,end) を同じ教師分布として比較。
# コールバックが順伝播・逆伝播ともに各タイル一回呼ばれることも検査。
shape, v, tile = (2, 3, 4), 13, 4
x = np.random.normal(0.0, 0.8, size=shape).astype(DTYPE)
w, b = parameters(shape[-1], v)
teacher_logits = np.random.normal(0.0, 1.0, size=shape[:-1] + (v,)).astype(DTYPE)
for tau in (0.7, 1.0, 2.0, 4.0):
    q = softmax(teacher_logits / tau)
    dense = make_layer(shape[-1], v, 3, w=w, b=b)
    streamed = make_layer(shape[-1], v, tile, w=w, b=b)
    calls = []

    def teacher_tile(start, end):
        calls.append((start, end))
        return q[..., start:end]

    yd, ld = dense.forward(x, q=q, tau=tau)
    ys, ls = streamed.forward(x, q=teacher_tile, tau=tau)
    same_id(ys, yd, 'streamed predicted IDs')
    same(ls, ld, 'streamed loss tau=' + str(tau))
    gd = dense.backward()
    gs = streamed.backward()
    same(gs, gd, 'streamed gx tau=' + str(tau))
    same(streamed.parameters.grad_w, dense.parameters.grad_w,
         'streamed gw tau=' + str(tau))
    same(streamed.parameters.grad_b, dense.parameters.grad_b,
         'streamed gb tau=' + str(tau))
    expected_calls = [(start, min(start + tile, v)) for start in range(0, v, tile)] * 2
    assert calls == expected_calls, ('teacher_tile call sequence', calls, expected_calls)
    passed_case('q array == streaming q, tau=' + str(tau))

# 4: scale=True のタイル非依存性と gamma 勾配の差分検査。
shape, v, tau = (2, 3, 4), 9, 2.0
x = np.random.normal(0.0, 0.6, size=shape).astype(DTYPE)
w, b = parameters(shape[-1], v)
q = softmax(np.random.normal(0.0, 1.0, size=shape[:-1] + (v,)).astype(DTYPE) / tau)
a = make_layer(shape[-1], v, None, scale=True, w=w, b=b)
c = make_layer(shape[-1], v, 2, scale=True, w=w, b=b,
               gamma=a.parameters.gamma)
ya, la = a.forward(x, q=q, tau=tau)
yc, lc = c.forward(x, q=q, tau=tau)
same_id(yc, ya, 'scale predicted IDs')
same(lc, la, 'scale loss')
ga, gc = a.backward(), c.backward()
same(gc, ga, 'scale gx')
same(c.parameters.grad_w, a.parameters.grad_w, 'scale gw')
same(c.parameters.grad_b, a.parameters.grad_b, 'scale gb')
same(c.parameters.ggamma, a.parameters.ggamma, 'scale ggamma')
ggamma = c.parameters.ggamma.copy()
# gammaだけの有限差分。float32なので差分幅は少し大きめにする。
eps = 1.e-2
gamma0 = c.parameters.gamma.copy()
c.parameters.gamma[...] = gamma0 + eps
_, lp = c.forward(x, q=q, tau=tau)
c.parameters.gamma[...] = gamma0 - eps
_, lm = c.forward(x, q=q, tau=tau)
c.parameters.gamma[...] = gamma0
finite_diff = (lp - lm) / (2.0 * eps)
same(ggamma, finite_diff, 'scale gamma numerical derivative',
     rtol=2.e-2, atol=3.e-3)
passed_case('scale=True: tiled/full gradients and finite-difference gamma')

# 5: SGDの一歩で、固定バッチのKD損失が減少すること。
shape, v, tau = (2, 3, 4), 11, 2.0
x = np.random.normal(0.0, 0.5, size=shape).astype(DTYPE)
q = softmax(np.random.normal(0.0, 1.0, size=shape[:-1] + (v,)).astype(DTYPE) / tau)
layer = make_layer(shape[-1], v, 3)
_, before = layer.forward(x, q=q, tau=tau)
layer.backward()
w_old = layer.parameters.w.copy()
layer.update(eta=0.001)
_, after = layer.forward(x, q=q, tau=tau)
assert bool(after < before), ('KD loss did not decrease after SGD', before, after)
assert bool(np.any(layer.parameters.w != w_old)), 'weight was not updated'
passed_case('SGD one step decreases fixed-batch KD loss')

# 6: 旧CEを優先する仕様。qを参照しないことも確認する。
shape, v = (2, 3, 4), 11
x = np.random.normal(0.0, 0.6, size=shape).astype(DTYPE)
t = np.random.randint(0, v, size=shape[:-1]).astype(np.int32)
w, b = parameters(shape[-1], v)
ref = make_layer(shape[-1], v, 3, w=w, b=b)
new = make_layer(shape[-1], v, 3, w=w, b=b)


def unused_q(start, end):
    raise AssertionError('t優先時にqを参照しています')


y0, l0 = ref.forward(x, t)
y1, l1 = new.forward(x, t, q=unused_q, tau=2.5)
same_id(y1, y0, 't priority predicted IDs')
same(l1, l0, 't priority CE loss')
g0, g1 = ref.backward(), new.backward()
same(g1, g0, 't priority gx')
same(new.parameters.grad_w, ref.parameters.grad_w, 't priority gw')
same(new.parameters.grad_b, ref.parameters.grad_b, 't priority gb')
passed_case('t priority: q and tau are ignored in CE')

# 7: KDでないときのtauは計算に影響しない（CE/推論）。
y0, l0 = ref.forward(x, t)
y1, l1 = new.forward(x, t, tau=0.0)  # CEでは温度は常に1
same_id(y1, y0, 'CE ignores tau IDs')
same(l1, l0, 'CE ignores tau loss')
g0, g1 = ref.backward(), new.backward()
same(g1, g0, 'CE ignores tau gx')
same(new.parameters.grad_w, ref.parameters.grad_w, 'CE ignores tau gw')
same(new.parameters.grad_b, ref.parameters.grad_b, 'CE ignores tau gb')
y0, z0 = ref.forward(x)
y1, z1 = new.forward(x, tau=5.0)
same_id(y1, y0, 'inference ignores tau IDs')
same(z1, z0, 'inference ignores tau maxlogit')
passed_case('tau ignored in ordinary CE and inference')

# 8: KDのW/b/x勾配について、実際の損失変化を有限差分で検証。
#   全語彙参照式による解析勾配との比較(#2)とは独立の確認。
shape, v, tau = (2, 3, 4), 9, 1.7
x = np.random.normal(0.0, 0.55, size=shape).astype(DTYPE)
w, b = parameters(shape[-1], v)
q = softmax(np.random.normal(0.0, 1.0,
                             size=shape[:-1] + (v,)).astype(DTYPE) / tau)
layer = make_layer(shape[-1], v, 3, w=w, b=b)
layer.forward(x, q=q, tau=tau)
gx = layer.backward()
gw = layer.parameters.grad_w.copy()
gb = layer.parameters.grad_b.copy()


def numerical_grad(arr, idx, eps=1.e-2):
    original = float(arr[idx])
    try:
        arr[idx] = original + eps
        _, lp = layer.forward(x, q=q, tau=tau)
        arr[idx] = original - eps
        _, lm = layer.forward(x, q=q, tau=tau)
    finally:
        arr[idx] = original
    return (lp - lm) / (2.0 * eps)


for idx in ((0, 0), (2, 7), (3, 8)):
    same(gw[idx], numerical_grad(layer.parameters.w, idx),
         'finite-difference W' + str(idx), rtol=2.e-2, atol=3.e-3)
for idx in ((0,), (5,), (8,)):
    same(gb[idx], numerical_grad(layer.parameters.b, idx),
         'finite-difference b' + str(idx), rtol=2.e-2, atol=3.e-3)
for idx in ((0, 0, 0), (0, 2, 2), (1, 1, 3)):
    same(gx[idx], numerical_grad(x, idx),
         'finite-difference x' + str(idx), rtol=2.e-2, atol=3.e-3)
passed_case('finite-difference KD gradients: selected W/b/x elements')

print('\nALL', passed, 'KD CASES PASSED')
print('KD数値検証。教師は模擬logitsから作った確率分布を使用。')
print('CE互換8ケース、非恒等PrePhaseは別テスト。')
