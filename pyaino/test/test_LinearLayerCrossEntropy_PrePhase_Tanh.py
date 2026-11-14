# -*- coding: utf-8 -*-
"""LinearLayerCrossEntropy: 非恒等PrePhase (Tanh) の勾配テスト。

pyaino.Neuron 内の新版 LinearLayerCrossEntropy を直接使用する。
Neuron.py は書き換えない。Pyaino/CuPy または NumPy backend に対応。

目的:
    h = tanh(x)
    z = h @ W + b
    CE: L = mean(-log softmax(z)[t])
    KD: L = tau**2 * mean(KL(q || softmax(z/tau)))

特に、逆伝播が元の入力 x に対する勾配
    dL/dx = ((dL/dz) @ W.T) * (1 - h**2)
を返すか検査する。旧版で見つかった PrePhase 前後の不整合を検出するため、
テスト基準式は PrePhase / ScaleDotLinear を使わずに独立計算する。
"""
from pyaino.Config import *
from pyaino.Neuron import LinearLayerCrossEntropy

DTYPE = Config.dtype
RTOL = 8.e-5
ATOL = 8.e-5
passed = 0


def same(a, b, name, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=name)


def softmax(z):
    u = z - z.max(axis=-1, keepdims=True)
    e = np.exp(u)
    return e / e.sum(axis=-1, keepdims=True)


def reference(x, w, b, t=None, q=None, tau=1.0):
    """Tanh、全語彙Softmax、CE/KL、微分を独立した配列計算で求める。"""
    kd = t is None and q is not None
    T = tau if kd else 1.0
    h = np.tanh(x)
    z = h @ w
    if b is not None:
        z = z + b

    u = z / T
    u = u - u.max(axis=-1, keepdims=True)
    logp = u - np.log(np.exp(u).sum(axis=-1, keepdims=True))
    p = np.exp(logp)
    N = x.size // x.shape[-1]
    v = w.shape[-1]

    if kd:
        logq = np.log(np.maximum(q, np.asarray(1.e-30, dtype=DTYPE)))
        loss = T**2 * np.mean(np.sum(q * (logq - logp), axis=-1))
        gz = (p - q) * (T / N)
    else:
        lt = np.take_along_axis(logp, t[..., None], axis=-1).squeeze(-1)
        loss = -np.mean(lt)
        onehot = np.eye(v, dtype=DTYPE)[t]
        gz = (p - onehot) / N

    gh = gz @ w.T
    gx = gh * (1 - h*h)
    gw = h.reshape(-1, h.shape[-1]).T @ gz.reshape(-1, v)
    gb = gz.reshape(-1, v).sum(axis=0) if b is not None else None
    return z.argmax(axis=-1), loss, gx, gw, gb


def make_layer(d, v, tile, w, b):
    layer = LinearLayerCrossEntropy(
        d, v, matmul=True, bias=b is not None,
        activate='Tanh', tile_size=tile, optimize='SGD'
    )
    _ = layer.parameters()
    layer.parameters.w[...] = w
    if b is not None:
        layer.parameters.b[...] = b
    assert layer.prephase.activator is not None, 'Tanh が PrePhase に設定されていません'
    return layer


def run_case(label, shape, v, tile, tau=None, stream=False, bias=True):
    global passed
    d = shape[-1]
    x = np.random.normal(0, 0.9, size=shape).astype(DTYPE)
    w = np.random.normal(0, 0.3, size=(d, v)).astype(DTYPE)
    b = np.random.normal(0, 0.2, size=(v,)).astype(DTYPE) if bias else None

    # 恒等PrePhaseとの違いを明示的に確認する。
    h = np.tanh(x)
    assert float(np.max(np.abs(h - x))) > 0.1, '入力がTanhによって十分変化していません'

    if tau is None:
        t = np.random.randint(0, v, size=shape[:-1]).astype(np.int32)
        q = None
        args = {'t': t}
    else:
        t = None
        logits_t = np.random.normal(0, 1, size=shape[:-1] + (v,)).astype(DTYPE)
        q = softmax(logits_t / tau)
        if stream:
            def q_tile(start, end):
                return q[..., start:end]
            args = {'q': q_tile, 'tau': tau}
        else:
            args = {'q': q, 'tau': tau}

    layer = make_layer(d, v, tile, w, b)
    y_ref, l_ref, gx_ref, gw_ref, gb_ref = reference(x, w, b, t, q, tau or 1.0)
    y, loss = layer.forward(x, **args)
    np.testing.assert_array_equal(y, y_ref, err_msg=label + ': 予測ID')
    same(loss, l_ref, label + ': 損失')
    gx = layer.backward()
    same(gx, gx_ref, label + ': 入力勾配 gx')
    same(layer.parameters.grad_w, gw_ref, label + ': 重み勾配 gw')
    if bias:
        same(layer.parameters.grad_b, gb_ref, label + ': bias勾配 gb')

    passed += 1
    err = float(np.max(np.abs(gx - gx_ref)))
    print(' PASS', passed, '-', label, '  max|gx - ref| =', f'{err:.3e}')
    return layer, x, args, gx


def finite_difference_case(label, tau=None):
    """実装のlossから入力xの数値微分を求め、backward() と比較。"""
    global passed
    shape, v = (2, 3, 4), 11
    d = shape[-1]
    x = np.random.normal(0, 0.7, size=shape).astype(DTYPE)
    w = np.random.normal(0, 0.25, size=(d, v)).astype(DTYPE)
    b = np.random.normal(0, 0.12, size=(v,)).astype(DTYPE)
    layer = make_layer(d, v, 3, w, b)

    if tau is None:
        t = np.random.randint(0, v, size=shape[:-1]).astype(np.int32)
        args = {'t': t}
        q = None
    else:
        t = None
        q = softmax(np.random.normal(0, 1, size=shape[:-1] + (v,)).astype(DTYPE) / tau)
        args = {'q': q, 'tau': tau}

    layer.forward(x, **args)
    gx = layer.backward().copy()
    gx_ref = reference(x, w, b, t=t, q=q, tau=tau or 1.0)[2]
    same(gx, gx_ref, label + ': 解析微分')

    eps = 1.e-2  # float32 の中心差分
    indices = [(0, 0, 0), (0, 2, 2), (1, 1, 3), (1, 2, 0)]
    for idx in indices:
        xp = x.copy()
        xm = x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        _, lp = layer.forward(xp, **args)
        _, lm = layer.forward(xm, **args)
        fd = (lp - lm) / (2 * eps)
        same(gx[idx], fd, label + ': 有限差分 ' + str(idx),
             rtol=3.e-2, atol=3.e-3)

    passed += 1
    print(' PASS', passed, '-', label, ': x の有限差分', len(indices), '点')


print('Backend:', np.__name__)
print('Layer:', LinearLayerCrossEntropy.__module__ + '.' + LinearLayerCrossEntropy.__name__)
print('PrePhase: Tanh (非恒等), scale=False')

# CE: 非恒等PrePhase、タイル幅・入力次元・biasの違い。
run_case('CE 3D tile=1',        (2, 3, 4), 11, 1)
run_case('CE 3D tile=3',        (2, 3, 4), 11, 3)
run_case('CE 3D tile=None',     (2, 3, 4), 11, None)
run_case('CE 2D bias=False',    (5, 4),    9,  2, bias=False)

# KD: tau、教師qの取得形式、biasの違い。
run_case('KD 3D T=0.7 tile=1',  (2, 3, 4), 11, 1, tau=0.7)
run_case('KD 3D T=2 tile=3',    (2, 3, 4), 11, 3, tau=2.0)
run_case('KD 3D T=4 tile=None', (2, 3, 4), 11, None, tau=4.0)
run_case('KD 3D callback',      (2, 3, 4), 11, 3, tau=2.0, stream=True)
run_case('KD 2D bias=False',    (5, 4),    9,  2, tau=1.3, bias=False)

# 特に重要: 変換前xの微分を、lossの実際の変化量で検証。
finite_difference_case('CE Tanh', tau=None)
finite_difference_case('KD Tanh', tau=2.0)

print('\nALL', passed, 'TANH PREPHASE CASES PASSED')
print('Tanh 前処理を含む CE/KD の全語彙参照・入力勾配・有限差分を検証しました。')
