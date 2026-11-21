"""QK-Norm numerical compatibility and Backward regression tests (REV3).

Extends test_qknorm_selfattention.py (REV2):
  - old behaviour / GQA / RoPE / QueryChunk / Qwen3 shapes unchanged;
  - independent Qwen3-equivalent FP32 RMSNorm reference (forward, dx, dgamma);
  - comprehensive finite differences for SelfAttention x, q_gamma, k_gamma;
  - verifies Q/K gamma update from SelfAttention.update().

Run: python test_qknorm_selfattention_REV3.py
No torch, transformers or additional packages required. Uses Config.np, as pyaino does.
Requires the *already applied* SelfAttention dimension-generalization and QK-Norm patches.
"""
from pyaino.Config import *
from pyaino.Neuron import MultiHeadSelfAttention, RMSNormalization

set_seed(200)

print('Backend:', np.__name__)
print('QK-Norm numerical/Backward tests (REV3)')


def scalar(x):
    return float(x)


def assert_close(name, actual, expected, *, atol=3e-5, rtol=1e-4):
    a = np.asarray(actual)
    b = np.asarray(expected)
    assert a.shape == b.shape, (name, 'shape mismatch', a.shape, b.shape)
    assert bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(b))), (name, 'non-finite')
    err = np.abs(a - b)
    bound = atol + rtol * np.abs(b)
    maximum = scalar(np.max(err))
    if not bool(np.all(err <= bound)):
        idx = tuple(int(v) for v in np.unravel_index(int(np.argmax(err)), err.shape))
        raise AssertionError((name, 'value mismatch', 'max_abs=', maximum,
                              'index=', idx, 'actual=', scalar(a[idx]),
                              'expected=', scalar(b[idx]), 'atol=', atol, 'rtol=', rtol))
    return maximum


def rms_reference(x, gamma, eps=1e-6):
    """Reference Qwen3RMSNorm: weighted RMS across last (head_dim) axis."""
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    y = x * inv * gamma
    return y, inv


def rms_backward_reference(x, gamma, gy, eps=1e-6):
    """Independent analytic derivative, without pyaino's RMSNormalization.backward.

    inv = (mean(x*x)+eps)^(-1/2)
    dx = gy*gamma*inv - x*inv^3*mean(gy*gamma*x)
    dgamma = sum(gy*x*inv) over all dimensions except head_dim.
    """
    _, inv = rms_reference(x, gamma, eps)
    dx = gy * gamma * inv - x * inv**3 * np.mean(gy * gamma * x, axis=-1, keepdims=True)
    axes = tuple(range(x.ndim - 1))
    dgamma = np.sum(gy * x * inv, axis=axes)
    return dx, dgamma


def head_reference(x, gamma, heads, dh, eps=1e-6):
    b, t, _ = x.shape
    y, _ = rms_reference(x.reshape(b, t, heads, dh), gamma.reshape(1, 1, 1, dh), eps)
    return y.reshape(b, t, -1)


def build(**kwargs):
    config = dict(emb_dim=12, head_dim=4, n_head=(4, 2),
                  causality=True, rope=False, rope_mode='half')
    config.update(kwargs)
    return MultiHeadSelfAttention(**config)


# ----- Original REV2 checks (retained) -----
set_seed(201)
a = build()
x = (np.random.randn(2, 5, 12) * .2).astype(Config.dtype)
ya = a.forward(x, dropout=0.)
set_seed(201)
b = build(qk_norm=False)
x2 = (np.random.randn(2, 5, 12) * .2).astype(Config.dtype)
assert_close('same input', x, x2, atol=0, rtol=0)
yb = b.forward(x2, dropout=0.)
assert_close('legacy output', ya, yb, atol=0, rtol=0)
assert a.q_norm is None and b.q_norm is None
print('PASS legacy behavior unchanged')


def exercise(name, *, rope=False, chunk_size=None):
    opts = dict(qk_norm=True, qk_norm_eps=1e-6, rope=rope)
    if chunk_size is not None:
        opts['chunk_size'] = chunk_size
    model = build(**opts)
    x = (np.random.randn(2, 4, 12) * .2).astype(Config.dtype)
    model.forward(x, dropout=0.)
    qnorm, knorm = model.q_norm, model.k_norm
    assert qnorm is not knorm
    assert qnorm.beta is None and knorm.beta is None
    assert qnorm.gamma.size == knorm.gamma.size == 4
    assert qnorm.gamma.shape[-1] == 4
    qgamma = np.asarray([.5, 1.1, 1.4, .8], dtype=Config.dtype)
    kgamma = np.asarray([1.3, .6, .9, 1.2], dtype=Config.dtype)
    qnorm.gamma[...] = qgamma.reshape(qnorm.gamma.shape)
    knorm.gamma[...] = kgamma.reshape(knorm.gamma.shape)

    # Reference Q/K RMSNorm inserted just before existing AttentionUnit.
    z = model.linear_i.forward(x)
    q, k, v = np.split(z, [16, 24], axis=-1)
    q = head_reference(q, qgamma, 4, 4)
    k = head_reference(k, kgamma, 2, 4)
    y_ref = model.linear_o.forward(model.attention.forward(q, k, v, dropout=0.))
    y = model.forward(x, dropout=0.)
    max_y = assert_close(name + '/headwise forward', y, y_ref)

    gy = (np.random.randn(*y.shape) * .2).astype(Config.dtype)
    gx = model.backward(gy)
    assert gx.shape == x.shape and bool(np.all(np.isfinite(gx)))
    assert bool(np.all(np.isfinite(qnorm.ggamma)))
    assert bool(np.all(np.isfinite(knorm.ggamma)))

    # Keep REV2's sampled finite differences for each combination.
    for label, param, index, analytic in [
        ('q_gamma', qnorm.gamma.reshape(-1), 2, scalar(qnorm.ggamma.reshape(-1)[2])),
        ('k_gamma', knorm.gamma.reshape(-1), 1, scalar(knorm.ggamma.reshape(-1)[1])),
        ('input', x.reshape(-1), 5, scalar(gx.reshape(-1)[5])),
    ]:
        old = scalar(param[index])
        delta = .005
        try:
            param[index] = old + delta
            f_plus = scalar(np.sum(model.forward(x, dropout=0.) * gy))
            param[index] = old - delta
            f_minus = scalar(np.sum(model.forward(x, dropout=0.) * gy))
        finally:
            param[index] = old
        numeric = (f_plus - f_minus) / (2 * delta)
        assert abs(analytic - numeric) < 3e-3 + .05 * abs(numeric), (
            name, label, analytic, numeric)
    print('PASS', name, 'Q/K gamma shape=', qnorm.gamma.shape,
          'forward max_abs=', f'{max_y:.3g}')


exercise('GQA/QK-Norm')
exercise('GQA/QK-Norm/RoPE', rope=True)
exercise('GQA/QK-Norm/QueryChunk', chunk_size=2)
exercise('GQA/QK-Norm/QueryChunk/RoPE', rope=True, chunk_size=2)

for label, emb, hq, hk in [('0.6B', 1024, 16, 8), ('4B', 2560, 32, 8)]:
    model = MultiHeadSelfAttention(emb_dim=emb, head_dim=128, n_head=(hq, hk),
                                   qk_norm=True, qk_norm_eps=1e-6)
    model.fix_configuration((1, 1, emb))
    assert model.q_norm is not None and model.k_norm is not None
    assert model.linear_o.config == (hq * 128, emb)
    print('PASS Qwen3-' + label + '/configuration')


# ----- NEW: independent numerical compatibility of Qwen3's Q/K RMSNorm -----
def check_norm_reference(label, heads, *, small=False):
    dh = 4
    x = (np.random.randn(2, 3, heads, dh) * (.0001 if small else .7)).astype(Config.dtype)
    gy = (np.random.randn(*x.shape) * .5).astype(Config.dtype)
    gamma = np.asarray([.6, 1.25, .75, 1.4], dtype=Config.dtype)
    layer = RMSNormalization(axis=-1, scale_and_bias='scale', eps=1e-6)
    layer.forward(x)  # initialize gamma on the actual configured backend
    assert layer.sb and not layer.use_beta and layer.beta is None
    assert layer.gamma.size == dh
    layer.gamma[...] = gamma.reshape(layer.gamma.shape)
    y = layer.forward(x)
    gx = layer.backward(gy)
    y_ref, _ = rms_reference(x, gamma.reshape(1, 1, 1, dh))
    gx_ref, gg_ref = rms_backward_reference(x, gamma.reshape(1, 1, 1, dh), gy)
    err_y = assert_close(label + '/forward', y, y_ref, atol=8e-6, rtol=5e-5)
    err_dx = assert_close(label + '/dx', gx, gx_ref, atol=1e-4, rtol=3e-4)
    err_dw = assert_close(label + '/dgamma', layer.ggamma.reshape(-1), gg_ref,
                          atol=1e-4, rtol=3e-4)
    print(f'PASS {label}: max_abs(y, dx, dgamma) = '
          f'({err_y:.3g}, {err_dx:.3g}, {err_dw:.3g})')


check_norm_reference('Qwen3 Q-Norm headwise reference', 4)
check_norm_reference('Qwen3 K-Norm headwise reference', 2)
check_norm_reference('Qwen3 Q-Norm near-zero input/eps', 4, small=True)


# Optional additional check against the *actual* Hugging Face Qwen3RMSNorm.
# This is an independent framework comparison, not a required dependency.
def hf_qwen3_norm_check():
    try:
        import torch
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    except (ImportError, OSError) as e:
        print('SKIP optional Hugging Face Qwen3RMSNorm comparison:', e)
        return

    import numpy as cpu_np

    def host(a):
        return cpu_np.asarray(a.get() if hasattr(a, 'get') else a)

    for name, heads in [('Q', 4), ('K', 2)]:
        x = (np.random.randn(2, 3, heads, 4) * .5).astype(Config.dtype)
        gy = (np.random.randn(*x.shape) * .3).astype(Config.dtype)
        gamma = np.asarray([.7, 1.2, .8, 1.35], dtype=Config.dtype)

        py_norm = RMSNormalization(axis=-1, scale_and_bias='scale', eps=1e-6)
        py_norm.forward(x)
        py_norm.gamma[...] = gamma.reshape(py_norm.gamma.shape)
        py_y = py_norm.forward(x)
        py_dx = py_norm.backward(gy)

        layer = Qwen3RMSNorm(4, eps=1e-6).float()
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(host(gamma), dtype=torch.float32))
        xt = torch.tensor(host(x), dtype=torch.float32, requires_grad=True)
        gt = torch.tensor(host(gy), dtype=torch.float32)
        yt = layer(xt)
        (yt * gt).sum().backward()

        e_y = assert_close(name + '/HF Qwen3 forward', py_y, np.asarray(yt.detach().numpy()))
        e_dx = assert_close(name + '/HF Qwen3 dx', py_dx, np.asarray(xt.grad.numpy()),
                            atol=1e-4, rtol=3e-4)
        e_dw = assert_close(name + '/HF Qwen3 dgamma', py_norm.ggamma.reshape(-1),
                            np.asarray(layer.weight.grad.numpy()), atol=1e-4, rtol=3e-4)
        print(f'PASS HF Qwen3 {name}-Norm: max_abs(y, dx, dgamma) = '
              f'({e_y:.3g}, {e_dx:.3g}, {e_dw:.3g})')


hf_qwen3_norm_check()


# ----- NEW: full finite-difference gradients through the complete attention -----
def exhaustive_gradient_check(label, *, rope=False, chunk_size=None):
    # Small enough to differentiate *every* input and gamma component.
    kwargs = dict(emb_dim=8, head_dim=4, n_head=(2, 1),
                  qk_norm=True, qk_norm_eps=1e-6,
                  causality=True, rope=rope, rope_mode='half')
    if chunk_size is not None:
        kwargs['chunk_size'] = chunk_size
    model = MultiHeadSelfAttention(**kwargs)
    x = (np.random.randn(1, 3, 8) * .4).astype(Config.dtype)
    model.forward(x, dropout=0.)
    qnorm, knorm = model.q_norm, model.k_norm
    qnorm.gamma[...] = np.asarray([.8, 1.1, .65, 1.35], dtype=Config.dtype).reshape(qnorm.gamma.shape)
    knorm.gamma[...] = np.asarray([1.3, .55, 1.15, .9], dtype=Config.dtype).reshape(knorm.gamma.shape)
    gy = (np.random.randn(*x.shape) * .3).astype(Config.dtype)

    model.forward(x, dropout=0.)
    gx = model.backward(gy).copy()
    gq = qnorm.ggamma.copy()
    gk = knorm.ggamma.copy()

    def objective():
        return scalar(np.sum(model.forward(x, dropout=0.) * gy))

    def numerical_gradient(param, delta=.004):
        result = np.empty_like(param)
        flat = param.reshape(-1)
        out = result.reshape(-1)
        for i in range(param.size):
            original = scalar(flat[i])
            try:
                flat[i] = original + delta
                fp = objective()
                flat[i] = original - delta
                fm = objective()
            finally:
                flat[i] = original
            out[i] = (fp - fm) / (2 * delta)
        return result

    dx_num = numerical_gradient(x)
    dq_num = numerical_gradient(qnorm.gamma)
    dk_num = numerical_gradient(knorm.gamma)
    err_dx = assert_close(label + '/all dx', gx, dx_num, atol=.002, rtol=.025)
    err_dq = assert_close(label + '/all qgamma', gq, dq_num, atol=.002, rtol=.025)
    err_dk = assert_close(label + '/all kgamma', gk, dk_num, atol=.002, rtol=.025)
    print(f'PASS {label}: gradient max_abs(dx, dqgamma, dkgamma) = '
          f'({err_dx:.3g}, {err_dq:.3g}, {err_dk:.3g}); '
          f'components=({x.size},{qnorm.gamma.size},{knorm.gamma.size})')


exhaustive_gradient_check('all-gradients GQA/QK-Norm')
exhaustive_gradient_check('all-gradients GQA/QK-Norm/RoPE/QueryChunk',
                          rope=True, chunk_size=2)


# ----- NEW: ensure Q/K gamma participate in SelfAttention.update() -----
model = MultiHeadSelfAttention(emb_dim=8, head_dim=4, n_head=(2, 1),
                               causality=True, qk_norm=True, qk_norm_eps=1e-6)
x = (np.random.randn(2, 4, 8) * .5).astype(Config.dtype)
y = model.forward(x, dropout=0.)
model.backward((np.random.randn(*y.shape) * .5).astype(Config.dtype))
q0 = model.q_norm.gamma.copy()
k0 = model.k_norm.gamma.copy()
assert bool(np.any(np.abs(model.q_norm.ggamma) > 1e-8)), 'Q gamma gradient is zero'
assert bool(np.any(np.abs(model.k_norm.ggamma) > 1e-8)), 'K gamma gradient is zero'
model.update(eta=.01)
assert bool(np.any(model.q_norm.gamma != q0)), 'Q gamma was not updated'
assert bool(np.any(model.k_norm.gamma != k0)), 'K gamma was not updated'
print('PASS Q/K independent gamma parameters updated via SelfAttention.update()')

print('ALL CHECKS PASSED (QK-Norm REV3)')
