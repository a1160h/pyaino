"""pyaino SelfAttention bias tuple compatibility / numerical tests.

Four settings: (input_projection_bias, output_projection_bias)
  (False, True): existing pyaino default
  (False, False): Qwen3/Llama-style
  (True, False):  Qwen2-style
  (True, True):   GPT-2/OPT-style

Checks:
  - actual presence/absence and shapes of bias parameters;
  - independent NumPy/CuPy grouped causal-attention forward reference;
  - all input gradients, selected projection-weight gradients, and all active
    bias gradients by finite differences;
  - default (omitted bias) vs explicitly specified (False, True) equivalence;
  - all combinations become identical with zero biases and identical weights;
  - bias updates, and combinations with QK-Norm + RoPE + QueryChunk.

Run with the NumPy/CuPy backend pyaino already configured to use.
No torch, transformers, or model checkpoint is needed.

Requires SelfAttention with bias=(bias_i, bias_o) support.
This script does not modify pyaino.
"""
from pyaino.Config import *
from pyaino.Neuron import MultiHeadSelfAttention

set_seed(200)

EMB, DH, HQ, HK = 6, 4, 2, 1
BIAS_MODES = ((False, True), (False, False), (True, False), (True, True))
EPS = 0.004


def to_float(a):
    return float(a)


def max_abs(a):
    return to_float(np.max(np.abs(a)))


def close(label, actual, expected, *, atol=4e-5, rtol=3e-4):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (label, 'shape', a.shape, b.shape)
    assert bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(b))), (label, 'non-finite')
    err = np.abs(a - b)
    ok = np.all(err <= atol + rtol * np.abs(b))
    if not bool(ok):
        ind = int(np.argmax(err))
        raise AssertionError(
            f'{label}: max_abs={max_abs(err):.5g}, '
            f'actual[{ind}]={to_float(a.reshape(-1)[ind]):.6g}, '
            f'expected[{ind}]={to_float(b.reshape(-1)[ind]):.6g}, '
            f'atol={atol}, rtol={rtol}')
    return max_abs(err)


def build(mode=None, **extra):
    args = dict(emb_dim=EMB, head_dim=DH, n_head=(HQ, HK),
                causality=True, rope=False, optimize='SGD')
    args.update(extra)
    if mode is not None:
        args['bias'] = mode
    try:
        return MultiHeadSelfAttention(**args)
    except (TypeError, ValueError) as exc:
        if mode is not None:
            raise AssertionError(
                f'Failed to construct SelfAttention(bias={mode}). '
                'Check that SelfAttention.__init__ pops the bias tuple '
                'before passing kwargs into LinearLayer/AttentionUnit.') from exc
        raise


def reference(x, wi, wo, bi=None, bo=None):
    """Independent grouped causal scaled-dot-product attention (without RoPE/QK-Norm)."""
    bsz, seq, _ = x.shape
    z = np.matmul(x, wi)
    if bi is not None:
        z = z + bi
    q, k, v = np.split(z, [HQ * DH, (HQ + HK) * DH], axis=-1)
    q = q.reshape(bsz, seq, HQ, DH).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq, HK, DH).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq, HK, DH).transpose(0, 2, 1, 3)
    k = np.repeat(k, HQ // HK, axis=1)
    v = np.repeat(v, HQ // HK, axis=1)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * (DH ** -0.5)
    tril = np.tril(np.ones((seq, seq), dtype=bool))[None, None, :, :]
    scores = np.where(tril, scores, np.float32(-1e9))
    ex = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    att = ex / np.sum(ex, axis=-1, keepdims=True)
    y = np.matmul(att, v).transpose(0, 2, 1, 3).reshape(bsz, seq, HQ * DH)
    y = np.matmul(y, wo)
    if bo is not None:
        y = y + bo
    return y


def finite_difference(layer, x, gy, param, indices, *, label):
    """Compare a saved analytic gradient to finite differences of sum(y*gy)."""
    def objective():
        return to_float(np.sum(layer.forward(x, dropout=0.) * gy))

    flat_param = param.reshape(-1)
    worst = 0.0
    for i in indices:
        old = to_float(flat_param[i])
        try:
            flat_param[i] = old + EPS
            pos = objective()
            flat_param[i] = old - EPS
            neg = objective()
        finally:
            flat_param[i] = old
        numeric = (pos - neg) / (2 * EPS)
        analytic = to_float(label[1].reshape(-1)[i])
        diff = abs(numeric - analytic)
        tolerance = 2e-3 + 0.03 * abs(numeric)
        if diff > tolerance:
            raise AssertionError(
                f'{label[0]} component={i}: analytic={analytic:.7g}, '
                f'numeric={numeric:.7g}, abs_error={diff:.4g}, '
                f'tolerance={tolerance:.4g}')
        worst = max(worst, diff)
    return worst


def samples(n):
    return sorted({0, n // 5, n // 2, 4 * n // 5, n - 1})


def install(layer, x, wi, wo, bi, bo):
    layer.forward(x, dropout=0.)  # lazy parameter initialization
    pi, po = layer.linear_i.parameters, layer.linear_o.parameters
    close('linear_i size', np.asarray(pi.w.shape), np.asarray(wi.shape), atol=0, rtol=0)
    close('linear_o size', np.asarray(po.w.shape), np.asarray(wo.shape), atol=0, rtol=0)
    pi.w[...] = wi
    po.w[...] = wo
    if layer.linear_i.bias:
        assert pi.b is not None and pi.b.shape == bi.shape
        pi.b[...] = bi
    else:
        assert pi.b is None, 'input bias=False must not allocate bias vector'
    if layer.linear_o.bias:
        assert po.b is not None and po.b.shape == bo.shape
        po.b[...] = bo
    else:
        assert po.b is None, 'output bias=False must not allocate bias vector'
    return pi, po


def inspect_grads(mode, layer, x, gy):
    pi, po = layer.linear_i.parameters, layer.linear_o.parameters
    layer.forward(x, dropout=0.)
    gx = layer.backward(gy).copy()
    assert gx.shape == x.shape and bool(np.all(np.isfinite(gx)))
    assert pi.grad_w.shape == pi.w.shape and po.grad_w.shape == po.w.shape
    assert bool(np.all(np.isfinite(pi.grad_w))) and bool(np.all(np.isfinite(po.grad_w)))
    if not mode[0]:
        assert pi.grad_b is None, 'unexpected input bias gradient'
    if not mode[1]:
        assert po.grad_b is None, 'unexpected output bias gradient'
    if mode[1]:
        close(f'{mode}/output bias analytic sum', po.grad_b,
              np.sum(gy, axis=(0, 1)), atol=2e-5, rtol=1e-4)

    errors = {}
    errors['dx'] = finite_difference(layer, x, gy, x, range(x.size),
                                     label=(f'{mode}/dx', gx))
    errors['dwi'] = finite_difference(layer, x, gy, pi.w, samples(pi.w.size),
                                      label=(f'{mode}/dwi', pi.grad_w.copy()))
    errors['dwo'] = finite_difference(layer, x, gy, po.w, samples(po.w.size),
                                      label=(f'{mode}/dwo', po.grad_w.copy()))
    if mode[0]:
        errors['dbi'] = finite_difference(layer, x, gy, pi.b, range(pi.b.size),
                                          label=(f'{mode}/dbi', pi.grad_b.copy()))
    if mode[1]:
        errors['dbo'] = finite_difference(layer, x, gy, po.b, range(po.b.size),
                                          label=(f'{mode}/dbo', po.grad_b.copy()))

    # The numerical-difference evaluations overwrite cached forward states.
    # Restore the correct state before testing the optimizer update path.
    layer.forward(x, dropout=0.)
    layer.backward(gy)
    old_bi = pi.b.copy() if mode[0] else None
    old_bo = po.b.copy() if mode[1] else None
    layer.update(eta=0.02)
    if mode[0]:
        assert max_abs(pi.b - old_bi) > 1e-8, f'{mode} input bias not updated'
    if mode[1]:
        assert max_abs(po.b - old_bo) > 1e-8, f'{mode} output bias not updated'
    return errors


def main():
    print('Backend:', np.__name__, ' dtype:', Config.dtype)
    print('SelfAttention bias tuple: 4-mode numerical + backward tests')

    # Identical data/weights for every mode, so any difference comes from bias.
    set_seed(2026)
    x = (np.random.randn(1, 3, EMB) * 0.7).astype(Config.dtype)
    gy = (np.random.randn(*x.shape) * 0.6).astype(Config.dtype)
    wi = (np.random.randn(EMB, (HQ + 2 * HK) * DH) * 0.45).astype(Config.dtype)
    wo = (np.random.randn(HQ * DH, EMB) * 0.45).astype(Config.dtype)
    bi = np.linspace(-0.23, 0.19, wi.shape[1]).astype(Config.dtype)
    bo = np.linspace(0.31, -0.14, EMB).astype(Config.dtype)

    zero_outputs = []
    zero_dxs = []
    for mode in BIAS_MODES:
        layer = build(mode)
        assert (layer.linear_i.bias, layer.linear_o.bias) == mode, (
            mode, 'actual layer bias flags',
            layer.linear_i.bias, layer.linear_o.bias)
        pi, po = install(layer, x, wi, wo, bi, bo)
        y = layer.forward(x, dropout=0.)
        expected = reference(x, wi, wo, bi if mode[0] else None,
                             bo if mode[1] else None)
        forward_error = close(f'{mode}/independent forward', y, expected)

        # Every mode must produce exactly the same result with the active
        # bias vectors set to zero and identical projection weights.
        if mode[0]:
            pi.b[...] = 0
        if mode[1]:
            po.b[...] = 0
        z = layer.forward(x, dropout=0.).copy()
        g = layer.backward(gy).copy()
        zero_outputs.append(z)
        zero_dxs.append(g)
        if mode[0]:
            pi.b[...] = bi
        if mode[1]:
            po.b[...] = bo
        layer.forward(x, dropout=0.)
        grad_errors = inspect_grads(mode, layer, x, gy)
        print(f'PASS {mode}: forward max_abs={forward_error:.3g}; '
              + ', '.join(f'{name}={err:.3g}' for name, err in grad_errors.items()))

    for i, mode in enumerate(BIAS_MODES[1:], start=1):
        close(f'zero-bias output {mode}', zero_outputs[i], zero_outputs[0], atol=3e-6)
        close(f'zero-bias dx {mode}', zero_dxs[i], zero_dxs[0], atol=3e-6)
    print('PASS 4 modes: zero biases give identical outputs and dx')

    # Legacy: omitted bias must be exactly equivalent to (False, True).
    default_layer = build()
    explicit_layer = build((False, True))
    for layer in (default_layer, explicit_layer):
        install(layer, x, wi, wo, bi, bo)
    yd = default_layer.forward(x, dropout=0.)
    ye = explicit_layer.forward(x, dropout=0.)
    close('legacy/default forward', yd, ye, atol=0, rtol=0)
    gd = default_layer.backward(gy)
    ge = explicit_layer.backward(gy)
    close('legacy/default dx', gd, ge, atol=0, rtol=0)
    for tag in ('linear_i', 'linear_o'):
        d = getattr(default_layer, tag).parameters
        e = getattr(explicit_layer, tag).parameters
        close(f'legacy/{tag} dw', d.grad_w, e.grad_w, atol=0, rtol=0)
        if d.bias:
            close(f'legacy/{tag} db', d.grad_b, e.grad_b, atol=0, rtol=0)
    print('PASS legacy: omitted bias == explicit (False, True), forward/backward/gradients')

    # Regression: the bias tuple must coexist with the already added QK-Norm,
    # half-RoPE and QueryChunk implementations.
    for mode in BIAS_MODES:
        layer = build(mode, qk_norm=True, qk_norm_eps=1e-6,
                      rope=True, rope_mode='half', chunk_size=2)
        y = layer.forward(x, dropout=0.)
        gx = layer.backward(gy)
        assert y.shape == gx.shape == x.shape
        assert bool(np.all(np.isfinite(y))) and bool(np.all(np.isfinite(gx)))
        assert (layer.linear_i.bias, layer.linear_o.bias) == mode
        assert layer.q_norm is not None and layer.k_norm is not None
    print('PASS all 4 modes with QK-Norm + RoPE(half) + QueryChunk')
    print('ALL CHECKS PASSED (SelfAttention bias tuple)')


if __name__ == '__main__':
    main()
