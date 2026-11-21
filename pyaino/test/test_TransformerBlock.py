"""pyaino TransformerBlock constructor generalization / Qwen3 compatibility tests.

Requires the 2026-11-21 SelfAttention fixes (GQA independent head_dim,
QK-Norm, bias tuple) and the TransformerBlock constructor patch in this bundle.
Runs against the installed pyaino and its configured NumPy/CuPy backend.
No Qwen weights, PyTorch, or Transformers installation is required.
"""
from pyaino.Config import *
from pyaino import nucleus
from pyaino import Neuron as nn
from pyaino import stems_blocks_heads as blocks

set_seed(200)
EPS = 0.004

def max_abs(a):
    return float(np.max(np.abs(a)))


def close(label, x, y, atol=4e-5, rtol=4e-4):
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape, (label, 'shape', x.shape, y.shape)
    assert bool(np.all(np.isfinite(x))) and bool(np.all(np.isfinite(y))), (label, 'nonfinite')
    err = np.abs(x - y)
    if not bool(np.all(err <= atol + rtol * np.abs(y))):
        raise AssertionError(f'{label}: max_abs={max_abs(err):.5g}')
    return max_abs(err)


class LegacyTransformerBlock(blocks.TransformerBlock):
    """Previous __init__ reconstructed for side-by-side compatibility checks.

    Forward/backward/update are inherited from the actual TransformerBlock.
    """
    def __init__(self, emb_dim=64, n_head=4, causality=None, proj=False, ffwd='ffn',
                 intermediate=None, expansion=4, rms=False, activate='Mish',
                 chunk_size=None, **kwargs):
        nucleus.Function.__init__(self)
        self.sa = nn.MultiHeadSelfAttention(
            emb_dim, emb_dim // n_head, n_head,
            causality=causality, chunk_size=chunk_size, **kwargs)
        if ffwd == 'ffn':
            self.ffwd = blocks.FeedForward(
                emb_dim, intermediate, expansion, activate=activate, **kwargs)
        elif ffwd == 'swiglu':
            self.ffwd = blocks.SwiGLU(
                emb_dim, intermediate, expansion, activate=activate, **kwargs)
        elif ffwd == 'ccff':
            self.ffwd = blocks.CausalConvFeedForward(
                emb_dim, intermediate, expansion, activate=activate, **kwargs)
        else:
            raise NotImplementedError(ffwd)
        Norm = nn.RMSNormalization if rms else nn.LayerNormalization
        self.ln1 = Norm(**kwargs)
        self.ln2 = Norm(**kwargs)
        self.proj = nn.LinearLayer(emb_dim, **kwargs) if proj else None
        self.proj_used = False


def legacy_check(ffwd, rms):
    # Original defaults must remain unchanged, including legacy SwiGLU's
    # activate='Mish' forwarded to each LinearLayer PrePhase.
    args = dict(emb_dim=8, n_head=2, causality=True,
                intermediate=16, ffwd=ffwd, rms=rms, optimize='SGD')
    set_seed(511)
    x = (np.random.randn(1, 3, 8) * 0.6).astype(Config.dtype)
    gy = (np.random.randn(*x.shape) * 0.3).astype(Config.dtype)
    set_seed(710)
    old = LegacyTransformerBlock(**args)
    y_old = old.forward(x.copy(), dropout=0.0).copy()
    gx_old = old.backward(gy.copy()).copy()
    set_seed(710)
    new = blocks.TransformerBlock(**args)
    y_new = new.forward(x.copy(), dropout=0.0).copy()
    gx_new = new.backward(gy.copy()).copy()
    ey = close(f'legacy {ffwd}/forward', y_new, y_old, atol=0, rtol=0)
    eg = close(f'legacy {ffwd}/backward', gx_new, gx_old, atol=0, rtol=0)
    if ffwd == 'swiglu':
        assert new.ffwd.linear_g.prephase.activator is not None, (
            'Legacy SwiGLU pre-activation must be preserved')
    assert new.sa.linear_i.bias is False and new.sa.linear_o.bias is True
    print(f'PASS legacy/{ffwd} (rms={rms}): forward={ey:.3g}, dx={eg:.3g}')


def build_small(*, chunk_size=None):
    return blocks.TransformerBlock(
        emb_dim=8,
        n_head=(4, 2),
        head_dim=4,                 # 4*4 != emb_dim=8
        causality=True,
        chunk_size=chunk_size,
        ffwd='swiglu', intermediate=20,
        activate=None,             # Qwen3: no extra pre-activation
        rms=True, scale_and_bias='scale', eps=1e-6,
        bias=(False, False),
        qk_norm=True, qk_norm_eps=1e-6,
        rope=True, rope_mode='half', rope_base=1000000.0,
        optimize='SGD')


def inspect_small(block, x):
    y = block.forward(x.copy(), dropout=0.0)
    assert y.shape == x.shape
    assert block.sa.config == (8, 4, (4, 2, 2)), block.sa.config
    assert block.sa.linear_i.config == (8, 32)
    assert block.sa.linear_o.config == (16, 8)
    assert block.sa.linear_i.bias is False
    assert block.sa.linear_o.bias is False
    assert block.sa.q_norm is not None and block.sa.k_norm is not None
    assert block.sa.q_norm.gamma.size == 4 and block.sa.k_norm.gamma.size == 4
    assert block.ffwd.config == (8, 20, 4), block.ffwd.config
    assert block.ln1.sb and not block.ln1.use_beta
    assert block.ln2.sb and not block.ln2.use_beta
    assert block.ln1.gamma.size == 8 and block.ln2.gamma.size == 8
    assert block.ln1.eps == 1e-6 and block.ln2.eps == 1e-6
    for linear in (block.ffwd.linear_g, block.ffwd.linear_i, block.ffwd.linear_o):
        assert linear.bias is False, 'Qwen3 SwiGLU must have bias=False'
        assert linear.prephase.activator is None, (
            'Extra pre-activation found in SwiGLU. Set TransformerBlock activate=None.')
    return y


def numeric_gradient(block, x, gy, param, grad, indices, label):
    """FP32 centered differences of sum(Block.forward(x)*gy)."""
    f = param.reshape(-1)
    g = grad.reshape(-1)
    worst = 0.0
    for i in indices:
        before = float(f[i])
        try:
            f[i] = before + EPS
            a = float(np.sum(block.forward(x.copy(), dropout=0.0) * gy))
            f[i] = before - EPS
            b = float(np.sum(block.forward(x.copy(), dropout=0.0) * gy))
        finally:
            f[i] = before
        num = (a - b) / (2 * EPS)
        ana = float(g[i])
        err = abs(ana - num)
        tol = 0.0025 + 0.035 * abs(num)
        if err > tol:
            raise AssertionError(f'{label}[{i}]: analytical={ana:.6g} '
                                 f'numerical={num:.6g} err={err:.3g} tol={tol:.3g}')
        worst = max(worst, err)
    return worst


def qwen_small_check():
    set_seed(812)
    x = (np.random.randn(1, 3, 8) * 0.5).astype(Config.dtype)
    gy = (np.random.randn(*x.shape) * 0.35).astype(Config.dtype)
    set_seed(900)
    block = build_small()
    y = inspect_small(block, x)
    gx = block.backward(gy.copy()).copy()
    errors = {}
    errors['dx'] = numeric_gradient(block, x, gy, x, gx, range(x.size), 'dx')
    parts = {
        'ln1.gamma': (block.ln1.gamma, block.ln1.ggamma),
        'ln2.gamma': (block.ln2.gamma, block.ln2.ggamma),
        'q_norm.gamma': (block.sa.q_norm.gamma, block.sa.q_norm.ggamma),
        'k_norm.gamma': (block.sa.k_norm.gamma, block.sa.k_norm.ggamma),
        'gate.weight': (block.ffwd.linear_g.parameters.w,
                        block.ffwd.linear_g.parameters.grad_w),
    }
    for label, (p, grad) in parts.items():
        assert grad is not None, f'No gradient produced for {label}'
        indices = range(p.size) if p.size <= 10 else sorted({0, p.size // 2, p.size - 1})
        errors[label] = numeric_gradient(block, x, gy, p, grad.copy(), indices, label)

    # Numeric-difference forward passes overwrite caches; refresh for update.
    block.forward(x.copy(), dropout=0.0)
    block.backward(gy.copy())
    old_gamma = block.ln1.gamma.copy()
    old_gate = block.ffwd.linear_g.parameters.w.copy()
    block.update(eta=0.01)
    assert max_abs(block.ffwd.linear_g.parameters.w - old_gate) > 0, 'FFN weight not updated'
    assert max_abs(block.ln1.gamma - old_gamma) > 0, 'RMSNorm gamma not updated'
    print('PASS Qwen3-style mini Block forward/backward/update: ' +
          ', '.join(f'{k}={v:.3g}' for k, v in errors.items()))

    # QueryChunk must preserve results for identical parameters and inputs.
    set_seed(900)
    full = build_small()
    a = inspect_small(full, x).copy()
    set_seed(900)
    chunked = build_small(chunk_size=2)
    b = inspect_small(chunked, x).copy()
    ec = close('QueryChunk vs full Attention', b, a, atol=5e-5, rtol=5e-4)
    gc = chunked.backward(gy.copy())
    set_seed(900)
    full = build_small()
    full.forward(x.copy(), dropout=0.0)
    gf = full.backward(gy.copy())
    eg = close('QueryChunk vs full Backward', gc, gf, atol=5e-5, rtol=5e-4)
    print(f'PASS QK-Norm + RoPE(half) + QueryChunk: forward={ec:.3g}, dx={eg:.3g}')


def config_checks():
    for label, emb, qh, kvh, inter in (
        ('Qwen3-0.6B', 1024, 16, 8, 3072),
        ('Qwen3-4B', 2560, 32, 8, 9728)):
        block = blocks.TransformerBlock(
            emb_dim=emb, n_head=(qh, kvh), head_dim=128,
            ffwd='swiglu', intermediate=inter, activate=None,
            causality=True, rms=True, scale_and_bias='scale', eps=1e-6,
            bias=(False, False), qk_norm=True, qk_norm_eps=1e-6,
            rope=True, rope_mode='half', rope_base=1000000.0)
        block.sa.fix_configuration((1, 1, emb))  # dimensions only; no huge weights
        assert block.sa.linear_i.config == (emb, (qh + 2*kvh)*128)
        assert block.sa.linear_o.config == (qh*128, emb)
        assert block.ffwd.config == (emb, inter, 4)
        print(f'PASS {label} configuration: qkv={block.sa.linear_i.config}, '
              f'output={block.sa.linear_o.config}, ffwd={block.ffwd.config}')


def main():
    print('Backend:', np.__name__, 'dtype:', Config.dtype)
    print('TransformerBlock constructor / Qwen3 compatibility tests')
    legacy_check('ffn', rms=False)
    legacy_check('swiglu', rms=True)
    qwen_small_check()
    config_checks()
    print('ALL CHECKS PASSED (TransformerBlock Qwen3 constructor)')


if __name__ == '__main__':
    main()
