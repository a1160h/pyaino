# Qwen3-oriented dimension generalization test for pyaino SelfAttention.
# Runs with the backend already selected by pyaino.Config (NumPy or CuPy).
from pyaino.Config import *
from pyaino.Neuron import MultiHeadSelfAttention

set_seed(200)


def check(name, emb_dim, head_dim, n_head, exp_i, exp_o,
          *, chunk_size=None, rope=False, run_forward=True):
    kwargs = {'emb_dim': emb_dim, 'head_dim': head_dim, 'n_head': n_head,
              'causality': True, 'rope': rope, 'rope_mode': 'half'}
    if chunk_size is not None:
        kwargs['chunk_size'] = chunk_size
    layer = MultiHeadSelfAttention(**kwargs)
    layer.fix_configuration((2, 5, emb_dim))
    assert layer.linear_i.config == exp_i, (name, 'linear_i', layer.linear_i.config, exp_i)
    assert layer.linear_o.config == exp_o, (name, 'linear_o', layer.linear_o.config, exp_o)
    if run_forward:
        x = np.random.randn(2, 5, emb_dim).astype(Config.dtype)
        y = layer.forward(x, dropout=0.0)
        assert y.shape == x.shape, (name, 'forward', y.shape, x.shape)
        assert bool(np.all(np.isfinite(y))), (name, 'nonfinite forward')
        gy = np.random.randn(*y.shape).astype(Config.dtype)
        gx = layer.backward(gy)
        assert gx.shape == x.shape, (name, 'backward', gx.shape, x.shape)
        assert bool(np.all(np.isfinite(gx))), (name, 'nonfinite backward')
    print('PASS', name, 'linear_i=', layer.linear_i.config,
          'linear_o=', layer.linear_o.config)


print('Backend:', np.__name__)
# Old behavior: head_dim inferred from emb_dim / q_head.
check('legacy/MHA', 24, None, 4, (24, 72), (24, 24))
check('legacy/GQA', 24, 6, (4, 2), (24, 48), (24, 24))

# New behavior: head dimension does not have to equal emb_dim / q_head.
check('expanded/GQA', 24, 8, (4, 2), (24, 64), (32, 24))
check('reduced/GQA', 24, 4, (4, 2), (24, 32), (16, 24))
check('expanded/GQA-chunk', 24, 8, (4, 2), (24, 64), (32, 24), chunk_size=2)
check('expanded/GQA-RoPE', 24, 8, (4, 2), (24, 64), (32, 24), rope=True)

# Numeric gradient check for the generalized GQA dimensions.
# Uses the user's configured backend; only a few input coordinates are sampled.
layer = MultiHeadSelfAttention(emb_dim=8, head_dim=6, n_head=(2, 1),
                               causality=True)
x = (np.random.randn(1, 3, 8) * 0.1).astype(Config.dtype)
gy = (np.random.randn(1, 3, 8) * 0.1).astype(Config.dtype)
layer.forward(x, dropout=0.0)
gx = layer.backward(gy)
eps = 1e-2
for idx in [(0, 0, 0), (0, 1, 4), (0, 2, 7)]:
    xp, xm = x.copy(), x.copy()
    xp[idx] += eps
    xm[idx] -= eps
    fp = float(np.sum(layer.forward(xp, dropout=0.0) * gy))
    fm = float(np.sum(layer.forward(xm, dropout=0.0) * gy))
    numerical = (fp - fm) / (2 * eps)
    analytical = float(gx[idx])
    assert abs(analytical - numerical) <= 5e-3 + 5e-2 * abs(numerical), (
        'numerical gradient mismatch', idx, analytical, numerical)
print('PASS numerical input gradient (generalized GQA)')

# Configuration-only tests: no 0.6B/4B weight allocation.
check('Qwen3-0.6B/config', 1024, 128, (16, 8),
      (1024, 4096), (2048, 1024), run_forward=False)
check('Qwen3-4B/config', 2560, 128, (32, 8),
      (2560, 6144), (4096, 2560), run_forward=False)

# Inferred head_dim must preserve old divisibility restriction.
layer = MultiHeadSelfAttention(emb_dim=25, n_head=4)
try:
    layer.fix_configuration((2, 5, 25))
except ValueError:
    print('PASS invalid implicit head_dim')
else:
    raise AssertionError('Expected ValueError for emb_dim=25 and n_head=4')

print('ALL CHECKS PASSED')
