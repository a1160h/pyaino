# test_rope_querychunk_equivalence.py
#
# Compare AttentionUnit and QueryChunkAttentionUnit.
# Main purpose:
#   Verify that RoPE behaves identically when Query-axis chunking is enabled.
#
# Conditions:
#   causality=True
#   dropout=0
#   regularizer=None
#
# The test checks:
#   1. forward output y
#   2. backward gradient gq
#   3. backward gradient gk
#   4. backward gradient gv
#
# for several chunk sizes.
#
# Run with current backend, or set BACKEND below to 'numpy' / 'cupy'.

from pyaino.Config import *

# ----------------------------------------------------------------------
# backend
# ----------------------------------------------------------------------
BACKEND = None
# BACKEND = 'numpy'
# BACKEND = 'cupy'

if BACKEND is not None:
    set_np(BACKEND)

np = Config.np

# Import after set_np so Neuron uses the selected backend.
from pyaino import Neuron


# ----------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------
SEED = 123

B = 2
T = 7
C = 8
HEAD = 2          # head_dim = 4 -> even, suitable for RoPE

CHUNK_SIZES = [1, 2, 3, 4, T]

ATOL = 1.0e-4
RTOL = 1.0e-4


def scalar(x):
    if hasattr(x, 'item'):
        return float(x.item())
    return float(x)


def max_error(a, b):
    return scalar(np.max(np.abs(a - b)))


def check_close(name, a, b):
    err = max_error(a, b)
    ok = bool(np.allclose(a, b, atol=ATOL, rtol=RTOL))
    print(f'{name:18s} max error = {err:.8e}', 'OK' if ok else 'NG')
    if not ok:
        raise AssertionError(
            f'{name} mismatch: max error={err}, '
            f'atol={ATOL}, rtol={RTOL}'
        )
    return err


def make_data():
    np.random.seed(SEED)

    q = np.random.randn(B, T, C).astype(Config.dtype)
    k = np.random.randn(B, T, C).astype(Config.dtype)
    v = np.random.randn(B, T, C).astype(Config.dtype)
    gy = np.random.randn(B, T, C).astype(Config.dtype)

    return q, k, v, gy


def run_reference(q, k, v, gy, rope):
    unit = Neuron.AttentionUnit(
        head=HEAD,
        causality=True,
        scale=True,
        rope=rope,
        regularizer=None,
    )

    y = unit.forward(
        q.copy(),
        k.copy(),
        v.copy(),
        dropout=0.0,
    )

    gq, gk, gv = unit.backward(gy.copy())

    return y.copy(), gq.copy(), gk.copy(), gv.copy()


def run_chunk(q, k, v, gy, rope, chunk_size):
    unit = Neuron.QueryChunkAttentionUnit(
        head=HEAD,
        causality=True,
        scale=True,
        rope=rope,
        regularizer=None,
        chunk_size=chunk_size,
    )

    y = unit.forward(
        q.copy(),
        k.copy(),
        v.copy(),
        dropout=0.0,
    )

    gq, gk, gv = unit.backward(gy.copy())

    return y.copy(), gq.copy(), gk.copy(), gv.copy()


def test_one(rope):
    print('\n' + '=' * 72)
    print('rope =', rope)
    print('=' * 72)

    q, k, v, gy = make_data()

    y_ref, gq_ref, gk_ref, gv_ref = run_reference(
        q, k, v, gy, rope
    )

    for chunk_size in CHUNK_SIZES:
        print(f'\n--- chunk_size = {chunk_size} ---')

        y, gq, gk, gv = run_chunk(
            q, k, v, gy,
            rope=rope,
            chunk_size=chunk_size,
        )

        check_close('forward y', y_ref, y)
        check_close('backward gq', gq_ref, gq)
        check_close('backward gk', gk_ref, gk)
        check_close('backward gv', gv_ref, gv)


def main():
    print('np           =', np.__name__)
    print('Config.dtype =', Config.dtype)
    print('shape        =', (B, T, C))
    print('head         =', HEAD)
    print('head_dim     =', C // HEAD)
    print('chunk sizes  =', CHUNK_SIZES)
    print('atol / rtol  =', ATOL, RTOL)

    # Regression check: existing QueryChunk path without RoPE.
    test_one(rope=False)

    # Main check: QueryChunk + RoPE.
    test_one(rope=True)

    print('\n' + '=' * 72)
    print('All AttentionUnit / QueryChunkAttentionUnit tests passed.')
    print('=' * 72)


if __name__ == '__main__':
    main()
