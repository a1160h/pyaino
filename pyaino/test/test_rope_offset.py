# test_rope_offset.py
#
# Position-offset RoPE tests, independent of KV cache / AttentionUnit.
# Checks full-sequence vs sliced-sequence equivalence, cache growth/reuse,
# default offset=0, rotation at a nonzero position, and backward gradients.
# Run with the current pyaino backend, or select NumPy / CuPy below.

from pyaino.Config import *
from pyaino.Neuron import RoPE

SEED = 123
ATOL = 1.0e-6
RTOL = 1.0e-5
ATOL_GRAD = 3.0e-3
EPS = 1.0e-3
MODES = ('interleaved', 'half')
BASES = (10000.0, 500000.0)


def scalar(x):
    if hasattr(x, 'get'):
        x = x.get()
    return float(x)


def check_close(name, a, b, atol=ATOL, rtol=RTOL):
    error = scalar(np.max(np.abs(a - b)))
    ok = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    print(f'{name:30s} max error = {error:.8e}', 'OK' if ok else 'NG')
    if not ok:
        raise AssertionError(f'{name}: max error={error}, atol={atol}, rtol={rtol}')


def make_data(shape=(2, 11, 8)):
    np.random.seed(SEED)
    q = np.random.randn(*shape).astype(Config.dtype)
    k = np.random.randn(*shape).astype(Config.dtype)
    return q, k


def test_default_offset(mode, base):
    print('\n--- test_default_offset ---')
    q, k = make_data()
    default = RoPE(base=base, mode=mode)
    explicit = RoPE(base=base, mode=mode)
    q0, k0 = default.forward(q.copy(), k.copy())
    q1, k1 = explicit.forward(q.copy(), k.copy(), offset=0)
    check_close('default / offset=0 q', q0, q1)
    check_close('default / offset=0 k', k0, k1)


def test_full_vs_slices(mode, base, shape):
    print('\n--- test_full_vs_slices ---')
    q, k = make_data(shape)
    T = q.shape[-2]
    full = RoPE(base=base, mode=mode)
    q_full, k_full = full.forward(q.copy(), k.copy(), offset=0)

    for start, stop in ((0, 1), (1, 5), (7, 10), (T - 1, T)):
        if stop > T:
            continue
        local = RoPE(base=base, mode=mode)
        qs, ks = local.forward(q[..., start:stop, :].copy(),
                               k[..., start:stop, :].copy(), offset=start)
        print(f'  slice=[{start}:{stop}], shape={qs.shape}')
        check_close('forward q', qs, q_full[..., start:stop, :])
        check_close('forward k', ks, k_full[..., start:stop, :])


def test_cache_reuse(mode, base):
    print('\n--- test_cache_reuse ---')
    q, k = make_data((2, 13, 8))
    ref_q, ref_k = RoPE(base=base, mode=mode).forward(q.copy(), k.copy())
    rope = RoPE(base=base, mode=mode)

    # Same instance, non-monotonic offsets and varying lengths:
    # this exercises both cache extension and cache reuse.
    for start, stop in ((7, 9), (2, 5), (10, 13), (0, 3), (6, 12)):
        qs, ks = rope.forward(q[..., start:stop, :].copy(),
                              k[..., start:stop, :].copy(), offset=start)
        print(f'  reuse slice=[{start}:{stop}]')
        check_close('cached q', qs, ref_q[..., start:stop, :])
        check_close('cached k', ks, ref_k[..., start:stop, :])


def test_nonzero_position(mode, base):
    print('\n--- test_nonzero_position ---')
    q, k = make_data((1, 1, 8))
    qr, kr = RoPE(base=base, mode=mode).forward(q.copy(), k.copy(), offset=7)
    difference = max(scalar(np.max(np.abs(qr - q))),
                     scalar(np.max(np.abs(kr - k))))
    print(f'nonzero position rotation      = {difference:.8e}',
          'OK' if difference > ATOL else 'NG')
    if difference <= ATOL:
        raise AssertionError('offset=7 unexpectedly behaved as position=0')


def numerical_grad(rope, q, k, gyq, gyk, target):
    src = q if target == 'q' else k
    grad = np.zeros_like(src)
    for idx in np.ndindex(src.shape):
        qp, kp = q.copy(), k.copy()
        qm, km = q.copy(), k.copy()
        if target == 'q':
            qp[idx] += EPS
            qm[idx] -= EPS
        else:
            kp[idx] += EPS
            km[idx] -= EPS
        yp_q, yp_k = rope.forward(qp, kp, offset=7)
        ym_q, ym_k = rope.forward(qm, km, offset=7)
        lp = np.sum(yp_q * gyq) + np.sum(yp_k * gyk)
        lm = np.sum(ym_q * gyq) + np.sum(ym_k * gyk)
        grad[idx] = (lp - lm) / (2.0 * EPS)
    return grad


def test_backward(mode, base):
    print('\n--- test_backward ---')
    q, k = make_data((1, 3, 8))
    np.random.seed(SEED + 1)
    gyq = np.random.randn(*q.shape).astype(Config.dtype)
    gyk = np.random.randn(*k.shape).astype(Config.dtype)

    rope = RoPE(base=base, mode=mode)
    rope.forward(q.copy(), k.copy(), offset=7)
    gq, gk = rope.backward(gyq.copy(), gyk.copy())

    num = RoPE(base=base, mode=mode)
    ngq = numerical_grad(num, q, k, gyq, gyk, 'q')
    ngk = numerical_grad(num, q, k, gyq, gyk, 'k')
    check_close('backward grad q', gq, ngq, atol=ATOL_GRAD, rtol=0)
    check_close('backward grad k', gk, ngk, atol=ATOL_GRAD, rtol=0)


def main():
    print('np           =', np.__name__)
    print('Config.dtype =', Config.dtype)
    print('atol / rtol  =', ATOL, RTOL)
    for mode in MODES:
        for base in BASES:
            print('\n' + '=' * 72)
            print('mode =', mode, 'base =', base)
            print('=' * 72)
            test_default_offset(mode, base)
            test_full_vs_slices(mode, base, (2, 11, 8))
            test_full_vs_slices(mode, base, (2, 3, 11, 8))
            test_cache_reuse(mode, base)
            test_nonzero_position(mode, base)
            test_backward(mode, base)
    print('\n' + '=' * 72)
    print('All RoPE position-offset tests passed.')
    print('=' * 72)


if __name__ == '__main__':
    main()
