# smoke_test_regularizers_takepair_backward.py
# AttentionRegularizer による F.TakePair 一元管理
# forward / backward smoke test

from pyaino.Config import *
from pyaino import Regularizers as R


np.random.seed(1234)

B, H, Tq, Tk = 2, 8, 5, 5
a = np.random.random((B, H, Tq, Tk)).astype(np.float32) + 0.2
a /= np.sum(a, axis=-1, keepdims=True)


# ------------------------------------------------------------
# 1. 従来相当:
#    JSDivergence -> PairwiseGap を直接接続
# ------------------------------------------------------------
jsd_base = R.JSDivergence()
gap_base = R.PairwiseGap(gap=0.1)

result_base = jsd_base(a)
loss_base = gap_base(result_base)

g_result_base = gap_base.backward(1.0)
g_a_base = jsd_base.backward(g_result_base)


# ------------------------------------------------------------
# 2. AttentionRegularizer 管理
# ------------------------------------------------------------
jsd = R.JSDivergence()
gap = R.PairwiseGap(gap=0.1)

take_pair_d_before = jsd.take_pair
take_pair_r_before = gap.take_pair

reg = R.AttentionRegularizer(
    divergence1=None,
    divergence2=jsd,
    regularize2=gap,
    eta2=1.0,
)

loss_managed = reg(a)
g_a_managed = reg.backward(1.0)


print('loss base/managed =', float(loss_base), float(loss_managed))
print('result shape      =', result_base.shape, reg.result2.shape)
print('gradient shape    =', g_a_base.shape, g_a_managed.shape)

print('max loss diff   =', float(np.max(np.abs(loss_base - loss_managed))))
print('max result diff =',
      float(np.max(np.abs(result_base.reshape(-1) - reg.result2.reshape(-1)))))
print('max grad diff   =', float(np.max(np.abs(g_a_base - g_a_managed))))

assert np.allclose(loss_base, loss_managed)
assert np.allclose(result_base.reshape(-1), reg.result2.reshape(-1))
assert np.allclose(g_a_base, g_a_managed)

assert jsd.take_pair is reg.take_pair_d2
assert gap.take_pair is reg.take_pair_r2
assert jsd.take_pair is not take_pair_d_before
assert gap.take_pair is not take_pair_r_before

assert np.array_equal(
    jsd_base.take_pair.take.indices,
    jsd.take_pair.take.indices,
)
assert np.array_equal(
    gap_base.take_pair.take.indices,
    gap.take_pair.take.indices,
)

print('[OK] managed TakePair replacement')
print('[OK] forward compatibility')
print('[OK] backward compatibility')


# ------------------------------------------------------------
# 3. 文字列指定でも backward まで一致
# ------------------------------------------------------------
reg_str = R.AttentionRegularizer(
    divergence1=None,
    divergence2='JSDivergence()',
    regularize2='PairwiseGap(gap=0.1)',
    eta2=1.0,
)

loss_str = reg_str(a)
g_a_str = reg_str.backward(1.0)

assert np.allclose(loss_managed, loss_str)
assert np.allclose(g_a_managed, g_a_str)

print('[OK] string configuration forward/backward')


# ------------------------------------------------------------
# 4. 数値微分スポットチェック
# ------------------------------------------------------------
def loss_function(x):
    jsd = R.JSDivergence()
    gap = R.PairwiseGap(gap=0.1)
    return float(gap(jsd(x)))


indices = [
    (0, 0, 0, 0),
    (0, 3, 2, 4),
    (1, 7, 4, 1),
    (1, 2, 1, 3),
    (0, 5, 4, 2),
]

eps = 1e-4
errors = []

for index in indices:
    ap = a.copy()
    am = a.copy()

    ap[index] += eps
    am[index] -= eps

    g_numeric = (loss_function(ap) - loss_function(am)) / (2 * eps)
    g_analytic = float(g_a_base[index])

    error = abs(g_numeric - g_analytic)
    errors.append(error)

    print(
        'finite diff', index,
        'analytic =', g_analytic,
        'numeric =', g_numeric,
        'error =', error,
    )

print('finite diff max error =', max(errors))
assert max(errors) < 5e-4

print()
print('========== deeper smoke test passed ==========')
