# test_Regularizers7b_round_robin.py
# Round Robin:
#   NestedPairRoundRobin : schedule生成
#   Nest0                : PairDivergence用pair抽出
#   Nest1                : PairwiseGap用pair抽出
#   PairDivergence       : 自分のschedule/indexを保持
#   PairwiseGap          : 自分のschedule/indexを保持
#   AttentionRegularizer : 測定結果とrecordを管理

from pyaino.Config import *
from pyaino import Regularizers as R


np.random.seed(1234)

B, H, Tq, Tk = 2, 8, 5, 5
a = np.random.random((B, H, Tq, Tk)).astype(np.float32) + 0.2
a /= np.sum(a, axis=-1, keepdims=True)


# ------------------------------------------------------------
# 1. schedule
# ------------------------------------------------------------
schedule = R.NestedPairRoundRobin(H).make_schedule()

assert len(schedule) == 189
assert schedule[0] == [
    ((0, 1), (6, 7)),
    ((0, 2), (5, 7)),
]

nested_pairs = [
    nested_pair
    for step in schedule
    for nested_pair in step
]

canonical = lambda x: tuple(sorted(x))

assert len(nested_pairs) == 378
assert len({canonical(x) for x in nested_pairs}) == 378

for r in range(27):
    used = []
    for step in schedule[r*7:(r+1)*7]:
        for p, q in step:
            used.extend((p, q))

    assert len(used) == 28
    assert len(set(used)) == 28

print('[OK] round robin schedule')


# ------------------------------------------------------------
# 2. PairDivergence / PairwiseGap independently own same schedule/index
# ------------------------------------------------------------
jsd = R.JSDivergence(
    log_base=2,
    round_robin=True,
    n_head=H,
)
gap = R.PairwiseGap(
    gap=0.04,
    round_robin=True,
    n_head=H,
)

assert isinstance(jsd.take_pair, R.Nest0)
assert isinstance(gap.take_pair, R.Nest1)

assert jsd.take_pair.schedule == gap.take_pair.schedule
assert jsd.take_pair.schedule is not gap.take_pair.schedule

assert jsd.index == 0
assert gap.index == 0

print('[OK] independent schedule / index')


# ------------------------------------------------------------
# 3. first forward uses corresponding nest0 / nest1 selection
# ------------------------------------------------------------
result = jsd(a)
loss = gap(result)

assert jsd.take_pair.take.indices.shape == (4, 2)
assert np.array_equal(
    jsd.take_pair.take.indices,
    np.array([(0, 1), (6, 7), (0, 2), (5, 7)])
)

assert gap.take_pair.take.indices.shape == (2, 2)
assert np.array_equal(
    gap.take_pair.take.indices,
    np.array([(0, 1), (2, 3)])
)

assert jsd.index == 0
assert gap.index == 0

gy = gap.backward(1.0)
ga = jsd.backward(gy)

assert ga.shape == a.shape
assert jsd.index == 1
assert gap.index == 1

print('[OK] nest0 / nest1 first step')
print('[OK] independent indexes advance together')


# ------------------------------------------------------------
# 4. normal mode remains F.TakePair
# ------------------------------------------------------------
jsd_normal = R.JSDivergence(log_base=2)
gap_normal = R.PairwiseGap(gap=0.04)

result_normal = jsd_normal(a)
gap_normal(result_normal)

assert not isinstance(jsd_normal.take_pair, R.Nest0)
assert not isinstance(gap_normal.take_pair, R.Nest1)

print('[OK] normal mode')


# ------------------------------------------------------------
# 5. AttentionRegularizer has no RR object wiring
# ------------------------------------------------------------
jsd = R.JSDivergence(
    log_base=2,
    round_robin=True,
    n_head=H,
)
gap = R.PairwiseGap(
    gap=0.04,
    round_robin=True,
    n_head=H,
)

reg = R.AttentionRegularizer(
    divergence1=None,
    divergence2=jsd,
    regularize2=gap,
    axis2=(0, 2, 3),
    eta2=1.0,
)

reg(a)

assert jsd.index == 0
assert gap.index == 0

ga = reg.backward(1.0)

assert ga.shape == a.shape
assert jsd.index == 1
assert gap.index == 1
assert len(reg.get_record2()) == 0

print('[OK] AttentionRegularizer uses measurement/regularizer as configured')


# ------------------------------------------------------------
# 6. 7 backward -> one complete 28-JSD record
# ------------------------------------------------------------
jsd_base = R.JSDivergence(log_base=2)
result_base = jsd_base(a)
result_base = np.mean(result_base, axis=(0, 2, 3)).reshape(-1)

for _ in range(6):
    reg(a)
    reg.backward(1.0)

assert jsd.index == 7
assert gap.index == 7
assert len(reg.get_record2()) == 1

result_rr = reg.get_record2()[0]

assert result_rr.shape == (28,)
assert np.allclose(result_rr, result_base, rtol=1e-6, atol=1e-7)

print('[OK] 28-JSD record reconstruction')


# ------------------------------------------------------------
# 7. string configuration
# ------------------------------------------------------------
reg_str = R.AttentionRegularizer(
    divergence1=None,
    divergence2='JSDivergence(log_base=2, round_robin=True, n_head=8)',
    regularize2='PairwiseGap(gap=0.04, round_robin=True, n_head=8)',
    axis2=(0, 2, 3),
    eta2=1.0,
)

reg_str(a)
ga_str = reg_str.backward(1.0)

assert ga_str.shape == a.shape

print('[OK] string configuration')

print()
print('========== Regularizers7b RoundRobin smoke test passed ==========')
