"""
新しい settings/result/record 内部構造が正しく機能するかの軽量テスト

"""
# smoke_test_attention_regularizer_normalized_settings_clean.py
# Regularizers_normalized_settings_clean.py を Regularizers.py として反映後に実行

from pyaino.Config import *
from pyaino import Regularizers as R


np.random.seed(1234)

B, H, Tq, Tk = 2, 8, 5, 5
a = np.random.random((B, H, Tq, Tk)).astype(np.float32) + 0.2
a /= np.sum(a, axis=-1, keepdims=True)

reg = R.AttentionRegularizer(
    divergence1=R.EntropyDivergence(),
    regularize1=None,
    axis1=(0,2,3),
    eta1=0,

    divergence2='JSDivergence()',
    regularize2='PairwiseGap(gap=0.1)',
    scheduler2=None,
    axis2=(0,2,3),
    eta2=1.0,
)

assert len(reg.settings) == 3

loss = reg(a)

s1 = reg.settings[0]
s2 = reg.settings[1]
s3 = reg.settings[2]

assert s1['result'] is not None
assert s2['result'] is not None
assert s3['result'] is None

# forwardだけではrecordされない
assert len(reg.get_record1()) == 0
assert len(reg.get_record2()) == 0
assert len(reg.get_record3()) == 0

assert s2['take_pair_d'] is s2['divergence'].take_pair
assert s2['take_pair_r'] is s2['regularize'].take_pair

ga = reg.backward(1.0)
assert ga.shape == a.shape

# backwardまで到達したforwardだけrecordされる
assert len(reg.get_record1()) == 1
assert len(reg.get_record2()) == 1
assert len(reg.get_record3()) == 0

reg(a)

# 2回目もforwardだけでは増えない
assert len(reg.get_record1()) == 1
assert len(reg.get_record2()) == 1

reg.backward(1.0)

assert len(reg.get_record1()) == 2
assert len(reg.get_record2()) == 2

print('result1 shape =', s1['result'].shape)
print('result2 shape =', s2['result'].shape)
print('record lengths =', len(reg.get_record1()), len(reg.get_record2()))
print('gradient shape =', ga.shape)
print('========== normalized settings clean smoke test passed ==========')
