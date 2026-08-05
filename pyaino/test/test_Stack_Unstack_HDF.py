from pyaino.Config import *
set_higher_derivative(True)
from pyaino import HDFunctions as F

print('\n##### Stack / Unstack HDF test #####')

x0 = np.hdarray([
    [1.0, 2.0],
    [3.0, 4.0],
])

x1 = np.hdarray([
    [5.0, 6.0],
    [7.0, 8.0],
])

# forward: Stack
y = F.stack((x0, x1), axis=1)

expected_y = np.stack(
    (np.array(x0), np.array(x1)),
    axis=1,
)

print('y =\n', y)
assert np.allclose(y, expected_y)

# loss = sum(y ** 2)
loss = F.sum(F.square(y))

# first derivative:
# Stack.__backward__()からUnstackが呼ばれる
loss.backtrace()

gx0 = x0.grad
gx1 = x1.grad

print('gx0 =\n', gx0)
print('gx1 =\n', gx1)

assert np.allclose(gx0, 2.0 * np.array(x0))
assert np.allclose(gx1, 2.0 * np.array(x1))

# second derivative:
# Unstackの全出力を明示的に二階微分の対象とする
grad_sum = F.sum(gx0) + F.sum(gx1)
grad_sum.backtrace()

ggx0 = x0.grad
ggx1 = x1.grad

print('ggx0 =\n', ggx0)
print('ggx1 =\n', ggx1)

assert np.allclose(ggx0, np.full_like(x0, 2.0))
assert np.allclose(ggx1, np.full_like(x1, 2.0))
