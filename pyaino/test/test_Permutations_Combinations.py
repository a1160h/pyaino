from pyaino.Config import *
from pyaino import Functions as F


x = np.arange(48).reshape(2, 4, 3, 2).astype(Config.dtype)

f = F.Permutations(1) # 対称軸は1
y = f(x)

gy = np.arange(y.size).reshape(*y.shape)
gx = f.backward(gy)

print('x:', x.shape, 'y:', y.shape, 'gx', gx.shape)

if input('内容を表示しますか=> ') in ('Y','y','Yes', 'yes'):
    print('x:\n', x)
    print('y:\n',y)
    print('gy:\n', gy)
    print('gx:\n', gx)

f = F.Combinations(1) # 対称軸は1
y = f(x)

gy = np.arange(y.size).reshape(*y.shape)
gx = f.backward(gy)

print('x:', x.shape, 'y:', y.shape, 'gx', gx.shape)

if input('内容を表示しますか=> ') in ('Y','y','Yes', 'yes'):
    print('x:\n', x)
    print('y:\n',y)
    print('gy:\n', gy)
    print('gx:\n', gx)
