from pyaino.Config import *
from pyaino import Functions as F


# ------------------------------------------------
# Stack
# ------------------------------------------------

model = F.Stack(axis=-1)

x0 = np.arange(6, dtype=Config.dtype).reshape(2, 3)
x1 = x0 + 10
x2 = x0 + 20

print('x0:\n', x0)
print('x1:\n', x1)
print('x2:\n', x2)

y = model(x0, x1, x2)
print('y:\n', y)

gy = np.arange(y.size, dtype=Config.dtype).reshape(y.shape)
print('gy:\n', gy)

gx0, gx1, gx2 = model.backward(gy)

print('gx0:\n', gx0)
print('gx1:\n', gx1)
print('gx2:\n', gx2)


# ------------------------------------------------
# Unstack
# ------------------------------------------------

model = F.Unstack(axis=-1)

y0, y1, y2 = model(y)

print('y0:\n', y0)
print('y1:\n', y1)
print('y2:\n', y2)

gy0 = np.ones(y0.shape, dtype=Config.dtype)
gy1 = np.ones(y1.shape, dtype=Config.dtype) * 2
gy2 = np.ones(y2.shape, dtype=Config.dtype) * 3

gx = model.backward(gy0, gy1, gy2)
print('gx:\n', gx)

