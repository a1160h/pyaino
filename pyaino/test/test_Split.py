from pyaino.Config import *
from pyaino import Functions as F

model = F.Split([2, 5], axis=-1)

x = np.arange(16, dtype=Config.dtype).reshape(2, 8)
print(x)

y0, y1, y2 = model(x)

print(y0)
print(y1)
print(y2)

gy0 = np.ones(y0.shape, dtype=Config.dtype)
gy1 = np.ones(y1.shape, dtype=Config.dtype) * 2
gy2 = np.ones(y2.shape, dtype=Config.dtype) * 3

print(gy0)
print(gy1)
print(gy2)

gx = model.backward(gy0, gy1, gy2)

print(gx)
    
