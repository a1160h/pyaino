from pyaino.Config import *
from pyaino import Functions as F

model = F.Concatenate(-1)
x0 = np.arange(4).reshape(2,-1)
x1 = np.arange(8).reshape(2,-1)
x2 = np.arange(6).reshape(2,-1)

print(x0)
print(x1)
print(x2)

y = model(x0,x1,x2)

print(y)

gy = np.arange(y.size).reshape(*y.shape)
print(gy)
gx0, gx1, gx2 = model.backward(gy)

print(gx0)
print(gx1)
print(gx2)
    
