from pyaino.Config import *
from pyaino.nucleus import HDArray, CompositFunction
from pyaino import Functions as F
import matplotlib.pyplot as plt

Config.enable_debug_print=True

class TestFunc(CompositFunction):
    def _forward(self, x, z):
        y = F.Add()(x, 1)
        return y

x = np.linspace(0, 4, 5)
z = np.linspace(4, 0, 5)

func = TestFunc()
y = func.forward(x, z)
gx, gz = func.backward()
         
print(x)
print(z)
print(y)
print(gx)
print(gz)

plt.plot(x.tolist(), label='x')
plt.plot(z.tolist(), label='z')
plt.plot(y.tolist(), label='y=f(x, z)')
plt.plot(gx.tolist(), label='gx')
plt.plot(gz.tolist(), label='gz')
plt.legend()
plt.show()
