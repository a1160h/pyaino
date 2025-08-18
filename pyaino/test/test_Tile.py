from pyaino.Config import *
from pyaino import Functions as f
set_derivative(True)

x = np.hdarray([[1, 2], [3, 4]])

y = np.tile(x, (1, 3))
print(y)

y2 = f.tile(x, (1, 3))
print(y2)

y2.backtrace()
print(x.grad)
