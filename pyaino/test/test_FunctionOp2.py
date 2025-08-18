from pyaino.Config import *
from pyaino import Functions as F

functions = (F.Add, F.Sub, F.Mul, F.Div)

x0 = np.arange(6).reshape(3, 2)
x1 = np.arange(1, 3).reshape(1, 2)

print('x0\n', x0)
print('x1\n', x1)

for f in functions:
    func = f()
    print('=== test ', func.__class__.__name__, '===')
    y = func(x0, x1)
    print('result\n', y)

    gx0, gx1 = func.backward()
    print('backward result gx0\n', gx0)
    print('backward result gx1\n', gx1)
