from pyaino.Config import *
#set_np('numpy'); np=Config.np
set_derivative(True)
from pyaino import Functions as F
#Config.enable_debug_print = True

x = np.arange(2)
x = np.hdarray(x)
print('x =', x)

model = F.SumVariadic()

y = model.forward([x for _ in range(2)])
print('y =', y)
y.backtrace()
print('gx =', x.grad)

y = model.forward(*[x for _ in range(2)])
print('y =', y)
y.backtrace()
print('gx =', x.grad)

y = model.forward(x for _ in range(2))
print('y =', y)
y.backtrace()
print('gx =', x.grad)

y = model.forward((x for _ in range(2)))
print('y =', y)
y.backtrace()
print('gx =', x.grad)

y = model.forward(tuple(x for _ in range(2)))
print('y =', y)
y.backtrace()
print('gx =', x.grad)

y = model.forward(*(x for _ in range(2)))
print('y =', y)
y.backtrace()
print('gx =', x.grad)

y = model.forward([x for _ in range(1)])
print('y =', y)
y.backtrace()
print('gx =', x.grad)
