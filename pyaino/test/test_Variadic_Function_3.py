from pyaino.Config import *
#set_np('numpy'); np=Config.np
from pyaino import Functions as F
#Config.enable_debug_print = True

x = np.arange(2)
print('x =', x)

model = F.SumVariadic()

y = model.forward([x for _ in range(2)])
print('y =', y)
gx = model.backward()
print('gx =', gx)

y = model.forward(*[x for _ in range(2)])
print('y =', y)
gx = model.backward()
print('gx =', gx)

y = model.forward(x for _ in range(2))
print('y =', y)
gx = model.backward()
print('gx =', gx)

y = model.forward((x for _ in range(2)))
print('y =', y)
gx = model.backward()
print('gx =', gx)

y = model.forward(tuple(x for _ in range(2)))
print('y =', y)
gx = model.backward()
print('gx =', gx)

y = model.forward(*(x for _ in range(2)))
print('y =', y)
gx = model.backward()
print('gx =', gx)

y = model.forward([x for _ in range(1)])
print('y =', y)
gx = model.backward()
print('gx =', gx)
