from pyaino.Config import *
set_np('numpy'); np=Config.np
from pyaino import HDFunctions as F
set_higher_derivative(True)    


print('Addのテスト')
x0 = np.hdarray(np.linspace(-2, 2, 5)) 
x1 = np.hdarray(np.linspace(2, -2, 5))
print('x0 =', x0, ', x1 =', x1)

func = F.Add()

y = func(x0, x1)
print('y =', y)

y.backtrace()
gx0, gx1 = x0.grad, x1.grad
gx0 = x0.grad
gx1 = x1.grad
print('### debug y.backtrace passed', func.__class__.__name__,
      '\ngx0 =', gx0, ', gx1 =', gx1)

gx0.backtrace()
gx0x0 = x0.grad
gx0x1 = x1.grad
print('### debug gx0.backtrace passed', func.__class__.__name__,
      '\ngx0x0 =', gx0x0, ', gx0x1 =', gx0x1)

gx1.backtrace()
gx1x0 = x0.grad
gx1x1 = x1.grad
print('### debug gx1.backtrace passed', func.__class__.__name__,
      '\ngx1x0 =', gx1x0, ', gx1x1 =', gx1x1)
