from pyaino.Config import *
#set_np('numpy'); np=Config.np
set_derivative(True)
#Config.enable_debug_print=True
from pyaino import Functions as F

class Model_ref:
    """ Reference with regular Function """
    def __init__(self):
        self.func = F.Add()
        
    def forward(self, x1, x2):
        return self.func.forward(x1, x2)

    def backward(self, gy=1):
        return self.func.backward(gy)

class Model:
    """ Target with Variadic Function """
    def __init__(self):
        self.func = F.SumVariadic()
        
    def forward(self, x1, x2):
        return self.func.forward(x1, x2)

    def backward(self, gy=1):
        return self.func.backward(gy)


x1 = np.hdarray(range(2))
x2 = np.hdarray(range(2))
print('x1 =', x1)
print('x2 =', x2)
model_ref = Model_ref()
model     = Model()

print('\nreference forward')
yr = model_ref.forward(x1, x2)
print('yr =', yr)
print('backtrace')
yr.backtrace()
print('gx1 =', x1.grad)
print('gx2 =', x2.grad)
print('backward')
gx1, gx2 = model_ref.backward()
print('gx1 =', gx1)
print('gx2 =', gx2)

#x1.reset()
#x2.reset()
print('\ntarget forward')
y = model.forward(x1, x2)
print('y =', y)
print('backtrace')
y.backtrace()
print('gx1 =', x1.grad)
print('gx2 =', x2.grad)
print('backward')
gx1, gx2 = model.backward()
print('gx1 =', gx1)
print('gx2 =', gx2)

