from pyaino.Config import *
#set_np('numpy'); np=Config.np
set_derivative(True)
from pyaino import Functions as F

class Model:
    """ Variadic関数に引数を様々な渡し方をする """
    def __init__(self):
        self.func = F.SumVariadic()
        
    def forward0(self, x):
        return self.func.forward([x for _ in range(2)])

    def forward1(self, x):
        return self.func.forward(*[x for _ in range(2)])

    def forward2(self, x):
        return self.func.forward(x for _ in range(2))

    def forward3(self, x):
        return self.func.forward((x for _ in range(2)))

    def forward4(self, x):
        return self.func.forward(tuple(x for _ in range(2)))

    def forward5(self, x):
        return self.func.forward(*(x for _ in range(2)))

    def forward6(self, x):
        return self.func.forward([x for _ in range(1)])


x = np.arange(2)
x = np.hdarray(x)
print('x =', x)
model = Model()
print('\nforward0')
y = model.forward0(x)
print('y =', y)
y.backtrace(); print('gx =', x.grad)
print('\nforward1')
y = model.forward1(x)
print('y =', y)
y.backtrace(); print('gx =', x.grad)
print('\nforward2')
y = model.forward2(x)
print('y =', y)
y.backtrace(); print('gx =', x.grad)
print('\nforward3')
y = model.forward3(x)
print('y =', y)
y.backtrace(); print('gx =', x.grad)
print('\nforward4')
y = model.forward4(x)
print('y =', y)
y.backtrace(); print('gx =', x.grad)
print('\nforward5')
y = model.forward5(x)
print('y =', y)
y.backtrace(); print('gx =', x.grad)
print('\nforward6')
y = model.forward6(x)
print('y =', y)
y.backtrace(); print('gx =', x.grad)

