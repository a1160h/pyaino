from pyaino.Config import *
set_np('numpy'); np=Config.np  
from pyaino import Functions as F

class Std2:
    def __init__(self):
        self.var  = F.Var()
        self.sqrt = F.Sqrt()
    
    def forward(self, x):
        var = self.var(x)
        y   = self.sqrt(var)
        return y 
    
    def backward(self, dy):
        dvar = self.sqrt.backward(dy)
        dx   = self.var.backward(dvar)
        return dx 

functions =(Std2, F.Std)

x = np.array([1, 2, 3, 4, 5])
print(x)

for func in functions:
    print('##### test', func.__name__, '#####')
    model = func()
    y = model.forward(x)
    print("Forward y =", y)

    dy = 1 # 逆伝播の入力として、通常は損失関数の勾配が与えられる
    dx = model.backward(dy)
    print("Backward gx =", dx)
    
