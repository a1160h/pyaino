# test_Var_veriously
# 20240306 井上
"""
結果の簡単にわかるケースで分散VARを求める
pyaino.Functionsに定義された関数を用いて様々な組み合わせでVarを求め、
それらおよび、Varを直接求めるものまで、その結果を比較する

"""

from pyaino.Config import *
set_np('numpy'); np=Config.np  
from pyaino import Functions as F

class Var0:
    def __init__(self):
        self.x = None
        self.N = None
        self.mean = None
    
    def forward(self, x):
        self.x = x
        y = np.var(x)
        return y
    
    def backward(self, dy):
        N = self.x.shape[0]
        mean = np.mean(self.x)
        dx = (2 / N) * (self.x - mean)
        return dx * dy

class Var1:
    def __init__(self):
        self.mean = F.Mean()
        self.sub  = F.Sub()
        self.pow  = F.Pow(2)
        self.sum  = F.Sum()
        self.div  = F.Div()
    
    def forward(self, x):
        self.x = x
        n = np.array(len(x))
        y = self.mean(x)
        y = self.sub(x, y)
        y = self.pow(y)
        y = self.sum(y)
        y = self.div(y, n)
        return y
    
    def backward(self, dy):
        dx, _ = self.div.backward(dy)
        dx = self.sum.backward(dx)
        dx = self.pow.backward(dx)
        dx0, dz = self.sub.backward(dx)
        dx1 = self.mean.backward(dz)
        dx = dx0 + dx1
        return dx

class Var2:
    def __init__(self):
        self.mean = F.Mean()
        self.sub  = F.Sub()
        self.sqs  = F.SquareSum()
        self.div  = F.Div()
    
    def forward(self, x):
        self.x = x
        n = np.array(len(x))
        y = self.mean(x)
        y = self.sub(x, y)
        y = self.sqs(y)
        y = self.div(y, n)
        return y
    
    def backward(self, dy):
        dx, _ = self.div.backward(dy)
        dx = self.sqs.backward(dx)
        dx0, dz = self.sub.backward(dx)
        dx1 = self.mean.backward(dz)
        dx = dx0 + dx1
        return dx

class Var3:
    def __init__(self):
        self.mean = F.Mean()
        self.sub  = F.Sub()
        self.sqm  = F.SquareMean()
    
    def forward(self, x):
        self.x = x
        n = np.array(len(x))
        y = self.mean(x)
        y = self.sub(x, y)
        y = self.sqm(y)
        return y
    
    def backward(self, dy):
        dx = self.sqm.backward(dy)
        dx0, dz = self.sub.backward(dx)
        dx1 = self.mean.backward(dz)
        dx = dx0 + dx1
        return dx

functions = (Var0, Var1, Var2, Var3, F.Var)

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

