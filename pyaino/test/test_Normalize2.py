from pyaino.Config import *
set_np('numpy'); np=Config.np  
from pyaino import HDFunctions as F


class Normalize:
    def __init__(self):
        self.mean = F.Mean()
        self.std  = F.Std()
        self.sub  = F.Sub()
        self.div  = F.Div()
    
    def forward(self, x):
        self.x = x
        mu  = self.mean(x)
        std = self.std(x)
        z = self.sub(x, mu)
        y = self.div(z, std)
        return y
    
    def backward(self, dy):
        dz, dstd = self.div.backward(dy)
        dx0, dmu = self.sub.backward(dz)
        dx1 = self.mean.backward(dmu)
        dx2 = self.std.backward(dstd)
        dx = dx0 + dx1 + dx2
        return dx

x = np.array([1, 2, 3, 4, 5])
print(x)
func = Normalize()
y = func.forward(x)
print("Forward y =", y)

dy = 1 #np.ones_like(y) #1 # 逆伝播の入力として、通常は損失関数の勾配が与えられる
dx = func.backward(dy)
print("Backward gx =", dx)

    
