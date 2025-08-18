from pyaino.Config import *
from pyaino.nucleus import HDArray, CompositFunction
from pyaino import Functions as F
import matplotlib.pyplot as plt

#Config.enable_debug_print=True

class TanComposit(CompositFunction):
    """ 関数の組合わせ """
    def _forward(self, x):
        sinx = F.sin(x)
        cosx = F.cos(x)
        y = sinx / cosx
        return y

class TanComposit2(CompositFunction):
    """ 自身のメソッドを使う """
    def _forward(self, x):
        sinx = self.sin(x)
        cosx = self.cos(x)
        y = F.div(sinx, cosx)
        return y

    def sin(self, x):
        return F.sin(x)

    def cos(self, x):
        return F.cos(x)


x = np.linspace(-1, 1, 10)

tan = TanComposit()
y = tan.forward(x)
gx = tan.backward()
         
plt.plot(x.tolist(), y.tolist())
plt.plot(x.tolist(), gx.tolist())
plt.show()

tan = TanComposit2()
y = tan.forward(x)
gx = tan.backward()
         
plt.plot(x.tolist(), y.tolist())
plt.plot(x.tolist(), gx.tolist())
plt.show()
