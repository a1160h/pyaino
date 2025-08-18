from pyaino.Config import *
from pyaino.nucleus import HDArray, CompositFunction
from pyaino import Functions as F
import matplotlib.pyplot as plt

Config.enable_debug_print=True

class CrossEntropyErrorNS(CompositFunction):
    """ 関数のインスタンス化に注意（同一関数は別にする必要あり） """

    def _forward(self, y, t):
        l1 = F.Mul()(t, F.Log()(y)) # +1e-7は困る
        l2 = F.Mul()(F.Sub()(1, t), F.Log()(F.Sub()(1, y)))
        l = F.Sum()(F.Add()(l1, l2))
        l = F.Neg()(l)
        return l

class CrossEntropyErrorNS2(CompositFunction):
    """ 演算子オーバーロードを使って """

    def _forward(self, y, t):
        l = - F.sum(t * F.log(y) + (1 - t) * F.log(1 - y))
        return l


y = np.linspace(0.1, 0.9, 10)
t = np.linspace(0.9, 0.1, 10)

loss = CrossEntropyErrorNS()
l = loss.forward(y, t)
gy, gt = loss.backward()
         
plt.plot(y.tolist(), label='y')
plt.plot(t.tolist(), label='t')
plt.plot(gy.tolist(), label='gy')
plt.legend()
plt.show()

loss = CrossEntropyErrorNS2()
l = loss.forward(y, t)
gy, gt = loss.backward()
         
plt.plot(y.tolist(), label='y')
plt.plot(t.tolist(), label='t')
plt.plot(gy.tolist(), label='gy')
plt.legend()
plt.show()
