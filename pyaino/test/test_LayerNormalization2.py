from pyaino.Config import *
set_derivative(True)
from pyaino import Neuron
from pyaino import Functions as F
import matplotlib.pyplot as plt

# 入力はAttentionUnitの扱うデータ形状(B,T,h,H)を想定
x = np.arange(1*2*2*3).reshape(1,2,2,3)
axis = -1

print(x.shape)


from pyaino.nucleus import CompositFunction
class Normalize(CompositFunction):
    """ CompositFunctionによるNormalize """
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis
    
    def _forward(self, x):
        mu  = F.Mean(axis=self.axis, keepdims=True)(x)
        std = F.Std(axis=self.axis, keepdims=True)(x)
        z = F.Sub()(x, mu)
        y = F.Div()(z, std)
        return y

# 初期状態ではgamma=1とbeta=0だから、modelとmodel2の挙動は一致するはず
model = Neuron.LayerNormalization(axis=axis, scale_and_bias=True)
model2 = Normalize(axis=axis)  

# 順伝播
y = model.forward(x)
y2 = model2.forward(x)
print(y.shape, y2.shape)
assert np.allclose(y, y2)

# 逆伝播
gy = np.ones_like(y)
gx = model.backward(gy)
gx2 = model2.backward(gy)
print(gx.shape, gx2.shape)
assert np.allclose(gx, gx2)

# グラフで確認 順伝播
plt.plot(x.reshape(-1).tolist(), marker='x')
plt.plot(y.reshape(-1).tolist(), marker='x')
plt.show()

# グラフで確認 逆伝播
plt.plot(gx.reshape(-1).tolist(), marker='x')
plt.plot(model.scale_and_bias.ggamma.reshape(-1).tolist(), marker='x')
plt.plot(model.scale_and_bias.gbeta.reshape(-1).tolist(), marker='x')
plt.show()
