from pyaino.Config import *
set_np('numpy'); np = Config.np
from pyaino import Neuron
from pyaino import common_function as cf
from pyaino.nucleus import HDArray # 追加 20250327N
from pyaino import HDFunctions # 追加 20250327N
from pyaino import Activators # 追加 20250327N
from pyaino import LossFunctions # 追加 20250327N

set_derivative(True) # 追加 20250327N

batch_size=100
x = np.linspace(-5, 5, batch_size)
t = np.linspace(0, 1, batch_size)
x = x.reshape(batch_size, 1)
t = t.reshape(batch_size, 1)

# pyainoのクラスでシグモイド関数とクロスエントロピー誤差　
# func = Neuron.activate_function(activator='sigmoid')
# loss = Neuron.loss_function('x_entropy2')
# func_loss = Neuron.activate_function(activator='sigmoid_with_loss')

func = Activators.Sigmoid()
loss = LossFunctions.CrossEntropyError2()
func_loss = Activators.SigmoidWithLoss()

y = func.forward(x)
l = loss.forward(y, t)
z = func_loss.forward(x)

print(l)
gy = loss.backward(1)     # 損失関数から逆伝播
gx = func.backward(gy)       # gy=dldyをsigmoidに逆伝播
gz = func_loss.backward(t)   # sigmoid_with_lossでtから

# core関数で順伝播と逆伝播
x2 = HDArray(x)
t2 = HDArray(t)
y2 = 1 / (1 + HDFunctions.exp(-x2)) # シグモイド関数　
l2 = - sum(t2*HDFunctions.log(y2+1e-7)+(1-t2)*HDFunctions.log(1-y2+1e-7))/batch_size # クロスエントロピー誤差　
l2.backtrace()

# グラフで結果比較
cf.xy_graph(x.tolist(), y.tolist())
cf.xy_graph(x2.tolist(), y2.tolist())
cf.xy_graph(x.tolist(), t.tolist())
cf.xy_graph(x.tolist(), gy.tolist())
cf.xy_graph(x2.tolist(), y2.grad.tolist())
cf.xy_graph(x.tolist(), gx.tolist())
cf.xy_graph(x2.tolist(), x2.grad.tolist())
cf.xy_graph(x.tolist(), gz.tolist())

import matplotlib.pyplot as plt
plt.plot(x.tolist(), y.tolist(), label='y')
plt.plot(x.tolist(), t.tolist(), label='t')
plt.plot(x.tolist(), gy.tolist(), label='gy')
plt.plot(x.tolist(), gx.tolist(), label='gx')
plt.plot(x.tolist(), gz.tolist(), label='gz')
plt.legend()
plt.show()

plt.plot(x2.tolist(), y2.tolist(), label='y')
plt.plot(x2.tolist(), t.tolist(), label='t')
plt.plot(x2.tolist(), y2.grad.tolist(), label='gy')
plt.plot(x2.tolist(), x2.grad.tolist(), label='gx')
plt.legend()
plt.show()
