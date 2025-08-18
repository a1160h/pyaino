from pyaino.Config import *
#set_np('numpy'); np = Config.np
from pyaino import common_function as cf
from pyaino import Activators # 追加 20250327N
from pyaino import LossFunctions # 追加 20250327N
from pyaino.nucleus import HDArray # 追加 20250327N
from pyaino import HDFunctions # 追加 20250327N

set_derivative(True) # 追加 20250327N

batch_size=100
x = np.linspace(-5, 5, batch_size)
t = np.linspace(0, 1, batch_size)
x = x.reshape(batch_size, 1)
t = t.reshape(batch_size, 1)

# pyainoのクラスでtanh関数と二乗和誤差　
# func = neuron.activate_function(activator='tanh')
# loss = neuron.loss_function('MSE')

# 書き換え 20250327N
func = Activators.Tanh()
loss = LossFunctions.MeanSquaredError()

y = func.forward(x)
l = loss.forward(y, t)

print(l)
gy = loss.backward(1)                # 損失関数から逆伝播
gx = func.backward(gy)                  # gy=dldyをtanhに逆伝播
gz = func.backward((y - t)/batch_size)  # tanhから逆伝播する場合

# core関数で順伝播と逆伝播
x2 = HDArray(x)
t2 = HDArray(t)
y2 = func(x2)                           # tanh関数　
l2 = 0.5 * sum((y2 - t2) ** 2) / batch_size # 二乗和誤差　
l2.backtrace()

# グラフで結果比較
cf.xy_graph(x.tolist(), t.tolist())
cf.xy_graph(x.tolist(), y.tolist())
cf.xy_graph(x2.tolist(), y2.tolist())

cf.xy_graph(x.tolist(), gy.tolist())
cf.xy_graph(x2.tolist(), y2.grad.tolist())

cf.xy_graph(x.tolist(), gx.tolist())
cf.xy_graph(x2.tolist(), x2.grad.tolist())
cf.xy_graph(x.tolist(), gz.tolist())
