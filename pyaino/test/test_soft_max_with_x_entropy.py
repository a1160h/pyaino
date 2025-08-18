from pyaino.Config import *
#set_np('numpy'); np = Config.np
from pyaino import common_function as cf
from pyaino import Activators # 追加 20250327N
from pyaino import LossFunctions # 追加 20250327N

set_derivative(True) # 追加 20250327N

variation  = 5

x = np.linspace(-5, 5, variation**2)
t = np.eye(variation)

x = x.reshape(-1, variation)

# pyainoのクラスでシグモイド関数とクロスエントロピー誤差　
func = Activators.Softmax()
loss = LossFunctions.CrossEntropyError()
func_loss = Activators.SoftmaxWithLoss()

y = func.forward(x)
l = loss.forward(y, t)
z = func_loss.forward(x)

print(x.shape, y.shape, t.shape, l.shape, z.shape)
print(y); print(z); print(l)

#'''#
gy = loss.backward(1) # 損失関数から逆伝播
gx = func.backward(gy)       # gy=dldyをsigmoidに逆伝播
gz = func_loss.backward(t)   # sigmoid_with_lossでtから
#print(z)
print(gx); print(gz) 

# グラフで結果比較
cf.xy_graph(x.tolist(), y.tolist())
cf.xy_graph(x.tolist(), t.tolist())
cf.xy_graph(x.tolist(), gy.tolist())
cf.xy_graph(x.tolist(), gx.tolist())
cf.xy_graph(x.tolist(), gz.tolist())

#'''#
