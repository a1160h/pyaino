from pyaino import *
from pyaino.Config import *
from pyaino.nucleus import HDArray # 追加 20250327N
from pyaino import Activators # 追加 20250327N
from pyaino import common_function as cf

set_derivative(True) # 追加 20250327N

beta = 0.5
x = HDArray(np.linspace(-5, 5, 100))
# forward の式
y = Activators.Mish()(x)
# backward の式
# coreを使った逆伝播
y.backtrace()

cf.xy_graph(x.tolist(), y.tolist())
cf.xy_graph(x.tolist(), x.grad.tolist())

