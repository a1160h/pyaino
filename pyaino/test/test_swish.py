from pyaino.Config import *
set_np('numpy'); np = Config.np
from pyaino import common_function as cf
from pyaino.nucleus import HDArray # 追加 20250327N
from pyaino import Activators # 追加 20250327N
from pyaino import HDFunctions # 追加 20250327N

set_derivative(True) # 追加 20250327N

beta = 0.5
x = HDArray(np.linspace(-5, 5, 100))
# forward の式
s = 1 / (1 + HDFunctions.exp(-beta*x))
y = x * s
# backward の式
#z = s + x * s * (1 - s)
#z = (beta * x * s + s * (1 - beta * x * s))
z = (1 + beta * x - beta * x * s) * s
# 1 / (1 + exp(-x)) + x * (1 /(1 + exp(-x))) * (1 - 1 / (1 + exp(-x)))
# coreを使った逆伝播
y.backtrace(create_graph=True)

cf.xy_graph(x.tolist(), y.tolist())
cf.xy_graph(x.tolist(), z.tolist())
cf.xy_graph(x.tolist(), x.grad.tolist())

