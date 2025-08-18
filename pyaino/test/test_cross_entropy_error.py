from pyaino.Config import *
set_np('numpy'); np = Config.np
from pyaino.nucleus import HDArray # 追加 20250327N
from pyaino import HDFunctions # 追加 20250327N

set_derivative(True) # 追加 20250327N

y = np.arange(0.1, 1.0, 0.01)
t = np.empty(*y.shape)
t[...] = 0.8

batch_size = len(y)

#print(y)
#print(t)
y = HDArray(y)
t = HDArray(t)
# forward の式
loss = - HDFunctions.sum(t * HDFunctions.log(y + HDArray(1e-7)) + (1 - t) * HDFunctions.log(1 - y + 1e-7))/ batch_size # sumとlogを明示 20250327N
print(loss)
# backward の式
#gy = -(t / y - (1 - t) / (1 - y)) / batch_size
gy = (y - t) / (y * (1 - y) * batch_size + 1e-7)

#print(gy)

from pyaino import common_function as cf

cf.xy_graph(y.tolist(), gy.tolist())

loss.backtrace() # backwardからbacktrace変更 20250327N
#print(y.grad)

cf.xy_graph(y.tolist(), y.grad.tolist())

print(sum((gy - y.grad)**2))
