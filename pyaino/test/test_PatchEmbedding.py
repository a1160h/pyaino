from pyaino.Config import *
set_derivative(True)
from pyaino import Neuron as nn

x = np.arange(1*3*4*4, dtype=Config.dtype).reshape(1,3,4,4)
x = np.hdarray(x)

P = 2
model1 = nn.PatchEmbedding(12, P, debug_mode=True)
model2 = nn.PatchEmbeddingSimple(12, P, debug_mode=True)

y1 = model1(x)
y2 = model2(x)
print(x)
print(y1)     # (B, Oh*Ow, C*P*P)
print(y2)
print('y all close ->', np.allclose(y1, y2))

"""
gy = np.ones_like(y1)
gx1 = model1.backward(gy)
gx2 = model2.backward(gy)
"""
y1.backtrace()
gx1 = x.grad

y2.backtrace()
gx2 = x.grad

print(gx1)
print(gx2)
print('gx all close ->', np.allclose(gx1, gx2))


