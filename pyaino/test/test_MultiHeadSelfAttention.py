from pyaino.Config import *
from pyaino import Neuron

B = 2
T = 3
H = 4
n = 2

x = np.arange(B*T*H).reshape(B,T,H)

print('x.shape = (B, T, H) =', x.shape)
print('x\n', x)

model = Neuron.MultiHeadSelfAttention(n_head=n, debug_mode=True)
y = model.forward(x)

print('y.shape = (B, T, H) =', y.shape)
print('y\n', y)

dy = np.ones_like(y)
print('dy.shape = (B, T, H) =', dy.shape)
print('dy\n', dy)

dx = model.backward(dy)

print('dx.shape = (B, T, H) =', dx.shape)
print('dx\n', dx)
