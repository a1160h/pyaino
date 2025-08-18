# SelfAttentionの動作確認

from pyaino.Config import *
set_np('numpy'); np=Config.np
from pyaino import Neuron

B = 4
T = 3
H = 5

#x = np.arange(B*T*H).reshape(B, T, H) #*0.1
x = np.random.randn(B, T, H)

print('x.shape = (B, T, H) =', x.shape)
print('x\n', x)

model = Neuron.SelfAttention()

y = model.forward(x)
#model.w[...] = 1

y = model.forward(x)

print('y.shape = (B, H) =', y.shape)
print('y\n', y)

dy = np.random.randn(B,T, H)
print('dy.shape = (B, H) =', dy.shape)
print('dy\n', dy)
print(dy.shape)
dx = model.backward(dy)

print('dx.shape = (B, T, H) =', dx.shape)
print('dx\n', dx)
