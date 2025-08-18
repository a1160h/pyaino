# AttentionUnitの動作確認

from pyaino.Config import *
from pyaino import Neuron

B  = 2
Tk = 3
H  = 4
Tq = 1

x = np.arange(B*Tk*H).reshape(B, Tk, H) 
q = np.arange(B*Tq*H).reshape(B, Tq, H)

print('x.shape = (B, Tk, H) =', x.shape, '　q.shape = (B, Tq, H) =', q.shape)
print('x\n', x, '\nq\n', q)

model = Neuron.AttentionUnit(head=1) 

y = model.forward(x, x, q)
# model.w[...] = 1

print('y.shape = (B, Tq, H) =', y.shape) 
print('y\n', y) 

dy = np.ones_like(y)
print('dy.shape = (B, Tq, H) =', dy.shape)
print('dy\n', dy)

dx, dk, dq = model.backward(dy)

print('dx.shape = (B, Tk, H) =', dx.shape) 
print('dq.shape = (B, Tq, H) =', dq.shape) 
print('dx\n', dx) 
print('dq\n', dq) 
