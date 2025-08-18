# SelfAttentionの動作確認
## 現在このプログラムは不要と判断 20250325N

from pyaino.Config import * # 20250318N
from pyaino import Neuron

B  = 2
Tk = 3
H  = 4
Tq = 2

x = np.arange(B*Tk*H).reshape(B, Tk, H) #*0.1
q = np.arange(B*Tq*H).reshape(B, Tq, H)
#x = np.random.randn(B, T, H)

print('x.shape = (B, Tk, H) =', x.shape, '　q.shape = (B, Tq, H) =', q.shape)
print('x\n', x, '\nq\n', q)

model1 = Neuron.SimpleAttentionLayer() # Attention()は存在しない 20250318N

y1 = model1.forward(x, q)
#model.w[...] = 1

print('y.shape = (B, Tq, H) =', y1.shape)
print('y1\n', y1)

dy = np.ones_like(y1)
print('dy.shape = (B, Tq, H) =', dy.shape)
print('dy\n', dy)

dx1, dq1 = model1.backward(dy)

print('dx.shape = (B, Tk, H) =', dx1.shape)
print('dq.shape = (B, Tq, H) =', dq1.shape)
print('dx1\n', dx1)
print('dq1\n', dq1)
