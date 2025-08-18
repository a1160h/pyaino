# AttentionUnitの動作確認

from pyaino.Config import *
from pyaino import Neuron

B  = 2
Tk = 3
Tq = 4
h  = 1
H  = 3

# SingleHeadAttentionのテスト
print('SingleHeadAttentionのテスト')
x = np.arange(B*Tk*H).reshape(B, Tk, H).astype(float) 
q = np.arange(B*Tq*H).reshape(B, Tq, H).astype(float)
print('x(v, k) : (B, Tk, H) =', x.shape, '　q :(B, Tq, H) =', q.shape)

model0 = Neuron.AttentionUnit() # SingleHeadなクラス(標準でhead=1) 20250325N

y0 = model0.forward(x, x, q)

print('y0.shape = (B, Tq, H) =', y0.shape)
print('y0\n', y0)

dy0 = np.ones_like(y0)
print('dy0.shape = (B, Tq, H) =', dy0.shape)
print('dy0\n', dy0)

dx0, dk0, dq0 = model0.backward(dy0)

print('dx0.shape = (B, Tk, H) =', dx0.shape)
print('dq0.shape = (B, Tq, H) =', dq0.shape)
print('dx0\n', dx0)
print('dq0\n', dq0)


x = np.arange(B*Tk*h*H).reshape(B, Tk, h*H).astype(float) 
q = np.arange(B*Tq*h*H).reshape(B, Tq, h*H).astype(float)
mask = np.array([[0, 1, 0, 1],[1, 0, 1, 0]])

print('x.shape = (B, Tk, H) =', x.shape, '　q.shape = (B, Tq, H) =', q.shape)
#print('x\n', x, '\nq\n', q)

model1 = Neuron.AttentionUnit(head=3) # MultiHeadなクラス 20250325N

y1 = model1.forward(x, x, q)#, scaling=True, mask=mask)
#model.w[...] = 1

print('y1.shape = (B, Tq, H) =', y1.shape)
print('y1\n', y1)

dy1 = np.ones_like(y1)
print('dy1.shape = (B, Tq, H) =', dy1.shape)
print('dy1\n', dy1)

dx1, dk1, dq1 = model1.backward(dy1)

print('dx1.shape = (B, Tk, H) =', dx1.shape)
print('dq1.shape = (B, Tq, H) =', dq1.shape)
print('dx1\n', dx1)
print('dq1\n', dq1)
