from pyaino.Config import *
#set_np('numpy'); np=Config.np
from pyaino import Neuron

m, n = 3, 3

model = Neuron.ContextualSelfAttention(m, n, debug_mode=True)

x = np.arange(2*3).reshape(1, 2, 3)
print('入力データx\n', x)

y = model.forward(x) 

print('\nmodelのa\n', model.attention.a)
print('\nmodelの結果y\n',y)

gx = model.backward()

print('\ngx\n', gx)
print('\ngrad_q\n', model.grad_q)
print('\ngrad_w\n', model.grad_w)

model.update(eta=1)
