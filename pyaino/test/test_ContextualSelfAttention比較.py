from pyaino.Config import *
#set_np('numpy'); np=Config.np
from pyaino import Neuron

m, n = 2, 2

model1 = Neuron.ContextualSelfAttention(m, n, debug_mode=True)
model2 = Neuron.ContextualSelfAttentionZ1(m, n, debug_mode=True)
model3 = Neuron.ContextualSelfAttentionZ2(m, n, debug_mode=True)
model4 = Neuron.ContextualSelfAttentionZ4(m, n, debug_mode=True)

x = np.arange(2*2).reshape(1, 2, 2)
print('入力データx\n', x)

y1 = model1.forward(x) 
y2 = model2.forward(x)
y3 = model3.forward(x)
y4 = model4.forward(x)

print('\nmodel1のq\n', model1.q)
print('\nmodel2のq\n', model2.q)
print('\nmodel3のq\n', model3.q)
print('\nmodel4のq\n', model4.q)

print('\nmodel1のa\n', model1.attention.a)
print('\nmodel2のa\n', model2.a)
print('\nmodel3のa\n', model3.a)
print('\nmodel4のa\n', model4.a)

print('\nmodel1の結果y1\n',y1)
print('\nmodel2の結果y2\n',y2)
print('\nmodel3の結果y3\n',y3)
print('\nmodel4の結果y4\n',y4)
