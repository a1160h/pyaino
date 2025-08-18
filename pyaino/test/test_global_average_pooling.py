# test_global_average_pooling
# 20230910 A.Inoue
from pyaino.Config import *
from pyaino.Neuron import * 

global_average_pooling = GlobalAveragePooling()
x = np.arange(0, 24).reshape(2, 3, 2, 2)
print(x)
y = global_average_pooling.forward(x)
print(y)
grad_y = np.ones_like(y)
print(grad_y)
grad_x = global_average_pooling.backward(grad_y)
print(grad_x)
