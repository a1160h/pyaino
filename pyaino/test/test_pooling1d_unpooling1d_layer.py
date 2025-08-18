from pyaino.Config import *
from pyaino.Neuron import Pooling1d, UnPooling1d
from pyaino.Neuron import Pooling1dLayer, UnPooling1dLayer
from pyaino.Neuron import Dropout
import copy

# test_pooling_unpooling

import matplotlib.pyplot as plt

C, Iw, pool, pad = 1, 8, 2, 0

Ow = (Iw+2*pad)//pool if (Iw+2*pad)%pool==0 else (Iw+2*pad)//pool + 1 

params  = C, Iw, pool, pad #, Ow, 'average'
params2 = C, Ow, pool, pad #, Iw, 'average'
       
pooling_layer = Pooling1dLayer(pool, pad, 'average')
unpooling_layer = UnPooling1dLayer(pool, pad, 'average')

x = np.arange(0, C*Iw).reshape(1, C, Iw)
print('x =', x)
s = int(x.size)
plt.imshow(x.reshape(1,s).tolist())
plt.show()

y = pooling_layer.forward(x)
print('y =', y)
s = int(y.size)
plt.imshow(y.reshape(1,s).tolist())
plt.title('y = pooling(x)')
plt.show()

z = unpooling_layer.forward(y)
print('z =', z)
s = int(z.size)
plt.imshow(z.reshape(1,s).tolist())
plt.title('z = unpooling(y)')
plt.show()

gy = unpooling_layer.backward(z)
print('gy =', gy)
s = int(gy.size)
plt.imshow(gy.reshape(1,s).tolist())
plt.title('gy = unpooling**-1(gz)')
plt.show()

gx = pooling_layer.backward(gy)
print('gx =', gx)
s = int(gx.size)
plt.imshow(gx.reshape(1,s).tolist())
plt.title('gx = pooling**-1(gy)')
plt.show()

