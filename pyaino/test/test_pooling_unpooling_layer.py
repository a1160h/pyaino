# test_pooling_unpooling
# 20201204 A.Inoue
from pyaino.Config import *
set_np('numpy'); np=Config.np
import matplotlib.pyplot as plt
from pyaino import Neuron as neuron

C, Ih, Iw, pool, pad = 1, 18, 18, 2, 1

Oh = (Ih+2*pad)//pool if (Ih+2*pad)%pool==0 else (Ih+2*pad)//pool + 1
Ow = (Iw+2*pad)//pool if (Iw+2*pad)%pool==0 else (Iw+2*pad)//pool + 1 

params  = C, Ih, Iw, pool, pad #, Oh, Ow, 'average'
params2 = C, Oh, Ow, pool, pad #, Ih, Iw, 'average'
       
pooling_layer = neuron.PoolingLayer(pool, pad, 'average')
unpooling_layer = neuron.UnPoolingLayer(pool, pad, 'average')

x = np.arange(0, C*Ih*Iw).reshape(1, C, Ih, Iw)
print('x =', x)
s = int(x.size**0.5)
plt.imshow(x.reshape(s,s).tolist())
plt.show()

y = pooling_layer.forward(x)
print('y =', y)
s = int(y.size**0.5)
plt.imshow(y.reshape(s,s).tolist())
plt.title('y = pooling(x)')
plt.show()

z = unpooling_layer.forward(y)
print('z =', z)
s = int(z.size**0.5)
plt.imshow(z.reshape(s,s).tolist())
plt.title('z = unpooling(y)')
plt.show()

gy = unpooling_layer.backward(z)
print('gy =', gy)
s = int(gy.size**0.5)
plt.imshow(gy.reshape(s,s).tolist())
plt.title('gy = unpooling**-1(gz)')
plt.show()

gx = pooling_layer.backward(gy)
print('gx =', gx)
s = int(gx.size**0.5)
plt.imshow(gx.reshape(s,s).tolist())
plt.title('gx = pooling**-1(gy)')
plt.show()

