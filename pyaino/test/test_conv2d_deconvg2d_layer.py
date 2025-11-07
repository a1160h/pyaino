# test_conv_deconv
# 20240610 A.Inoue
from pyaino.Config import *
set_np('numpy'); np = Config.np
import matplotlib.pyplot as plt
from pyaino import Neuron as neuron
from pyaino import LossFunctions

C, image_size, M, kernel_size, stride, pad = 1, 16, 1, 2, (1, 1), 1

conv_layer = neuron.Conv2dLayer(M, kernel_size, stride, pad)
deconv_layer = neuron.DeConv2dLayer(C, kernel_size, stride, pad)
loss_function = LossFunctions.MeanSquaredError()

x = np.arange(0, C*image_size**2).reshape(1, C, image_size, image_size)
print('x =', x)
s = int(x.size**0.5)
plt.imshow(x.reshape(s,s).tolist())
plt.title('x')
plt.show()

y = conv_layer.forward(x)
conv_layer.parameters.w[...] = 0.25
y = conv_layer.forward(x)
print('y =', y)
s = int(y.size**0.5)
plt.imshow(y.reshape(s,s).tolist())
plt.title('y = conv(x)')
plt.show()

z = deconv_layer.forward(y)
deconv_layer.parameters.w[...] = 0.5
z = deconv_layer.forward(y)
print('z =', z)
s = int(z.size**0.5)
plt.imshow(z.reshape(s,s).tolist())
plt.title('z = deconv(y)')
plt.show()

l = loss_function.forward(x, z)
gl = loss_function.backward()

gy = deconv_layer.backward(gl)
print('gy =', gy)
s = int(gy.size**0.5)
plt.imshow(gy.reshape(s,s).tolist())
plt.title('gy = deconv**-1(z)')
plt.show()

gx = conv_layer.backward(gy)
print('gx =', gx)
s = int(gx.size**0.5)
plt.imshow(gx.reshape(s,s).tolist())
plt.title('gx = conv**-1(gy)')
plt.show()

conv_layer = neuron.Conv2dLayer(M, kernel_size, stride, pad, optimize='Adam')
deconv_layer = neuron.DeConv2dLayer(C, kernel_size, stride, pad, optimize='Adam')

error_record = []
for i in range(1000):
    y = conv_layer.forward(x)
    z = deconv_layer.forward(y)
    l = loss_function.forward(z, x)
    print(i, l)
    error_record.append(float(l))
    gl = loss_function.backward()
    gy = deconv_layer.backward(gl)
    gx = conv_layer.backward(gy)
    conv_layer.update(eta=0.01)
    deconv_layer.update(eta=0.01)

print('loss =', l)
plt.plot(error_record)
plt.show()

s = int(z.size**0.5)
plt.imshow(z.reshape(s,s).tolist())
plt.title('z = deconv(y)')
plt.show()
   
    
