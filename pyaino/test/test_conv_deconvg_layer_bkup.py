# test_conv_deconv
# 20230622 A.Inoue

## 削除 20250401N

from pyaino.Config import *
set_np('numpy'); np = Config.np
import matplotlib.pyplot as plt
from pyaino import Neuron
from pyaino import LossFunctions

C, Ih, Iw, M, Fh, Fw, stride, pad = 1, 8, 8, 1, 2, 2, 1, 0

# 修正 20250327N
conv_layer = Neuron.ConvLayer(M, (Fh, Fw), stride, pad)
deconv_layer = Neuron.DeConvLayer(C, (Fh, Fw), stride, pad)
loss_function = LossFunctions.MeanSquaredError()

x = np.arange(0, C*Ih*Iw).reshape(1, C, Ih, Iw)
print('x =', x)
s = int(x.size**0.5)
plt.imshow(x.reshape(s,s).tolist())
plt.title('x')
plt.show()

y = conv_layer.forward(x)
conv_layer.w[...] = 0.25
y = conv_layer.forward(x)
print('y =', y)
s = int(y.size**0.5)
plt.imshow(y.reshape(s,s).tolist())
plt.title('y = conv(x)')
plt.show()

z = deconv_layer.forward(y)
deconv_layer.w[...] = 0.5
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

error_record = []
for i in range(100):
    y = conv_layer.forward(x)
    z = deconv_layer.forward(y)
    l = loss_function.forward(z, x)
    print(l)
    error_record.append(float(l))
    gl = loss_function.backward()
    gy = deconv_layer.backward(gl)
    gx = conv_layer.backward(gy)
    conv_layer.update(eta=0.00001)
    deconv_layer.update(eta=0.00001)

plt.plot(error_record)
plt.show()

s = int(z.size**0.5)
plt.imshow(z.reshape(s,s).tolist())
plt.title('z = deconv(y)')
plt.show()
   
    
