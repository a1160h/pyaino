# inplace演算の確認
from pyaino.Config import *
from pyaino import Neuron

x  = np.arange(6, dtype='f4').reshape(2, 3)
gy = np.arange(6, dtype='f4').reshape(2, 3)
 
func = Neuron.GeneralNormalizationBase(axis=-1, scale_and_bias=True,
                                       mask_enable=False, inplace=False)
y = func(x)
gx = func.backward(gy)
print('x\n', x, '\ny\n', y, '\ngy\n', gy, '\ngx\n', gx)  # xと別

input('wait')

func = Neuron.GeneralNormalizationBase(axis=-1, scale_and_bias=True,
                                       mask_enable=False, inplace=True)
y = func(x)
gx = func.backward(gy)
print('x\n', x, '\ny\n', y, '\ngy\n', gy, '\ngx\n', gx)  # xと同一

