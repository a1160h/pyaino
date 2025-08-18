# inplace演算の確認
from pyaino.Config import *
from pyaino import Neuron

#x = np.array([np.arange(100000) for _ in range(10000)])
x = np.arange(500000000, dtype='f8').reshape(-1, 10000)
# x = np.arange(50000, dtype='f8').reshape(-1, 100) #  テスト用 
print(x.shape)
 
func = Neuron.Normalization(axis=-1, mask_enable=False, inplace=True)
y = func(x)
print('x :', id(x), 'y :', id(y))  # xと別
func.backward()

input('wait')

func = Neuron.Normalization(axis=-1, mask_enable=False, inplace=False)
y = func(x)
print('x :', id(x), 'y :', id(y))  # xと同一
func.backward()

