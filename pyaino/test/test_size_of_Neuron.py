# inplace演算の確認
from pyaino.Config import *
from pyaino import Neuron
from pyaino import common_function as cf

x = np.arange(1000000, dtype='f4').reshape(-1, 1000)
print(x.shape)

Funcs = (
Neuron.LinearLayer(100),    
Neuron.NeuronLayer(100),
 )

for func in Funcs:
    print('\ntest', func.__class__.__name__, func.config)
    y = func(x)
    print('x :', id(x), 'y :', id(y))  # xと別
    func.backward()

    print(f'func : {cf.get_obj_size(func, verbose=True)/1024**2 :8.2f} MB') 
    print(f'x    : {cf.get_obj_size(x)   /1024**2 :8.2f} MB') 
    print(f'y    : {cf.get_obj_size(y)   /1024**2 :8.2f} MB') 
    print(f'w    : {cf.get_obj_size(func.parameters.w)/1024**2 :8.2f} MB') 
    print(f'b    : {cf.get_obj_size(func.parameters.b)/1024**2 :8.2f} MB') 
    print(f'total: {cf.get_obj_size([func, x, y])/1024**2 :8.2f} MB')
