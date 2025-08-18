from pyaino.Config import *
from pyaino import Neuron

Dropouts = Neuron.Dropout, Neuron.Dropout2


Axis = (-1,2,3,4), (-1,3,4), (-1, 4), (-1,)
Rates = 0.0, 0.5, 1.0
Inplaces = True, False

for f in Dropouts:
    for a in Axis:
        for r in Rates:
            for i in Inplaces:
                x = np.arange(1*2*3*4, dtype=Config.dtype)
                func = f(inplace=i)
                print('\naxis =', a, 'rate = ', r, 'inplace =', i)
                y = func(x.reshape(*a), dropout=r)
                print(x)
                print(y)
                gy = np.ones_like(y)
                gx = func.backward(gy)
                print(gx)

    
