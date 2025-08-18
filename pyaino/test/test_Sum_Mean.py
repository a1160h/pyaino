from pyaino.Config import *
from pyaino import Functions as F

functions = (
             F.Sum,
             F.Mean,
             F.Var,
             F.Std,
             F.SquareSum,
             F.SquareMean,
             F.RootSumSquare,
             F.RootMeanSquare,
             F.RMS
             )

a = np.arange(1, 21).reshape(5, 4)
print('input data\n', a)

for func in functions:
    print('##### test', func.__name__, '#'*30)
    for axis, keepdims in ((None, False), (None, True),
                           (0, False), (0, True),
                           (1, False), (1, True)
                           ):
        print('axis =', axis, 'keepdims =', keepdims, end=' ')
        model = func(axis=axis, keepdims=keepdims)

        b = model.forward(a)
        print('axis modify', a.shape, '->', b.shape)
        print('output data\n', b)

        #gb = np.arange(1, b.size+1).reshape(b.shape)
        gb = np.ones_like(b)
        c = model.backward()#gb)
        print('backward result shape =', c.shape)
        print('backward result\n', c)
        print('-'*50)
    
        

