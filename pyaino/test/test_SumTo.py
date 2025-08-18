from pyaino.Config import *
from pyaino import Functions as F
from pyaino import HDFunctions as HDF

a = np.arange(20).reshape(5, 4)
print('source\n', a)
for shape in (
              (),
              (5, 1),(5,),
              (1, 4),(4,),
              (1, 1),(1,) 
                        ):
    for model in F.SumTo(shape), HDF.SumTo(shape):
        print('##### test', model.__class__.__name__, '#####')
        print('target shape =', shape)

        b = model.forward(a)
        print(a.shape, '--forward-->', b.shape)
        print('forward result\n', b)

        gb = np.arange(1, b.size+1).reshape(b.shape)
        c = model.backward(gb)
        print(b.shape, '--backward-->', c.shape)
        print('backward result\n', c)

print('-'*30)
a = np.arange(24).reshape(2, 3, 4)
print('source\n', a)
for shape in (
              (),
              (2, 3, 1), (2, 3,),
              (1, 1, 4), (4,),
              (2, 1, 4), (2, 4),
              (1, 1, 1), (1, 1), (1,) 
                             ):
    for model in F.SumTo(shape), HDF.SumTo(shape):
        print('##### test', model.__class__.__name__, '#####')
        print('target shape =', shape)

        b = model.forward(a)
        print(a.shape, '--forward-->', b.shape)
        print('forward result\n', b)

        gb = np.arange(1, b.size+1).reshape(b.shape)
        c = model.backward(gb)
        print(b.shape, '--backward-->', c.shape)
        print('backward result\n', c)

