from pyaino.Config import *
from pyaino import Functions as F


a = np.arange(4)
print('source\n', a)

for shape in (
              ((4, 1),(4, 3)),
              ((1, 4),(3, 4)) 
              ):
    
    model = F.BroadcastTo(shape[1])
    print('##### test', model.__class__.__name__, '#####')
    print('source and target shape =', shape)
    a = a.reshape(shape[0])
    print('source\n', a)
    b = model.forward(a)
    print(a.shape, '--forward-->', b.shape)
    print('forward result\n', b)

    gb = np.arange(1, b.size+1).reshape(b.shape)
    c = model.backward(gb)
    print(b.shape, '--backward-->', c.shape)
    print('backward result\n', c)
