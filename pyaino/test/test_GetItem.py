from pyaino.Config import *
from pyaino import Functions as F

x = np.arange(20)
s = [0, 4, 10]
print(x)
print(s)

model = F.GetItem(s)

print('##### test', model.__class__.__name__, '#####')

y = model.forward(x)
print(x.shape, '->', y.shape)
print(y)

gx = model.backward()
print(gx.shape)
print(gx)

    
        

