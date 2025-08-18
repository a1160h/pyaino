from pyaino.Config import *
from pyaino import Functions as F

#x = np.arange(36).reshape(2,3,2,3)
x = np.arange(6).reshape(2,3)
print('x =', x.shape)

model = F.Pairwise(axis=-1, broadcast=False, diagonal_mask=True)

p, q = model.forward(x)
print('p =', p.shape)
print('q =', q.shape)

gp = np.ones_like(p)
gq = np.ones_like(q)

gx = model.backward()#gp, gq)
print('gx =', gx.shape)
