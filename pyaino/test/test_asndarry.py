from pyaino.Config import *
#set_np('numpy'); np=Config.np
from pyaino.nucleus import asndarray, HDArray

import numpy as np
import cupy  as cp

print(np.__name__)
print(cp.__name__)

x = [0, 1, 2, 3]
a = np.array(x)
b = cp.array(x)
ah = HDArray(a)
bh = HDArray(b)

print('a', type(a), a)
print('b', type(b), b)
print('ah', type(ah), ah)
print('bh', type(bh), bh)

print(isinstance(a, np.ndarray)) # True
print(isinstance(a, cp.ndarray)) # False
print(isinstance(b, np.ndarray)) # False
print(isinstance(b, cp.ndarray)) # True

xx = asndarray(x)
print('xx', xx)
ax = asndarray(a) # OK
print('ax', ax)
bx = asndarray(b) # OK
print('bx', bx)
ahx = asndarray(ah)
print('ahx', ahx)
bhx = asndarray(bh)
print('bhx', bhx)
