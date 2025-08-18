from pyaino.Config import *
from pyaino import Functions as F

a = np.array([[8,2,9],[4,7,4]])
print(a)

for axis, keepdims in (
                       (None, False),
                       (None, True),
                       (0, False),
                       (0, True),
                       (1, False),
                       (1, True)
                       ):
    print('axis =', axis, 'keepdims =', keepdims)

    max = F.Max(axis, keepdims)
    max_a = max.forward(a)#False)
    print(max_a)

    g_max_a = np.arange(1, max_a.size+1).reshape(max_a.shape)
    b = max.backward(g_max_a)
    print(b)
