from pyaino.Config import *
#set_np('numpy')
# may fail for cupy v7.1.1 or under with specification of axes
from pyaino import Functions

a = np.arange(12).reshape(3, 4)
print(a)

t = Functions.Transpose()
at = t.forward(a)#, (1, 0))
#at = t(a)
print(at)

b = t.backward(at)
print(b)

a = np.array([[-0.03693582],
       [-0.15924713],
       [-0.01082064],
       [-0.33459964],
       [-0.09040176],
       [ 0.36449564],
       [-0.01715612],
       [ 0.33593047],
       [-0.04929101],
       [-0.17628074],
       [-0.18513241],
       [ 0.1759126 ],
       [ 0.14679073],
       [ 0.26620117],
       [ 0.0908041 ],
       [ 0.07913128],
       [ 0.18366614],
       [-0.00796987],
       [-0.03853935],
       [ 0.07024993],
       [ 0.10415211],
       [ 0.42134604],
       [ 0.12671259],
       [ 0.01098609],
       [-0.01664528],
       [ 0.16216218],
       [ 0.097951  ],
       [ 0.11557411],
       [-0.20752858],
       [ 0.20491251],
       [ 0.1044351 ],
       [-0.18481664]], dtype=np.float32)

at = Functions.transpose(a)
print(a)
print(at)

