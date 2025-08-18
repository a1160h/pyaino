from pyaino.Config import *
set_np('numpy'); np=Config.np
from pyaino import HDFunctions as F
from pyaino.nucleus import HDArray
#set_higher_derivative()

set_derivative(True) # 追加 20250327N


dot = F.Dot()
dot_linear = F.DotLinear()

xs = (np.array([[0,1],[2,3]]),
      np.array([[0,1],[2,3]]),
      np.array([[0,1],[2,3]]),
      np.array([1,2]),
      np.array([1,2]),
      np.array([1,2]),
      np.array(2),
      np.array(2),
      np.array(2),
      )
ws = (np.array([[0,1],[2,3]]),
      np.array([0,1]),
      np.array(1),
      np.array([[0,1],[2,3]]),
      np.array([0,1]),
      np.array(2),
      np.array([[0,1],[2,3]]),
      np.array([0,1]),
      np.array(2),
      )

for x, w in zip(xs, ws):
    print()
    #x = HDArray(x)
    #w = HDArray(w)
    y = dot(x, w)
    print('forward :', x.shape, '*', w.shape, '->', y.shape)
    print(x)
    print(w)
    print(y)

    gy = np.ones_like(y)
    gx, gw = dot.backward(gy)
    print('backward:', gy.shape, '->', gx.shape, gw.shape)
    print(gy)
    print(gx)
    print(gw)

