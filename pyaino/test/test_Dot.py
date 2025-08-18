from pyaino.Config import *
set_np('numpy'); np=Config.np
from pyaino import HDFunctions as F
from pyaino.nucleus import HDArray
#set_higher_derivative()

set_derivative(True) # 追加 20250327N


dot = F.Dot()
dot_linear = F.DotLinear()

xs = (
      np.arange(4),
      np.arange(4),
      np.arange(6).reshape(3, 2),
      np.arange(2),
      np.arange(6).reshape(2, 3),
      ) 
ws = (
      np.arange(4),
      np.arange(4),
      np.arange(2),
      np.arange(6).reshape(2, 3),
      np.arange(6).reshape(3, 2),
      )

bs = (
      np.array(1),
      np.arange(1),
      np.array(1),
      np.arange(3),
      np.arange(2),
      )

print('\nDot')
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
    print(y.shape , gy.shape)
    gx, gw = dot.backward(gy)
    print('backward:', gy.shape, '->', gx.shape, gw.shape)
    print(gy)
    print(gx)
    print(gw)

print('\nDotLinear')
for x, w, b in zip(xs, ws, bs):
    print()
    #x = HDArray(x)
    #w = HDArray(w)
    #b = HDArray(b)
    y = dot_linear(x, w, b)
    print('forward :', x.shape, '*', w.shape, '+', b.shape, '->', y.shape)
    print(x)
    print(w)
    print(b)
    print(y)

    gy = np.ones_like(y)
    gx, gw, gb = dot_linear.backward(gy)
    print('backward:', gy.shape, '->', gx.shape, gw.shape, gb.shape)
    print(gy)
    print(gx)
    print(gw)
    print(gb)
