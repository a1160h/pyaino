# test_Var_veriously
# 20250602 井上

from pyaino.Config import *
set_np('numpy'); np=Config.np  
from pyaino import Functions as F


functions = F.Std, 

x = np.array([1, 2, 3, 4, 5])
print(x)

for func in functions:
    print('##### test', func.__name__, '#####')
    model = func()
    y = model.forward(x)
    print("Forward y =", y)

    dy = 1 # 逆伝播の入力として、通常は損失関数の勾配が与えられる
    dx = model.backward(dy)
    print("Backward gx =", dx)

