from pyaino.Config import *
set_derivative(True)
from pyaino import Neuron as nn

# 学習対象
p = nn.Parameter(1,2,3)

# 目標値
target = np.arange(1*2*3).reshape(1,2,3)

print('initial:', p())

for epoch in range(500):
    y = p()
    loss = ((y - target) ** 2).sum() # 二乗和誤差
    loss.backtrace()
    p.update(eta=0.01)

    if epoch % 50 == 0:
        print(f"{epoch:3d} loss={float(loss):6.3f}")
        

print('final:', p.w)
