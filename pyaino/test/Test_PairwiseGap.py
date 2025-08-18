# PaiwiseGapのnumerical gradientをリファレンスにした検証

from pyaino.Config import *
from pyaino.LossFunctions import PairwiseGap
from pyaino import common_function as cf

x = np.array([0.0, 0.5, 1.4, 3.0, 6.2, 3.8, 0.2, -1.0]).reshape(2, 4)
print('x\n', x)
model = PairwiseGap(gap=1, beta=1)
loss = model.forward(x)
print('loss =', loss)
print('diffs\n', model.diffs)
gx = model.backward()
print('gx\n', gx)
gx = cf.numerical_gradient(model.forward, x)
print('gx\n', gx)
