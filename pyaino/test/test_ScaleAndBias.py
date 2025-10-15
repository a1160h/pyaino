from pyaino.Config import *
from pyaino import Neuron
from pyaino import LossFunctions as lf


x = np.arange(2*3*4).reshape(4, 2, 3)

model = Neuron.ScaleAndBias(axis=(-2, -1), optimize='Adam')
loss  = lf.MeanSquaredError()

y = model.forward(x)
print('入力\n', x)
print('出力\n', y)

print('敢えて、gammaとbetaを乱数で初期化')
model.gamma = np.random.randn(*model.gamma.shape)
model.beta  = np.random.randn(*model.beta.shape)
print('gamma\n', model.gamma)
print('beta\n', model.beta)

print('学習により、modelが元の値を出力するように、gammaとbetaが調整される')
for i in range(20000):
    y = model.forward(x)
    l = loss.forward(y, x)
    gy = loss.backward()
    gx = model.backward(gy)
    model.update()
    #print(i, l)

print('学習後のgammaとbeta')
print('gamma\n', model.gamma)
print('beta\n', model.beta)

y = model.forward(x)
print('学習後の出力')
print(y)
    
assert np.allclose(x, y, rtol=1e-3, atol=1e-3)
