from pyaino.Config import *
from pyaino import LossFunctions as lf
from pyaino import Activators as A

logits = np.random.rand(3, 4)
t = np.array([1, 2, 0])

y = A.softmax(logits)

model1 = lf.CrossEntropyError(reduction='sample')
model2 = lf.CrossEntropyErrorMasked()
model3 = lf.CrossEntropyErrorForLogits()

loss1 = model1(y, t)
loss2 = model2(y, t)
loss3 = model3(logits, t)
print('loss1', loss1)
print('loss2', loss2)
print('loss3', loss3)

gy1 = model1.backward()
gy2 = model2.backward()
g_logits = model3.backward()
print('gy1\n', gy1)
print('gy2\n', gy2)
print('g_logits\n', g_logits)

print(model1.l_shape, model1.sample_size, model1._cached_denom)
print(model2.l_shape, model2.sample_size, model2._cached_denom)
print(model3.l_shape, model3.sample_size, model3._cached_denom)
