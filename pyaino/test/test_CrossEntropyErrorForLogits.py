"""
CrossEntropyError(Softmax(logits))と
CrossEntropyErrorForLogits(logits)との比較検証
前者はlogitsを確率に変換してから損失関数に入れるが、
後者はlogitsから直接損失を求める

"""

from pyaino.Config import *
from pyaino import Activators, LossFunctions
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 100) # 値の範囲を指定
z = z.reshape(2, -1)        # 2バッチ、50個ずつ
t = np.array([20, 30])

func = Activators.Softmax()  
y = func(z)
c = np.eye(y.shape[-1])[t]

loss_function1 = LossFunctions.CrossEntropyError()
loss_function2 = LossFunctions.CrossEntropyErrorForLogits()

loss1 = loss_function1(y, c)
loss2 = loss_function2(z, t)
print(f'loss1 ={loss1} loss2={loss2}')

gy1 = loss_function1.backward()
gz1 = func.backward(gy1)

gz2 = loss_function2.backward() # こちらは直接

plt.plot(z.T.tolist(), y.T.tolist())
plt.plot(z.T.tolist(), gz1.T.tolist())
plt.title(func.__class__.__name__+'+'+loss_function1.__class__.__name__)
plt.show()

plt.plot(z.T.tolist(), y.T.tolist())
plt.plot(z.T.tolist(), gz2.T.tolist())
plt.title(loss_function2.__class__.__name__)
plt.show()

