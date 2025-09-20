from pyaino.Config import *
from pyaino import Activators, LossFunctions
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 100) # 値の範囲を指定
z = z.reshape(2, -1)        # 2バッチ、50個ずつに分ける
t = np.array([20, 30])      # 指値(何でも良い)

func = Activators.SoftmaxCrossEntropy()
y, l = func.forward(z, t)
gz = func.backward()

ref = Activators.Softmax()
loss = LossFunctions.CrossEntropyError() 
yr = ref(z)
c = np.eye(y.shape[-1])[t]
lr = loss(yr, c)
gyr = loss.backward()
gzr = ref.backward(gyr)

print('y == yr :', np.allclose(y, yr))
print(f'loss ={l} loss_ref={lr}')

plt.plot(z.T.tolist(), y.T.tolist())
plt.plot(z.T.tolist(), gz.T.tolist())
plt.title(func.__class__.__name__)
plt.show()

plt.plot(z.T.tolist(), yr.T.tolist())
plt.plot(z.T.tolist(), gzr.T.tolist())
plt.title('reference:'+ref.__class__.__name__+'+'+loss.__class__.__name__)
plt.show()


