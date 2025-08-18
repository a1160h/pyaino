from pyaino.Config import *
from pyaino import LossFunctions as lf
import matplotlib.pyplot as plt

x = np.array([0.0, 0.5, 1.4, 3.0, 6.2, 3.8, 0.2, -1.0]).reshape(2, 4)
print('x\n', x)
model = lf.MeanStdDeviation(beta1=1, beta2=1)
loss = model.forward(x)
print('loss =', loss)
gx = model.backward()
print('gx\n', gx)
gx*=10
plt.plot(x.T.tolist())
plt.plot(gx.T.tolist())
plt.show()

