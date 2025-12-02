from pyaino.Config import *
from pyaino import Neuron
from pyaino.nucleus import Function, CompositFunction
from pyaino import Functions as F
import matplotlib.pyplot as plt

mu = (np.arange(12)*-0.1).reshape(3,4)
log_var = (np.arange(12)*0.2).reshape(3,4)
    
class KullbackLeiblerDivergence(CompositFunction):
    def _forward(self, mu, log_var):
        kll = -0.5 * F.sum(1 + log_var - mu**2 - F.exp(log_var))
        return kll / len(mu)

model1 = Neuron.KullbackLeiblerDivergenceNormal()
model2 = KullbackLeiblerDivergence()

kll1 = model1.forward(mu, log_var)
kll2 = model2.forward(mu, log_var)
print(kll1)
print(kll2)

gmu1, glogvar1 = model1.backward()
gmu2, glogvar2 = model2.backward()

plt.plot(mu.reshape(-1).tolist(), label='mu')
plt.plot(log_var.reshape(-1).tolist(), label='log_var')
plt.plot(gmu1.reshape(-1).tolist(), label='gmu1')
plt.plot(glogvar1.reshape(-1).tolist(), label='glogvar1')
plt.plot(gmu2.reshape(-1).tolist(), label='gmu2')
plt.plot(glogvar2.reshape(-1).tolist(), label='glogvar2')
plt.legend()
plt.show()

print('gmu1')
print(gmu1)
print('gmu2')
print(gmu2)
print('glogvar1')
print(glogvar1)
print('glogvar2')
print(glogvar2)
