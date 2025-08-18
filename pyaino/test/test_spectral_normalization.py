import matplotlib.pyplot as plt
from pyaino.Config import * 
from pyaino import Optimizers
import copy

spectral_normalization = Optimizers.SpectralNormalization()

W = np.random.randn(3, 3)
W_sn = W.copy()
spectral_normalization(W_sn)

print(W)
print(W_sn)

plt.plot(W.reshape(-1).tolist(), marker='x')
plt.plot(W_sn.reshape(-1).tolist(), marker='o')
plt.show()
