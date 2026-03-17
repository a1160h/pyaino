from pyaino.Config import *
from pyaino import Neuron

x = np.arange(1*3*4*4, dtype=Config.dtype).reshape(1,3,4,4)

P = 2

model = Neuron.PatchEmbedding(12, P, debug_mode=True)

y = model(x)
print(x)
print(y)     # (B, Oh*Ow, C*P*P)


