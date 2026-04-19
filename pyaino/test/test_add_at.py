from pyaino.Config import *
from pyaino import safe_np as snp


x = np.zeros(5)
idx = np.array([0, 0, 1])
val = np.array([1, 2, 3])

snp.add_at(x, idx, val)
print(x)

