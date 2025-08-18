# inplace演算の確認
from pyaino.Config import *
from pyaino import Functions
import time

#x = np.array([np.arange(100000) for _ in range(10000)])
x = np.arange(200000000, dtype='f8').reshape(-1, 10000)
print(x.shape)
w = np.ones_like(x).T
b = np.zeros(w.shape[-1], dtype='f8')


start = time.time() 
func = Functions.DotLinear()
y = func(x, w, b)
elapse = time.time() - start
print('x :', id(x), 'y :', id(y))
print(elapse)

start = time.time() 
func.backward()
elapse = time.time() - start
print(elapse)
