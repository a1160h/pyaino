# inplace演算の確認
from pyaino.Config import *
set_np('numpy'); np = Config.np
from pyaino import Activators
import tracemalloc
import gc

#x = np.array([np.arange(100000) for _ in range(10000)])
x = np.random.randn(20000, 10000).astype('f8')
print(x.shape)

input('wait')

Funcs = ['Identity', 'Step', 'Sigmoid', 'Tanh', 'ReLU', 'LReLU', 'ELU', 'Softmax', 'Swish', 'Softplus', 'Mish']

for F in Funcs:
    func = eval('Activators.'+F)()
    print('\n', func.__class__.__name__)

    tracemalloc.start()

    y = func(x)
    try: 
        func.backward()
    except:
        print('  backward failed', func.__class__.__name__)
        pass

    current, peak = tracemalloc.get_traced_memory()
    print(f"  Current memory usage: {current / 1024 / 1024:8.2f} MB")
    print(f"  Peak memory usage   : {peak / 1024 / 1024:8.2f} MB")
    tracemalloc.stop()
    
    y = None
    func = None
    gc.collect()



