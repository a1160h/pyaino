from pyaino.Config import *
import matplotlib.pyplot as plt
from pyaino import Optimizers as opt


funcs = (opt.Annealing(),
         opt.ExponentialDecay(decay_rate=0.1, time_scale=100),
         opt.CosineDecay(decay_rate=0.1, decay_start=100, decay_end=500),
         opt.LinearGrowCosineDecay(warmup=100, decay_rate=0.1, decay_start=100, decay_end=500),
         opt.SineGrowCosineDecay(initial=0.2, grow_start=100, grow_end=200, decay_rate=0.1, decay_start=300, decay_end=500)
         )


eta = 0.01
iters = range(0, 1000)

for f in funcs:
    etas = []    
    for t in iters:
        decay = f(t)
        decay = float(decay)
        etas.append(decay)

    plt.plot(iters, etas)
    plt.show()


