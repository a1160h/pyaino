from pyaino.Config import *
from pyaino import Functions as F
import matplotlib.pyplot as plt

functions = (F.Abs, F.Sin, F.Cos, F.Square, F.Sqrt)

x = np.linspace(-4, 4)

for f in functions:
    func = f()
    print('test ', func.__class__.__name__)
    y = func(x)
    gx = func.backward()
    
    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.show()

bases = (None, 0.2, 0.5, 1, 2, 3, 4)
x = np.linspace(-2, 2)

for a in bases:
    func = F.Exp(a)
    print('test ', func.__class__.__name__, 'base', a)
    
    y = func(x) 
    gx = func.backward()
    
    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.grid()
    plt.show()


bases = (None, 2, 4, 8)
x = np.linspace(0, 10)

for a in bases:
    func = F.Log(a)
    print('test ', func.__class__.__name__, 'base', a)
    
    # y = func(x)
    y = func(np.where(x==0,1e-20,x)) # Logは0を真数に取れない 20250325N
    gx = func.backward()
    
    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.show()

const = (1, 2, 3, 4, 5, 0.5, -0.5, -1, -2)
x = np.linspace(-2, 2)

for c in const:
    func = F.Pow(c)
    print('test ', func.__class__.__name__, 'exponent', c)
    y = func(x)
    gx = func.backward()
    
    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.show()

