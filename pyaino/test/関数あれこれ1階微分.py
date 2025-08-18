from pyaino.Config import *
import matplotlib.pyplot as plt
set_derivative(True)
#Config.enable_debug_print=True

x = np.hdarray(np.linspace(-2, 2))

f1 = lambda x : x + 1
f2 = lambda x : x**5 + 2*x**4 + 3*x**3 + 4*x**2 + 5*x + 6
f3 = lambda x : 1/(x + 3)
f4 = lambda x : 2 ** x

funcs = f1, f2, f3, f4

for f in funcs:
    y = f(x)
    label = "y"; labels = ["y=f(x)"]; logs = [y]
    print('backtrace')
    y.backtrace()
    if not hasattr(x, 'grad'):
        Exception('no grad for x held')
    label += "'"; labels.append(label); logs.append(x.grad)

    for i, y in enumerate(logs):
        plt.plot(x.tolist(), y.tolist(), label=labels[i])
    plt.legend()#loc='lower right')
    plt.show()


