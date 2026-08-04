from pyaino.Config import *
from pyaino import Functions as F


set_derivative(True)

x = np.array([[1, 4, 2, 3],[5, 0, 7, 6]], dtype=Config.dtype)
x = np.hdarray(x)

model = F.TopK(2, axis=-1) 

values, indices = model(x)

print("values:\n", values)
print("indices:\n", indices)

values.backtrace()

print("x.grad:\n", x.grad)    

gy = np.array([[1, 2],[3, 4]])
gx = model.backward(gy)

print("x.grad:\n", gx)    
