from pyaino.Config import *
from pyaino import NN_CNN
from pyaino import common_function as cf

model = NN_CNN.NN_1(1)#, debug_mode=True)

x = np.arange(4).reshape(2,2)
y = model.forward(x)
print(x)
print(y)
model.summary()

print('='*50)

cf.get_obj_info(model)

