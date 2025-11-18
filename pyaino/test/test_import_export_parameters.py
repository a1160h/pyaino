from pyaino.Config import *
from pyaino import NN_CNN
from pyaino import common_function as cf

model = NN_CNN.NN_m(1, ml_nn=3, debug_mode=True)

x = np.arange(6).reshape(2, 3)
params0 = cf.export_parameters(model).copy()
print('初期化前　params0\n', params0)
y = model.forward(x)
print(x)
print(y)

params = cf.export_parameters(model)
print('初期化後　params\n', params)

params10 = {}
for k, v in params.items():
    if v is not None:
        params10[k] = v*10

print('10倍　params10\n', params10)
cf.import_parameters(model, params0)
print('当初の　params0\n', params0)

cf.import_parameters(model, params10)
params = cf.export_parameters(model)
print('挿入後　params\n', params)

