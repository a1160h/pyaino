from pyaino.Config import *
set_derivative(True)
from pyaino import Neuron as nn
from pyaino import LossFunctions as lf
from pyaino import common_function as cf
import time


x = np.arange(1*3*4*4, dtype=Config.dtype).reshape(1,3,4,4)
print(x)

# -- 学習の設定 --
epoch = 10       # 学習回数(1epochでデータをすべて使い切る)　　　     
interval = 1      # 経過の表示間隔

H = 128
patch_size = 4
kwargs = {'optimize' : 'AdamT', 'w_decay' : 0.001}
img_size = x.shape[1:]

model = nn.SequentialWithLoss(
    nn.PatchEmbeddingSimple(H, patch_size, **kwargs),
    nn.Unpatchfy(img_size, patch_size, **kwargs),
    lf.MeanSquaredError(),
    )

cf.get_obj_info(model)


# -- 学習と経過の記録 --
for i in range(300):
    y, loss = model(x, x)
    model.backward()
    #loss.backtrace()
    model.update(eta=0.0003)
    print(i, loss)    

print(y)
