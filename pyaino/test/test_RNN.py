# train_RNN_el2f_野菊の墓
# 2020.04.30 A.Inoue
from pyaino.Config import *
set_np('numpy'); np=Config.np
print(np.__name__, 'is running in the background.', np.random.rand(1))

set_derivative(True) # 追加 20250327N

from pyaino import common_function as cf
from pyaino import Neuron

# -- 各設定値 --
time_size = 10     # 時系列の数 20
hidden_size = 20   # 中間層のニューロン数 128
batch_size = 1
CPT = None #1 # None         # RNN出力のキャプチャ幅

corpus = np.array([0,1,2,3,4])

vocab_size = 5

print(corpus)

time_size = 4
input_data, correct_data = cf.arrange_time_data(corpus, time_size, CPT=CPT, step=1)
print('input_data\n', input_data)
print('correct_data\n', correct_data)

data = cf.convert_one_hot(input_data, vocab_size)    # one hotに変換
print('data_shape', data.shape)

# -- 各層の初期化 --
RNN = Neuron.RNN(vocab_size, 3,debug_mode=True)
# RNN.init_parameter()
RNN.set_state(data)
# RNN.w[...]=0.2
# RNN.v[...]=0.2

idx = 0
x = data.reshape(-1,time_size, vocab_size)

y = RNN.forward(x, CPT=CPT)

#print(x)
print(y)



