# EmbeddingLayerのテストCBOW版
# 左 ？ 右 の並びで左右から？を推定する
# 2024.02.02 A.Inoue
from pyaino.Config import *
set_np('numpy'); np=Config.np
print(np.__name__, 'is running in the background.', np.random.rand(1))
import matplotlib.pyplot as plt
from pyaino import Neuron, Functions, LossFunctions
from pyaino import common_function as cf

def arrange_time_data(corpus, time_size=3):
    """ 一つながりのデータから指定された時系列窓の大きさで切り出す """
    data = []
    for i in range(len(corpus)-time_size+1): 
        data.append(corpus[i:i+time_size])            
    data = np.array(data, dtype='int')
    print('データの形状：', data.shape)
    return data

def arrange_context_window(corpus, window_size=1):
    """ 指定された長さの前後のコンテクストと中央のターゲットを切り出す """
    context, target = [], []
    for i in range(len(corpus)-window_size*2): 
        context.append(corpus[i:i+window_size])
        context.append(corpus[i+window_size+1:i+window_size*2+1])
        target.append(corpus[i+window_size])
    context = np.array(context, dtype='int').reshape(-1, window_size*2)
    target  = np.array(target,  dtype='int')
    print('データの形状：', context.shape, target.shape)
    return context, target

class CBOW:
    """ 前後両側のコンテクストから中央ターゲットを得る """
    def __init__(self, V=10000, D=100, share_weight=True):
        # V:vocablary size 語彙数、D:word vector size 語ベクトルの大きさ
        self.input_layer   = Neuron.Embedding(V, D)
        self.mean_layer    = Functions.Mean(axis=1) # axis=1はwindow_sizeの軸
        self.output_layer  = Neuron.NeuronLayer(D, V, activate='Softmax')
        self.loss_function = LossFunctions.CrossEntropyError()
        self.share_weight  = share_weight 

    def forward(self, x, t=None, **kwargs):
        # B,ws = x.shape # wsは両側併せたコンテクスト長
        y = self.input_layer.forward(x)        # window_size分を纏めてforward # B,ws,D = y.shape
        y = self.mean_layer.forward(y)         # window_sizeの軸で平均をとる
        y = self.output_layer.forward(y)
        if t is None:
            return y
        vocab_size = y.shape[-1]
        c = np.eye(vocab_size)[t]
        l = self.loss_function.forward(y, c)
        return y, l

    def backward(self, gl=1):
        gy = self.loss_function.backward(gl)
        gx = self.output_layer.backward(gy)
        gx = self.mean_layer.backward(gx)
        self.input_layer.backward(gx)
        return None    

    def update(self, **kwargs):
        if self.share_weight:
            self.output_layer.parameters.grad_w \
                              += self.input_layer.parameters.grad_w.T
            self.output_layer.update(**kwargs)
            self.output_layer.parameters.b[...] = 0
            self.input_layer.parameters.w = self.output_layer.parameters.w.T
        else:
            self.input_layer.update(**kwargs)
            self.output_layer.update(**kwargs)

    def generate(self, x):
        y = self.forward(x)            
        created_data = np.argmax(y, axis=-1)
        for d in created_data:
            print(d, end=' ')
        return created_data    


# -- 各設定値 --
word_vector_size = 5 # 語ベクトルの大きさ 
window_size = 3
epoch = 1000
batch_size = 4 

# -- 訓練用のデータ --
corpus = range(100)      # 0～99の整数
vocab_size = len(corpus) # 語彙数
print('vocab_size =', vocab_size)
#input_data = arrange_time_data(corpus)
contexts, targets = arrange_context_window(corpus, window_size) 

# -- モデルの生成 --
model = CBOW(vocab_size, word_vector_size, share_weight=True)

# -- 学習 -- 
error_record = []
print('学習を開始します')
n_batch = len(targets) // batch_size  # 1エポックあたりのバッチ数
for i in range(epoch):
    err_a = 0
    index_random = np.arange(len(targets))
    np.random.shuffle(index_random)  # インデックスをシャッフルする
    for j in range(n_batch):
        idx = index_random[j*batch_size : (j+1)*batch_size]
        x = contexts[idx]
        t = targets[idx]
        y, l = model.forward(x, t)
        err_a += l
        model.backward()
        model.update(eta=0.3)
        error_record.append(float(l))

    print('Epoch: {:3d} | Error {:8.5f}'.format(i+1, float(err_a/n_batch)))

cf.graph_for_error(error_record)

# -- 最終確認 --
gen_data = model.generate(contexts)
print(all(gen_data==targets))

img1 = model.input_layer.parameters.w.T
img2 = model.output_layer.parameters.w
plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1) # 2行1列に分割し1番目
plt.imshow(img1.tolist(), aspect=vocab_size/word_vector_size/3)
plt.subplot(2, 1, 2) # 2行1列に分割し2番目
plt.imshow(img2.tolist(), aspect=vocab_size/word_vector_size/3)
plt.show()
