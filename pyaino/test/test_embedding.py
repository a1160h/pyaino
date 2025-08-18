# EmbeddingLayerのテスト
# 2022.05.16 A.Inoue
from pyaino.Config import *
set_np('numpy'); np = Config.np
print(np.__name__, 'is running in the background.', np.random.rand(1))
import matplotlib.pyplot as plt
from pyaino import Neuron, LossFunctions
from pyaino import common_function as cf

# ネットワーク構築
class TestClass:
    def __init__(self,V=10000, D=100):
        # V:vocablary size 語彙数、D:word vector size 語ベクトルの大きさ
        global embedding_layer, output_layer
        global share_weight
       
        share_weight = True  # Embeddingと全結合層の重み共有 　

        # 各層の初期化 
        self.embedding_layer = Neuron.Embedding(V, D)
        self.output_layer    = Neuron.NeuronLayer(D, V,bias=True, activate='SoftmaxWithLoss')
     
        # if share_weight == True:
        #     self.output_layer.w = self.embedding_layer.w.T

    def forward(self,x, **kwargs):
        B, T = x.shape # バッチサイズと時間サイズ
        y = self.embedding_layer.forward(x)
        y = self.output_layer.forward(y)
        return y

    def backward(self,gy):
        gx = self.output_layer.backward(gy)
        self.embedding_layer.backward(gx)
        return gx

    def update(self,**kwargs):
        if share_weight == True:
            self.embedding_layer.grad_w += self.output_layer.grad_w.T
            self.embedding_layer.update(**kwargs)
            self.output_layer.w = self.embedding_layer.w.T
        else:
            self.embedding_layer.update(**kwargs)
            self.output_layer.update(**kwargs)

    # -- 損失関数 --
    def loss_function(self,y, c):
        return LossFunctions.CrossEntropyError().forward(y, c)

    def generate(self,seed):
        x = np.array(seed).reshape(-1, time_size)
        y = self.forward(x)                # バッチ処理
        y = y.reshape(-1, vocab_size) # バッチ処理分も含めて時系列に並べる
        created_data = np.argmax(y, axis=1)
        for d in created_data:
            print(d, end=' ')

def arrange_time_data(corpus, time_size, step=None):
    print('一つながりのデータから時系列長の入力データとそれに対する正解データを切り出します')
    data = []
    if step is None:
        step = time_size
    print('時系列長は',time_size, 'データの切出し間隔は', step, 'です')
    for i in range(0, len(corpus)-time_size+1, step):   # 時系列長＋１の長さのデータを一括して
        data.append(corpus[i : i+time_size])            # step幅ずつずらして切出す
    data = np.array(data, dtype='int')
    print('入力データの形状：', data.shape)
    return data


# -- 各設定値 --
time_size = 20        # 時系列の数 20
word_vector_size = 5 # 語ベクトルの大きさ 10
epoch = 200
batch_size = 4 # 128

# -- 訓練用のデータ --
corpus = range(100)      # 0～99の整数
vocab_size = len(corpus) # 語彙数

for i in corpus:
    print(i, end=' ')

# -- 時系列に並んだデータ=>入力、その次の１つのデータ=>正解値 --
input_data = arrange_time_data(corpus, time_size, step=10)
#input_data, correct_data = cf.arrange_time_data(corpus, time_size, step=10)

print('入力データの形状：', input_data.shape)
n_batch = len(input_data) // batch_size  # 1エポックあたりのバッチ数

# -- 各層の初期化 --
model=TestClass(vocab_size, word_vector_size)

# -- 学習 -- 
error_record = []
print('学習を開始します')
for i in range(epoch):
    err_a = 0
    index_random = np.arange(len(input_data))
    np.random.shuffle(index_random)  # インデックスをシャッフルする

    for j in range(n_batch):
        # ミニバッチを取り出す
        idx = index_random[j*batch_size : (j+1)*batch_size]
        x = input_data[idx]
        #t = correct_data[idx]
        y = model.forward(x)
        c = np.eye(vocab_size)[x]   #cf.convert_one_hot(x, vocab_size)
        #c = np.eye(vocab_size)[t]   #cf.convert_one_hot(x, vocab_size)
        error = model.loss_function(y, c) # 交差エントロピー誤差
        err_a += error
        model.backward(c)
        model.update(eta=10.0)
        error_record.append(float(error))

    # -- 誤差を求める --
    print('Epoch: {:3d} | Error {:8.5f}'.format(i+1, float(err_a/n_batch)))

cf.graph_for_error(error_record)

# -- 最終確認 --
model.generate(corpus)

img1 = model.embedding_layer.w.T
img2 = model.output_layer.w
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1) # 2行1列に分割し1番目
plt.imshow(img1.tolist(), aspect=vocab_size/word_vector_size/3)
plt.subplot(2, 1, 2) # 2行1列に分割し2番目　
plt.imshow(img2.tolist(), aspect=vocab_size/word_vector_size/3)
plt.show()



