# EmbeddingLayerのテストCBOW版
# 左 ？ 右 の並びで左右から？を推定する
# 2024.03.12 A.Inoue
from pyaino.Config import *
set_np('numpy'); np=Config.np
import matplotlib.pyplot as plt
from pyaino import Neuron, LossFunctions, Functions
from pyaino import common_function as cf
from pyaino import Activators

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

def arrange_samples(corpus, targets, sampling_size=1):
    targets = targets.reshape(-1, 1)
    negative_samples = np.random.choice(corpus, size=len(targets)*sampling_size)
    negative_samples = negative_samples.reshape(len(targets), sampling_size)
    samples = np.concatenate([targets, negative_samples], axis=-1)
    results = np.array([samples[i]==t for i, t in enumerate(targets)])
    return samples, results


class CBOW_negative_sampling:
    """ 前後両側のコンテクストから中央ターゲットを得る """
    def __init__(self, V=10000, D=100, window_size=1, sampling_size=1, share_weight=True):
        # V:vocablary size 語彙数、D:word vector size 語ベクトルの大きさ
        # window_size:片側コンテクスト長、sampling_size：negative samplingの数
        self.embedding_layer1 = Neuron.Embedding(V, D)
        self.embedding_layer2 = Neuron.Embedding(V, D)
        self.window_size = window_size
        self.sampling_size = sampling_size
        self.mean = Functions.Mean(axis=1)
        self.mul = Functions.Mul()
        self.sum = Functions.Sum(axis=2)
        self.sigmoid = Activators.Sigmoid()
        self.loss_function = LossFunctions.CrossEntropyError()
        self.share_weight = share_weight 

    def forward(self, x, t=None, r=None, **kwargs):
        # x:入力、t:ターゲットを含むサンプル、r:サンプルに対する正解値        
        # 入力xの分散表現の平均
        y = self.embedding_layer1.forward(x)
        B, _, D = y.shape
        y = self.mean(y)
        # ターゲットを含むサンプルtの分散表現
        z = self.embedding_layer2.forward(t)
        # yとzの類似度
        y = y.reshape(-1,1,D)
        s = self.mul(y, z)
        s = self.sum(s)
        s = self.sigmoid(s)
        # 損失
        l = self.loss_function.forward(s, r)
        y = y.reshape(-1, D)
        return y, l

    def backward(self, gl=1):
        gs = self.loss_function.backward(gl)
        gs = self.sigmoid.backward(gs)
        gs = self.sum.backward(gs)
        gy, gz = self.mul.backward(gs)
        self.embedding_layer2.backward(gz)
        if len(gy.shape) == 3: # 追加 20250331N
            if gy.shape[1] == 1: # 追加 20250331N
                gy = gy.reshape(gy.shape[0],gy.shape[2]) # 追加 20250331N
        gy = self.mean.backward(gy)
        self.embedding_layer1.backward(gy)
        return None    

    def update(self, **kwargs):
        if self.share_weight:
            self.embedding_layer1.grad_w += self.embedding_layer2.grad_w
            self.embedding_layer1.update(**kwargs)
            self.embedding_layer2.w = self.embedding_layer1.w
        else:
            self.embedding_layer1.update(**kwargs)
            self.embedding_layer2.update(**kwargs)

# -- 各設定値 --
word_vector_size = 5 # 語ベクトルの大きさ 
window_size = 3
sampling_size = 5
epoch = 1000
batch_size = 10

# -- 訓練用のデータ --
corpus = list(range(100))      # 0～99の整数
vocab_size = len(corpus) # 語彙数
#input_data = arrange_time_data(corpus)
contexts, targets = arrange_context_window(corpus, window_size)
samples,  results = arrange_samples(corpus, targets, sampling_size)

# -- モデルの生成 --
model = CBOW_negative_sampling(
            vocab_size, word_vector_size, window_size, sampling_size,
            share_weight=True)

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
        t = samples[idx]
        r = results[idx]
        y, l = model.forward(x, t, r) # yは語ベクトル(分散表現)の並び
        err_a += l
        model.backward()
        model.update(eta=0.3)
        error_record.append(float(l))

    print('Epoch: {:3d} | Error {:8.5f}'.format(i+1, float(err_a/n_batch)))

cf.graph_for_error(error_record)

# -- 最終確認 --
img1 = model.embedding_layer1.w.T
img2 = model.embedding_layer2.w.T
plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1) # 2行2列に分割し1番目
plt.imshow(img1.tolist(), aspect=vocab_size/word_vector_size/3)
plt.subplot(2, 1, 2) # 2行2列に分割し2番目
plt.imshow(img2.tolist(), aspect=vocab_size/word_vector_size/3)
plt.show()

