# EmbeddingLayerのテストCBOW版
# 左 ？ 右 の並びで左右から？を推定する
# 2024.01.10 A.Inoue
from pyaino.Config import *
set_np('numpy'); np=Config.np

print(np.__name__, 'is running in the background.', np.random.rand(1))
import matplotlib.pyplot as plt
from pyaino import Neuron, LossFunctions
from pyaino import common_function as cf


# ネットワーク構築
def build(V=10000, D=100):#, share_weight=True):
    # V:vocablary size 語彙数、D:word vector size 語ベクトルの大きさ
    global embedding_layer1, embedding_layer2, output_layer
    global share_weight

    # 各層の初期化 
    embedding_layer1 = Neuron.Embedding(V, D)
    embedding_layer2 = Neuron.Embedding(V, D)
    output_layer     = Neuron.NeuronLayer(D, V, activate='SoftmaxWithLoss')

def forward(x1, x2, **kwargs):
    #print(x1.shape, x2.shape)
    B, T = x1.shape
    y1 = embedding_layer1.forward(x1)
    y2 = embedding_layer2.forward(x2)
    y = (y1 + y2) * 0.5
    y = output_layer.forward(y)
    return y

def backward(gy):
    gx = output_layer.backward(gy)
    embedding_layer1.backward(gx*0.5)
    embedding_layer2.backward(gx*0.5)

def update(**kwargs):
    if share_weight == True:
        embedding_layer1.grad_w += output_layer.grad_w.T
        embedding_layer1.update(**kwargs)
        embedding_layer2.w = embedding_layer1.w
        output_layer.w = embedding_layer1.w.T
    else:
        embedding_layer1.update(**kwargs)
        embedding_layer2.update(**kwargs)
        output_layer.update(**kwargs)

# -- 損失関数 --
def loss_function(y, c):
    return LossFunctions.CrossEntropyError().forward(y, c)

def generate(seed):
    x1 = seed[:,:,0]
    x2 = seed[:,:,2]
    y = forward(x1, x2)                # バッチ処理
    y = y.reshape(-1, vocab_size) # バッチ処理分も含めて時系列に並べる
    created_data = np.argmax(y, axis=1)
    for d in created_data:
        print(d, end=' ')

def arrange_time_data(corpus, time_size=7):
    print('\n一つながりのデータから学習データを切り出します')
    data = []
    for i in range(len(corpus)-2): 
        data.append(corpus[i:i+3])            
    data = np.array(data, dtype='int')
    data = data.reshape(-1, time_size, 3)
    print('データの形状：', data.shape)
    return data


# -- 各設定値 --
word_vector_size = 5 # 語ベクトルの大きさ 10
epoch = 300
batch_size = 4 # 128

# -- 訓練用のデータ --
corpus = range(100)      # 0～99の整数
vocab_size = len(corpus) # 語彙数

for i in corpus:
    print(i, end=' ')

# -- 時系列に並んだデータ=>入力、その次の１つのデータ=>正解値 --
input_data = arrange_time_data(corpus)

n_batch = len(input_data) // batch_size  # 1エポックあたりのバッチ数

# -- 各層の初期化 --
share_weight=True
build(vocab_size, word_vector_size)

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
        x1 = x[:,:,0]
        x2 = x[:,:,2]
        t  = x[:,:,1]
        y = forward(x1, x2)
        c = np.eye(vocab_size)[t]   #cf.convert_one_hot(x, vocab_size)
        error = loss_function(y, c) # 交差エントロピー誤差
        err_a += error
        backward(c)
        update(eta=2.0)
        error_record.append(float(error))

    # -- 誤差を求める --
    print('Epoch: {:3d} | Error {:8.5f}'.format(i+1, float(err_a/n_batch)))

cf.graph_for_error(error_record)

# -- 最終確認 --
generate(input_data)

img1 = embedding_layer1.w.T
img2 = embedding_layer2.w.T
img3 = output_layer.w
plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1) # 2行2列に分割し1番目
plt.imshow(img1.tolist(), aspect=vocab_size/word_vector_size/3)
plt.subplot(2, 2, 2) # 2行2列に分割し2番目
plt.imshow(img2.tolist(), aspect=vocab_size/word_vector_size/3)
plt.subplot(2, 2, 3) # 2行2列に分割し3番目
plt.imshow(img3.tolist(), aspect=vocab_size/word_vector_size/3)
plt.show()

