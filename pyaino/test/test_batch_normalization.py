# test_batch_normalization
# 20250220 A.Inoue
from pyaino.Config import *
import matplotlib.pyplot as plt
from pyaino import Neuron
from pyaino import CIFER10

# -- テストデータ準備 --
X, C, T, _, _, _ = CIFER10.get_data(image=True, normalize=False)
n_data = len(X)
print(n_data, '個のデータを取得しました')
if X.ndim <= 3: # 白黒画像
    B, Ih, Iw = X.shape; C = 1
else:           # カラー画像
    B, Ih, Iw, C = X.shape

# -- データ切出し --
B = 10      # バッチサイズ
offset = 68 # 先頭からのオフセット

idx = np.random.randint(0, n_data, B) # 0～n_dataのB個の整数
print('idx =', idx) 
x = X[idx]
print('x.shape =', x.shape)

# -- テストモデルの設定 --　
model = Neuron.BatchNormalization()#scale_and_bias=False)

# -- テスト実行 --
y = model.forward(x, train=True)
#gy = np.empty(y.shape); gy[...] = 1
gy = np.random.randn(*y.shape)
gx = model.backward(gy)


# -- 結果表示 --
print('---------------------------------------------------------------------')
print('データの素性 x   最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
      .format(float(np.max(x)), float(np.min(x)),  \
             float(np.mean(x)), float(np.var(x)) ))
print('---------------------------------------------------------------------')
print('データの素性 y   最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
      .format(float(np.max(y)), float(np.min(y)),  \
             float(np.mean(y)), float(np.var(y)) ))
print('---------------------------------------------------------------------')
print('データの素性 gy  最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
      .format(float(np.max(gy)), float(np.min(gy)),  \
             float(np.mean(gy)), float(np.var(gy)) ))
print('---------------------------------------------------------------------')
print('データの素性 gx  最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
      .format(float(np.max(gx)), float(np.min(gx)),  \
             float(np.mean(gx)), float(np.var(gx)) ))
print('---------------------------------------------------------------------')

if X.ndim > 3: # カラー画像の場合は表示のための調整が必要
    x   =   x.transpose(0, 2, 3, 1) 
    y   =   y.transpose(0, 2, 3, 1) 
    gx  =  gx.transpose(0, 2, 3, 1)

    x   = (x   - np.min(x))  / (np.max(x)   - np.min(x))
    y   = (y   - np.min(y))  / (np.max(y)   - np.min(y))
    gx  = (gx  - np.min(gx)) / (np.max(gx)  - np.min(gx))
    
plt.figure(figsize=(12, 6))
for i in range(B):
    # 入力
    ax = plt.subplot(5, B, i+1)
    plt.imshow(x[i].tolist())
    ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
    # 出力 　　　
    ax = plt.subplot(5, B, i+1+B)
    plt.imshow(y[i].tolist())
    ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
    # 入力の勾配 　　　
    ax = plt.subplot(5, B, i+1+3*B)
    plt.imshow(gx[i].tolist())
    ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
plt.show()


