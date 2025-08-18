# test_batch_normalization
# 20250227 A.Inoue
from pyaino.Config import *
import copy
from pyaino import Neuron
from pyaino import CIFER10

# -- テストデータ準備 --
x = np.random.rand(24).reshape(4, 3, 2)
#print(x)

# -- テストモデルの設定 --　
model1 = Neuron.BatchNormalization()
model2 = Neuron.batch_normalization2()

# -- テスト実行 (インプレース演算の対応が必要) --
y1 = model1.forward(x.copy(), train=True) 
y2 = model2.forward(x.copy(), train=True)

assert np.allclose(y1, y2, rtol=1e-5, atol=1e-5), \
       'Forward pass incorrect \ny1\n{}, \ny2\n{}'.format(y1, y2)
print('Forward  pass test passed.', y1.shape, y2.shape)

gy = np.random.randn(*y1.shape)
gx1 = model1.backward(gy.copy())
gx2 = model2.backward(gy.copy())

assert np.allclose(gx1, gx2, rtol=1e-3, atol=1e-3),\
       'Backward pass incorrect \ngx1\n{}, \ngx2\n{}'.format(gx1, gx2)
print('Backward pass test passed.', gx1.shape, gx2.shape)

# -- 結果表示 --
print('---------------------------------------------------------------------')
print('データの素性 x   最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
      .format(float(np.max(x)), float(np.min(x)),  \
             float(np.mean(x)), float(np.var(x)) ))
print('---------------------------------------------------------------------')
print('データの素性 y1   最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
      .format(float(np.max(y1)), float(np.min(y1)),  \
             float(np.mean(y1)), float(np.var(y1)) ))
print('データの素性 y2   最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
      .format(float(np.max(y2)), float(np.min(y2)),  \
             float(np.mean(y2)), float(np.var(y2)) ))
print('---------------------------------------------------------------------')
print('データの素性 gy  最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
      .format(float(np.max(gy)), float(np.min(gy)),  \
             float(np.mean(gy)), float(np.var(gy)) ))
print('---------------------------------------------------------------------')
print('データの素性 gx1  最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
      .format(float(np.max(gx1)), float(np.min(gx1)),  \
             float(np.mean(gx1)), float(np.var(gx1)) ))
print('データの素性 gx2  最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
      .format(float(np.max(gx2)), float(np.min(gx2)),  \
             float(np.mean(gx2)), float(np.var(gx2)) ))
print('---------------------------------------------------------------------')

