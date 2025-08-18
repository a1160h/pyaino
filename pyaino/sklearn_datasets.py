import matplotlib.pyplot as plt 
import random
from pyaino.Config import *
from pyaino import common_function as cf
from sklearn import datasets

def get_data(collection='iris', rate=None, one_hot=True, ndim=None, normalize=None):
    """ sklearnで提供される機械学習用のデータを用意する """
    if collection in('iris', 'Iris'):
        dataset = datasets.load_iris()
        image_size = None
    elif collection in('digits', 'Digits'):
        dataset = datasets.load_digits()
        image_size = 8
    elif collection in ('OlivettiFaces', 'olivetti_faces'):
        dataset = datasets.fetch_olivetti_faces()
        image_size = 64
    else:
        raise Exception('Bad collection specified')
    print('sklearn datasets', collection, 'is loaded.')
    print(dataset.keys())

    data   = np.array(dataset.data)
    target = np.array(dataset.target)

    # -- データ形状 --
    if ndim is not None and image_size:
        if   ndim==4:
            shape=(-1, 1, image_size, image_size)
        elif ndim==3:
            shape=(-1, image_size, image_size)
        elif ndim==2:
            shape=(-1, image_size*image_size)
        else:
            raise Exception("Wrong specification of 'ndim'")
        data = data.reshape(shape)
    data = data.astype('f4')

    print('data:', type(data), data.shape, data.dtype,
          '\ntarget:', type(target), target.shape, target.dtype)
    n_data = len(data)
    
    # -- データ標準化 --
    if normalize is not None:
        data = cf.normalize(data, normalize)

    # -- 正解をone-hot表現に --
    if one_hot:
        correct = np.eye(int(max(target))+1, dtype='int')[target]
    else:
        correct = None

    print('入力と正解値のデータが用意できました')
    print('入力データの形状', data.shape,    data.dtype)
    if one_hot:
        print('one_hotの正解', correct.shape, correct.dtype)    
    print('正解値の形状',     target.shape, target.dtype)

    if rate == None:
        return data, correct, target

    # -- 訓練データとテストデータの分離 data => x, correct => y, target => t --
    print('関数get_dataの引数でテストデータの割合を0～1の範囲で指定してください')
    print('指定された割合 =', rate)
    index = range(n_data)
    index_train = []; index_test = [] 
    if 0 < rate < 1.0:
        index_test  = random.sample(index, int(n_data*rate))
    for i in range(n_data):
        if index[i] in index_test:
            pass
        else:
            index_train.append(i)

    x_train = data   [index_train, :]  # 訓練 入力
    c_train = correct[index_train, :]  # 訓練 正解 one hot
    t_train = target [index_train]     # 訓練 正解
    x_test  = data   [index_test, :]   # テスト入力
    c_test  = correct[index_test, :]   # テスト正解 one hot
    t_test  = target [index_test]      # テスト正解　

    print('入力と正解値のデータを用意し、訓練データとテストデータに分離しました')
    print('x_train:', x_train.shape, 'c_train:', c_train.shape, 't_train:', t_train.shape)
    print('x_test:',  x_test .shape, 'c_test:',  c_test .shape, 't_test:',  t_test .shape)

    return x_train, c_train, t_train, x_test, c_test, t_test

# -- サンプルの表示 --
def show_sample(data, label=None):
    data = np.array(data.tolist())
    image_size = data.size
    ih = iw = int(image_size**0.5)
    print(ih, iw)
    max_picel = np.max(data); min_picel = np.min(data) # 画素データを0～1に補正
    rdata = (data - min_picel)/(max_picel - min_picel)
    rdata = rdata.reshape(ih, iw)
    plt.imshow(rdata.tolist(), cmap='gray')
    plt.title(str(label))
    plt.show()
        
# -- 複数サンプルを表示(端数にも対応) --
def show_multi_samples(data=None, target=None, label=None, collection=None): # data, targetは対応する複数のもの
    if data is None:
        if collection is None:
            raise Exception('collection need to be specified.')
        data = get_data(collection)[0][:50]
    data = np.array(data.tolist()) 
    image_size = data[0].size
    ih = iw = int(image_size**0.5)
    data = data.reshape(-1, ih, iw); n_data = len(data)
    n = 50 # 一度に表示する画像数
    for j in range(0, n_data, n):   # はじめのn個、次のn個と進める
        x = data[j:]
        t = target[j:] if target is not None else None
        plt.figure(figsize=(18, 10))
        m = min(n, n_data - j)      # n個以上残っていればn個、n個に満たない時はその端数
        for i in range(m):
            plt.subplot(5, 10, i+1) # 5行10列のi+1番目
            plt.imshow(x[i].tolist(), cmap='gray')
            if label is not None and target is not None:
                plt.title(label[int(t[i])])
            elif target is not None:
                plt.title(int(t[i]))
            plt.axis('off')
        plt.show()        

# -- 以下は実行サンプル --
if __name__=='__main__':
    data = get_data(rate=0.1)
    data = get_data('digits', 0.2)
    data = get_data('olivetti_faces', 0.3, normalize='-1to1')



