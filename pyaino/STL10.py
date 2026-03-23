from pyaino.Config import *
from pyaino import common_function as cf
print(np.__name__, 'is running in', __file__, np.random.rand(1))    
import matplotlib.pyplot as plt

path = __file__ + '/../../stl10_binary/'
print(path) 

def read_stl10_images(file):
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    data = data.reshape(-1, 3, 96, 96)
    data = np.transpose(data, (0, 1, 3, 2))  # CHW に変換
    return data

def read_stl10_targets(file):
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
        data -= 1 # 1～10 -> 0 ～ 9
    return data

def read_class_names(path=path):
    with open(path+'class_names.txt', "r") as  f:
        labels = f.read().splitlines() # 呼んでリスト形式に変換
    return labels

label_list = read_class_names()

def get_data(path=path, **kwargs):
    # unlabeledが指定されたら、そのデータだけを返す
    if kwargs.pop('unlabeled', False):
        x_train = read_stl10_images(path + 'unlabeled_X.bin')
        return x_train
    
    # 上記以外の通常の場合
    x_train = read_stl10_images(path + 'train_X.bin')
    t_train = read_stl10_targets(path + 'train_y.bin')
    x_test  = read_stl10_images(path + 'test_X.bin')
    t_test  = read_stl10_targets(path + 'test_y.bin')

    print('STL10のデータが読み込まれました')

    # -- データの軸の入替え --
    if kwargs.pop('transpose', False):
        print('データの軸を画像表示用に入れ替えます (B, C, Ih, Iw) -> (B, Ih, Iw, C)')
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test  = x_test .transpose(0, 2, 3, 1)    

    # -- データの標準化 --
    normalize = kwargs.pop('normalize', True)
    if normalize is not None: # True, 0to1, -1to1 など指定可能 
        x_train = cf.normalize(x_train, normalize)
        x_test  = cf.normalize(x_test,  normalize)

    # -- 正解をone-hot表現に --
    c_train = np.eye(10)[t_train].astype(Config.dtype)
    c_test  = np.eye(10)[t_test] .astype(Config.dtype)    

    print('訓練用の入力と正解値のデータが用意できました')
    print('入力データの形状', x_train.shape, '正解値の形状', c_train.shape)    
    print('データの素性 最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
        .format(float(np.max(x_train)), float(np.min(x_train))\
              , float(np.mean(x_train)), float(np.var(x_train))))
    print('評価用の入力と正解値のデータが用意できました')
    print('入力データの形状', x_test.shape, '正解値の形状', c_test.shape)
    print('データの素性 最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
        .format(float(np.max(x_test)), float(np.min(x_test))\
              , float(np.mean(x_test)), float(np.var(x_test))))

    return x_train, c_train, t_train, x_test, c_test, t_test

# -- サンプルの提示と順伝播の結果のテスト --
# CNNで扱う場合は C, Ih, Iw が良く　
# 画像表示には　　Ih, Iw, C が良い
# NNで扱う場合はベクトルだが C*Ih*Iw Ih*Iw*C どちらか不明
def show_sample(data, label=None):
    #print('### debug', data.shape)
    rdata = data[0] if data.ndim==4 else data
    rdata = data.reshape(3, 32, 32) if data.ndim<=2 else rdata
    rdata = rdata.transpose(1, 2, 0) if rdata.shape[0]==3 else rdata 
    max_picel = np.max(rdata); min_picel = np.min(rdata) # 画素データを0～1に補正
    rdata = (rdata - min_picel)/(max_picel - min_picel)
    plt.imshow(rdata.tolist())
    if label:
        plt.title(label)
    plt.show()

# -- 複数サンプルを表示(端数にも対応) --
def show_multi_samples(data=None, target=None, label_list=None, maxis=(1,2,3)):
    if data is None:
        data = get_data()
        if target is None:
            target = data[2][:100]
        if label_list is None:    
            label_list = read_class_names()
        data = data[0][:100]
        
    # data, targetは対応する複数のもの
    rdata = data.transpose(0, 2, 3, 1) if data.shape[1]==3 else data

    # 画素データを0～1に補正
    if maxis: # 補正軸
        max_picel = np.max(rdata, axis=maxis, keepdims=True)
        min_picel = np.min(rdata, axis=maxis, keepdims=True) 
        rdata = (rdata - min_picel)/(max_picel - min_picel)

    n_data = len(data)
    n = 50 # 一度に表示する画像数

    for j in range(0, n_data, n):   # はじめのn個、次のn個と進める
        x = rdata[j:]
        if target is not None:
            t = target[j:]
        plt.figure(figsize=(18, 10))
        m = min(n, n_data - j)      # n個以上残っていればn個、n個に満たない時はその端数
        for i in range(m):
            plt.subplot(5, 10, i+1) # 5行10列のi+1番目
            plt.imshow(x[i].tolist())
            if target is not None and label_list is not None:
                plt.title(label_list[int(t[i])])
            plt.axis('off')
        plt.show()        



# -- 以下は実行サンプル --
if __name__=='__main__':

    x_train, c_train, t_train, x_test, c_test, t_test = get_data()#normalize=True)
    #label_list = read_class_names()
    print(label_list)

    print('-- データの中身を確認 train --')
    while True:
        try:
            i = int(input('見たいデータの番号'))
            pick_data  = x_train[i]
            pick_label = label_list[int(t_train[i])]    
        except:    
            print('-- テスト終了 --')
            break
        print(pick_data.shape)
        show_sample(pick_data, pick_label)
        
    print('-- データの中身を確認 test --')
    while True:
        try:
            i = int(input('見たいデータの番号'))
            pick_data  = x_test[i]
            pick_label = label_list[int(t_test[i])]    
        except:    
            print('-- テスト終了 --')
            break
        show_sample(pick_data, pick_label)
    #'''#
    show_multi_samples() 
