from pyaino.Config import *
import matplotlib.pyplot as plt 
from pyaino import sklearn_datasets
#import os

# -- 手書き文字データセットの読み込み --
def get_data(*arges, **kwargs):
    return sklearn_datasets.get_data('digits', *arges, **kwargs)

# -- サンプルの表示 --
def show_sample(data, label=None):
    sklearn_datasets.show_sample(data, label)
      
# -- 複数サンプルを表示(端数にも対応) --
def show_multi_samples(data=None, target=None): # data, targetは対応する複数のもの
    sklearn_datasets.show_multi_samples(data, target, collection='digits')

def analize():
    import matplotlib.pyplot as plt
    data, _, target = get_data()
    print(' max  = {:4.1f}\n min  = {:4.1f}\n mean = {:6.3f}\n std  = {:6.3f}\n'
          .format(np.max(data), np.min(data), np.mean(data), np.std(data)))
    plt.hist(data.reshape(-1).tolist(), bins=int(np.max(data)-np.min(data)),
             log=True)
    plt.title('value distribution(log scale)')
    plt.show()

    

# -- 以下は実行サンプル --
if __name__=='__main__':
    analize()
    x_train, y_train, t_train, x_test, y_test, t_test = get_data(0.3)

    print('-- データの中身を確認 train --')
    while True:
        try:
            i = int(input('見たいデータの番号'))
            pick_data = x_train[i]; pick_label = t_train[i]    
        except:    
            print('-- テスト終了 --')
            break
        show_sample(pick_data, pick_label)
        
    print('-- データの中身を確認 test --')
    while True:
        try:
            i = int(input('見たいデータの番号'))
            pick_data = x_test[i]; pick_label = t_test[i]    
        except:    
            print('-- テスト終了 --')
            break
        show_sample(pick_data, pick_label)

    show_multi_samples(x_train[:100], t_train[:100])
    show_multi_samples(x_test[:50], t_test[:50])
