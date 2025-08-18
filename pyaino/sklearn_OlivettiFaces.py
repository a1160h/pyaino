from pyaino.Config import *
import matplotlib.pyplot as plt 
from pyaino import sklearn_datasets


# -- 顔データセットの読み込み --
def get_data(*arges, **kwargs):
    return sklearn_datasets.get_data('olivetti_faces', *arges, **kwargs)

# -- 名簿 --
def label_list(): # *は女性
    name = 'Jake', 'Tucker', 'Oliver', 'Cooper', 'Duke', 'Buster', 'Buddy', 'Kate*',\
           'Sam', 'Lora*', 'Toby', 'Cody', 'Ben', 'Baxter', 'Oscar', 'Rusty', 'Gizmo', \
           'Ted', 'Murphy', 'Alfie', 'Bentley', 'Wiston', 'William', 'Alex', 'Aaron', \
           'Colin','Daniel','Charlie','Connor','Devin', 'Henry','Sadie*','Ian','James', \
           'Gracie*','Jordan','Joseph','Kevin','Kyle','Luke' 
    print('名簿を提供します')
    return name

def analize():
    import matplotlib.pyplot as plt
    data, _, target = get_data()
    print(' max  = {:4.1f}\n min  = {:4.1f}\n mean = {:6.3f}\n std  = {:6.3f}\n'
          .format(np.max(data), np.min(data), np.mean(data), np.std(data)))
    plt.hist(data.reshape(-1).tolist(), bins=20, log=False)
    plt.title('value distribution(log scale)')
    plt.show()

# -- サンプルの提示と順伝播の結果のテスト --
def show_sample(data, label=None):
    sklearn_datasets.show_sample(data, label)

# -- 複数サンプルを表示(端数にも対応) --
def show_multi_samples(data=None, target=None, label=label_list()): # data, targetは対応する複数のもの
    sklearn_datasets.show_multi_samples(data, target, label, collection='olivetti_faces')


def data_expantion(data, target): # data と target は ndarray
    from PIL import Image
    data = data.reshape(-1, 64, 64) # 画像を加工するためにデータの形を整形 
    xdata=[]; xtarget=[]
    n_data = len(data); print('元のデータの数', n_data, '元のデータの形', data.shape)
    rt = 0.2
    dg = 15

    img = Image.fromarray(data[0])
    print('画像の情報', img.format, img.size, img.mode)

    for i in range(n_data):
        x =   data[i]
        t = target[i]
        x = x * 255 if np.max(x)==1 else x
        img = Image.fromarray(x) 
        ims = img.size
        imw = ims[0]; imh = ims[1]
      
        # 拡大縮小           left    upper  right      lower
        imgx  = img.crop(( imw*rt, imh*rt,imw*(1-rt),imh*(1-rt))).resize(ims)
        imgs  = img.crop((-imw*rt,-imh*rt,imw*(1+rt),imh*(1+rt))).resize(ims)
        # 回転
        imgr1 = img.rotate(dg)
        imgr2 = img.rotate(-dg)
        # 位置ずらし
        imgc1 = imgs.crop(( imw*rt, imh*rt,imw*(1+rt),imh*(1+rt))).resize(ims)
        imgc2 = imgs.crop((-imw*rt,-imh*rt,imw*(1-rt),imh*(1-rt))).resize(ims)
        imgc3 = imgs.crop(( imw*rt,-imh*rt,imw*(1+rt),imh*(1-rt))).resize(ims)
        imgc4 = imgs.crop((-imw*rt, imh*rt,imw*(1-rt),imh*(1+rt))).resize(ims)

        xdata.append(np.asarray(img)  /255); xtarget.append(t)
        xdata.append(np.asarray(imgx) /255); xtarget.append(t)
        xdata.append(np.array(imgs) /255); xtarget.append(t)
        xdata.append(np.array(imgr1)/255); xtarget.append(t)
        xdata.append(np.array(imgr2)/255); xtarget.append(t)
        xdata.append(np.array(imgc1)/255); xtarget.append(t)
        xdata.append(np.array(imgc2)/255); xtarget.append(t)
        xdata.append(np.array(imgc3)/255); xtarget.append(t)
        xdata.append(np.array(imgc4)/255); xtarget.append(t)

    xdata   = np.array(xdata).reshape(-1, 64*64) # データの形をもとに戻す　　
    xtarget = np.array(xtarget, dtype=int)

    return xdata, xtarget 

# -- 以下は実行サンプル --
if __name__=='__main__':
    analize()
    x, c, t = get_data()
    name = label_list()

    print('-- データの中身を確認 --')
    while True:
        try:
            i = int(input('見たいデータの番号'))
            pick_data = x[i]; pick_label = name[int(t[i])]    
        except:    
            print('-- テスト終了 --')
            break
        show_sample(pick_data, pick_label)
        
    rpr = [i*10 for i in range(len(name))]
    show_multi_samples(x[rpr], t[rpr], name)    
