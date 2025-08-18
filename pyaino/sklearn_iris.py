from pyaino import sklearn_datasets

def get_data(*arges, **kwargs):
    return sklearn_datasets.get_data('iris', *arges, **kwargs)

def label_list():
    target_names = ['setosa', 'versicolor', 'virginica']
    print('品種名を提供します')
    return target_names

def analize():
    import matplotlib.pyplot as plt
    import numpy
    data, _, target = get_data()
    print(' max  = {:4.1f}\n min  = {:4.1f}\n mean = {:6.3f}\n std  = {:6.3f}\n'
          .format(numpy.max(data), numpy.min(data), numpy.mean(data), numpy.std(data)))

    data = numpy.array(data.tolist())
    target = numpy.array(target.tolist())

    C = numpy.corrcoef(data, rowvar=False) # dataの列ごとの相関係数を求めたいからrowvar=False　
    print('相関係数　　ガクの長さ,　ガクの幅,　　花弁の長さ,　花弁の幅',
          '\nガクの長さ', C[0],
          '\nカクの幅　', C[1],
          '\n花弁の長さ', C[2],
          '\n花弁の幅　', C[3])

    fig = plt.figure(figsize=(10, 10))

    ax00 = fig.add_subplot(4, 4, 1)
    ax01 = fig.add_subplot(4, 4, 2)
    ax02 = fig.add_subplot(4, 4, 3)
    ax03 = fig.add_subplot(4, 4, 4)
    ax11 = fig.add_subplot(4, 4, 6)
    ax12 = fig.add_subplot(4, 4, 7)
    ax13 = fig.add_subplot(4, 4, 8)
    ax22 = fig.add_subplot(4, 4, 11)
    ax23 = fig.add_subplot(4, 4, 12)
    ax33 = fig.add_subplot(4, 4, 16)

    tgt0, = numpy.where(target==0)  
    tgt1, = numpy.where(target==1)  
    tgt2, = numpy.where(target==2) 

    ax01.scatter(data[tgt0,0], data[tgt0,1])
    ax01.scatter(data[tgt1,0], data[tgt1,1])
    ax01.scatter(data[tgt2,0], data[tgt2,1])
    ax02.scatter(data[tgt0,0], data[tgt0,2])
    ax02.scatter(data[tgt1,0], data[tgt1,2])
    ax02.scatter(data[tgt2,0], data[tgt2,2])
    ax03.scatter(data[tgt0,0], data[tgt0,3])
    ax03.scatter(data[tgt1,0], data[tgt1,3])
    ax03.scatter(data[tgt2,0], data[tgt2,3])
    ax12.scatter(data[tgt0,1], data[tgt0,2])
    ax12.scatter(data[tgt1,1], data[tgt1,2])
    ax12.scatter(data[tgt2,1], data[tgt2,2])
    ax13.scatter(data[tgt0,1], data[tgt0,3])
    ax13.scatter(data[tgt1,1], data[tgt1,3])
    ax13.scatter(data[tgt2,1], data[tgt2,3])
    ax23.scatter(data[tgt0,2], data[tgt0,3])
    ax23.scatter(data[tgt1,2], data[tgt1,3])
    ax23.scatter(data[tgt2,2], data[tgt2,3])

    ax00.hist((data[tgt0,0], data[tgt1,0], data[tgt2,0]), histtype='barstacked')
    ax11.hist((data[tgt0,1], data[tgt1,1], data[tgt2,1]), histtype='barstacked')
    ax22.hist((data[tgt0,2], data[tgt1,2], data[tgt2,2]), histtype='barstacked')
    ax33.hist((data[tgt0,3], data[tgt1,3], data[tgt2,3]), histtype='barstacked')

    fig.tight_layout()              
    plt.show()

if __name__=='__main__':
    analize()


