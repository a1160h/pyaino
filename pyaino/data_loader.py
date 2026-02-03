from pyaino.Config import *
#set_np('numpy');np=Config.np
from concurrent.futures import ThreadPoolExecutor # 別スレッドで前処理を走らせる
import threading                                  # 別スレッドで前処理を走らせる
from queue import Queue                     # スレッド間で安全にデータを受渡す箱
from PIL import Image, ImageOps
import glob, os
import numpy                                # numpyを明示的に使う 
from pyaino import common_function as cf

class ImageLoader:
    """
    イテレータ、ジェネレータ、スレッド、キューを最小限かつ適切に組み合わせた
    先読み機能付きのデータローダで、以下のようしてバッチを取り出せる仕組みを備える
    for batch in loader:
          …
    """
    def __init__(self, file_list, batch_size=32, prefetch=4, shuffle=True,
                 resize=None, transpose=False, normalize=False,
                 ):
        self.file_list = file_list   # データファイルのパスのリスト
        self.batch_size = batch_size # 1バッチ当たりのデータ数
        self.prefetch = prefetch     # 何バッチ先までキューに溜めておくか 
        self.shuffle = shuffle       # エポック毎にシャッフルするか

        self.queue = Queue(maxsize=prefetch) # 最大prefetch個のバッチを保持
                                             # 満杯になるとput()をブロック
        self.stop_event = threading.Event()  # スレッドに停止合図を送るフラグ

        self.executor = ThreadPoolExecutor(max_workers=1) # ワーカスレッドは1本
                                     # 並列化の目的は「計算」ではなく「I/O隠蔽」
        self.future = None

        self.resize = resize         # 横・縦をタプルで指定
        self.transpose = transpose
        self.normalize = normalize

    def _load_image(self, path):
        """ 画像を1枚読み込む """
        img = Image.open(path).convert('RGB') # 白黒が混じってもshapeを統一
        if self.resize is not None:
            img = self._resize_image(img)
        return numpy.asarray(img, dtype=numpy.uint8) # numpy配列化(0-255)

    def _resize_image(self, img):
        """ 画像のサイズ変更 """
        img = ImageOps.fit(img, self.resize, Image.LANCZOS)
        return img

    def _batch_generator(self):
        """ バッチを生成するジェネレータ(_prefetch_loopから呼ばれる) """
        files = self.file_list.copy() # 元のリストを壊さないためにcopy()
        if self.shuffle:
            numpy.random.shuffle(files)

        for i in range(0, len(files), self.batch_size): # バッチ分割(端数は最後の小バッチ)
            batch_files = files[i:i + self.batch_size]
            batch = [self._load_image(f) for f in batch_files]
            yield numpy.stack(batch) # 一度に全部返さずに1バッチずつ順に生成

    def _prefetch_loop(self):
        """ 先読みの本体(別スレッドで走る関数) """
        for batch in self._batch_generator(): # バッチを生成
            if self.stop_event.is_set(): # スレッドへの停止合図 
                break
            self.queue.put(batch) # バッチをキューに格納(キューが満杯なら自動的に待つ)
        self.queue.put(None)      # 終了シグナル、Noneは「もうデータは無い」の印

    def _formatting(self, x):
        """ cpu上(numpy)でデータの整形 """
        # data は numpy のまま整形する（先読みスレッド安全）
        if self.transpose:
            x = x.transpose(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)
        if self.normalize:
            # dtype 変換（CPU上で型だけConfig.dtypeに揃える）
            x = x.astype(getattr(numpy, Config.dtype), copy=False)
            x = self.normalization(x)
        return x

    def __iter__(self):
        """ __next__と併せてイテレータとして振る舞う仕組み """
        # for batch in loader;のforが始まると呼ばれる
        self.stop_event.clear()
        # バックグラウンドスレッドを起動(submitにより_prefetch_loopの非同期実行開始)
        self.future = self.executor.submit(self._prefetch_loop)
        return self

    def __next__(self):
        """ __iter__と併せてイテレータとして振る舞う仕組み """
        batch = self.queue.get()  # データが来るまで自動で待つ
        if batch is None:         # Noneは終了の印
            raise StopIteration   # forループの終了
        batch = self._formatting(batch)
        batch = np.asarray(batch) # ここではじめてConfig.npを使いGPU転送
        return batch

    def shutdown(self):
        """ 終了処理(スレッドを安全に止めるための後始末) """
        # 学習を途中で打ち切る場合などに使用
        self.stop_event.set()
        if self.future:
            self.future.cancel()
        self.executor.shutdown(wait=False)

    def normalization(self, x):
        """ -1to1の正規化をnumpyで実行(cfのそれはConfig.npで実行) """
        x_min = numpy.min(x); x_max = numpy.max(x)
        shift = (x_max + x_min)/2
        base  = (x_max - x_min)/2
        x = (x - shift) / base
        return x       
 

class CelebALoader(ImageLoader):

    def __init__(self, file_path, batch_size=64, prefetch=8, shuffle=True,
                 resize=False, transpose=False, normalize=False,
                 data_size=None):
        rawpath  = os.path.normpath(file_path + os.sep + "*")
        file_list = glob.glob(rawpath)
        if data_size is not None:
            file_list = file_list[:data_size]
        self.data_size = data_size
        super().__init__(file_list, batch_size, prefetch, shuffle,
                         resize, transpose, normalize)


if __name__=='__main__':
    
    path = r'D:\Python\img_align_celeba\img_align_celeba'
    loader = CelebALoader(
        path,
        resize=(48,64),
        transpose=True,
        normalize=True,
        )
    count = 0
    for x in loader:
        count += 1
        print(type(x), x.shape, x.dtype)
        if count > 100:
            break
