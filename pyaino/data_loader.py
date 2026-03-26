# data_loader
# 2026.03.26 A.Inoue

from pyaino.Config import *
#set_np('numpy'); np = Config.np
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
from PIL import Image, ImageOps
import glob, os, math
import numpy
from pyaino import common_function as cf

class ImageLoader:
    """
    汎用画像データローダ（基底クラス）

    データは index によりアクセスされ、各 index に対して
    _load_item(idx) が 1件のデータを返すことを前提とする。

    データの実体（データ総数、ファイル、バイナリ、メモリ等）は派生クラス側で管理

    Parameters
    ----------
    batch_size : int
        バッチサイズ
    prefetch : int
        先読みバッファサイズ
    shuffle : bool
        シャッフルするかどうか
    resize : tuple or None
        (H, W) のリサイズ指定
    source_order : str
        入力データの軸順（Bを除く、例: 'HWC', 'CWH'）
    target_order : str
        出力時の軸順（'asis' または 'CHW' など）
    normalize : tuple or None
        正規化設定
    """
    def __init__(self, batch_size=32, prefetch=4, shuffle=True,
                 resize=None, source_order='HWC', target_order='asis', normalize=None):
        #self.data_size = data_size
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.shuffle = shuffle
        self.queue = Queue(maxsize=prefetch)
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None

        self.resize = resize
        self.source_order = f'B{source_order.upper()}'
        self.target_order = f'B{target_order.upper()}' if target_order != 'asis' else 'asis'
        self.normalizer = cf.Normalize(normalize) if normalize is not None else None

    def _load_item(self, idx):
        """
        index に対応する1件のデータを返す

        Parameters
        ----------
        idx : int データインデックス

        Returns 
        -------
        ndarray : 画像1枚分（B軸なし）
        """
        raise NotImplementedError

    def _get_indices(self):
        indices = numpy.arange(self.data_size)
        if self.shuffle:
            numpy.random.shuffle(indices)
        return indices

    def _batch_generator(self):
        indices = self._get_indices()
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self._load_item(idx) for idx in batch_indices]
            yield numpy.stack(batch)
            
    def _formatting(self, x):
        """
        cpu上(numpy)でデータの整形
        self.source_order は x の現在の軸順
        """
        if self.target_order != 'asis':
            x = cf.change_axis_order(x, self.source_order, self.target_order)

        if self.normalizer is not None:
            x = x.astype(getattr(numpy, Config.dtype), copy=False)
            x = self.normalizer(x)

        return x

    def _prefetch_loop(self):
        """ 先読みの本体 """
        try:
            for batch in self._batch_generator():
                if self.stop_event.is_set():
                    break
                self.queue.put(batch)
        except Exception as e:
            self.queue.put(e)
        finally:
            self.queue.put(None)

    def __next__(self):
        batch = self.queue.get()
        if isinstance(batch, Exception):
            raise batch
        if batch is None:
            raise StopIteration
        batch = self._formatting(batch)
        batch = np.asarray(batch)
        return batch

    def __iter__(self):
        self.stop_event.clear()
        self.future = self.executor.submit(self._prefetch_loop)
        return self

    def shutdown(self):
        self.stop_event.set()
        if self.future:
            self.future.cancel()
        self.executor.shutdown(wait=False)


class CelebALoader(ImageLoader):
    """
    CelebA 画像ディレクトリ用ローダ

    data_source で指定されたディレクトリ内の画像ファイルを列挙し、
    index によってアクセスする。

    各画像はPIL.Imageで RGB (HWC) として読み込まれる。

    Parameters
    ----------
    data_source : str 画像ディレクトリのパス
    """
    def __init__(self, data_source, batch_size=64, prefetch=8, shuffle=True,
                 resize=None, target_order='asis', normalize=None, data_size=None):

        rawpath = os.path.normpath(data_source + os.sep + "*")
        self.file_list = glob.glob(rawpath)
        
        if data_size is not None:
            self.file_list = self.file_list[:data_size]
        self.data_size = len(self.file_list)    

        super().__init__(
            batch_size=batch_size,
            prefetch=prefetch,
            shuffle=shuffle,
            resize=resize,
            source_order='HWC',
            target_order=target_order,
            normalize=normalize,
        )

    def _load_item(self, idx):
        file_path = self.file_list[idx]
        img = Image.open(file_path).convert('RGB')
        if self.resize is not None:
            img = ImageOps.fit(img, self.resize, Image.LANCZOS)
        return numpy.asarray(img, dtype=numpy.uint8)


class STL10BinaryLoader(ImageLoader):
    """
    STL10 のバイナリデータ用ローダ

    data_source は複数画像が連続して格納された .bin ファイル
    CWH 形式で格納されているため、source_order='CWH' として扱う。

    Parameters
    ----------
    data_source : str STL10 の .bin ファイルパス
    target_order : str 出力時の軸順（例: 'CHW'）
    """
    def __init__(self, data_source, batch_size=64, prefetch=8, shuffle=True,
                 resize=None, target_order='CHW', normalize=None,
                 data_size=None):

        self.data_source = data_source
        self.axis_size = {'H': 96, 'W': 96, 'C': 3}
        self.dtype = numpy.uint8
        self.mmap_mode = 'r'
        source_order = 'CWH'

        self.source_shape = tuple(self.axis_size[axis] for axis in source_order)
        total_items = os.path.getsize(data_source) // numpy.dtype(self.dtype).itemsize
        self.data_size = total_items // math.prod(self.source_shape)

        if data_size is not None:
            self.data_size = min(self.data_size, data_size)
        
        # 生データをmemmapで持つ(データ形状の通り)
        self.data = numpy.memmap(
            data_source,
            dtype=self.dtype,
            mode=self.mmap_mode,
            shape=(self.data_size, *self.source_shape)
        )
        
        super().__init__(
            batch_size=batch_size,
            prefetch=prefetch,
            shuffle=shuffle,
            resize=resize,
            source_order=source_order,
            target_order=target_order,
            normalize=normalize,
        )

    def _load_item(self, idx):
        """
        idxが指すmemmapの画像1枚分を取出して、
        source_order の指定に従って多次元配列にして返す
        """
        x = self.data[idx]
        if self.resize is None:
            return x

        # PIL は HWC 前提なので、いったん HWC にしてから resize
        x = cf.change_axis_order(x, self.source_order[1:], 'HWC')
        img = Image.fromarray(x)
        img = ImageOps.fit(img, self.resize, Image.LANCZOS)
        x = numpy.asarray(img, dtype=numpy.uint8)

        # 元の画像軸順に戻す
        x = cf.change_axis_order(x, 'HWC', self.source_order[1:])
        return x


if __name__ == '__main__':
    from pyaino import STL10

    # CelebA の例
    data_source = r'D:\Python\img_align_celeba\img_align_celeba'
    loader = CelebALoader(
        data_source,
        batch_size=50,
        resize=(48, 64),
        target_order='CHW',
        normalize=True,
        data_size=1000,
    )
    count = 0
    for x in loader:
        count += 1
        print('CelebA', type(x), x.shape, x.dtype)
        if count > 2:
            break
    loader.shutdown()
    STL10.show_multi_samples(x) # celebAでも使える
    
    # STL10 の例
    data_source = r'D:\Python\STL10\stl10_binary\train_X.bin'
    loader = STL10BinaryLoader(
        data_source,
        batch_size=50,
        resize=(48, 48),
        normalize=True,
        data_size=1000,
    )
    count = 0
    for x in loader:
        count += 1
        print('STL10', type(x), x.shape, x.dtype)
        if count > 2:
            break
    loader.shutdown()

    #from pyaino import STL10
    STL10.show_multi_samples(x)
