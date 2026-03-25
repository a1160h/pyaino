from pyaino.Config import *
#set_np('numpy'); np = Config.np
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
from PIL import Image, ImageOps
import glob, os
import numpy
from pyaino import common_function as cf

class ImageLoader:
    """
    先読み機能付きデータローダ

    役割分担:
      - _load_item()     : 1件を numpy 配列として読む
      - _formatting()    : バッチ単位の軸変換・正規化
      - __next__()       : 最後に Config.np へ変換（必要ならGPU転送）
    """
    def __init__(self, item_list, batch_size=32, prefetch=4, shuffle=True,
                 resize=None, source_order='HWC', target_order='asis', normalize=None):
        self.item_list = item_list
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
        if normalize is not None:  
            self.normalizer = cf.Normalize(normalize)
        else:
            self.normalizer = None

    def _load_item(self, path):
        """
        画像ファイル1件を読んで大きさを調整して numpy 配列にして返す。
        画像ファイルでない場合には派生クラスで上書きする前提。
        """
        img = Image.open(path).convert('RGB')
        if self.resize is not None:
            img = ImageOps.fit(img, self.resize, Image.LANCZOS)
        x = numpy.asarray(img, dtype=numpy.uint8)
        return x

    def _batch_generator(self):
        """ バッチを生成するジェネレータ """
        items = self.item_list.copy()
        if self.shuffle:
            numpy.random.shuffle(items)

        for i in range(0, len(items), self.batch_size):
            batch_items = items[i:i + self.batch_size]
            batch = [self._load_item(item) for item in batch_items]
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


class BinaryImageLoader(ImageLoader):
    """
    1つの .bin に複数画像が入っている場合のローダ

    例:
      source_order = 'CWH'
      axis_size    = {'H':96, 'W':96, 'C':3}

    読み出し直後の軸順は source_order の通り。
    軸変換は _formatting() 側で target_order に合わせて行う。
    """
    def __init__(self, path, batch_size=64, prefetch=8, shuffle=True,
                 resize=None, source_order='HWC', target_order='asis',
                 normalize=None,
                 data_size=None,
                 axis_size=None,
                 dtype=numpy.uint8,
                 mmap_mode='r'):

        if axis_size is None:
            raise ValueError("axis_size を指定してください。例: {'H':96, 'W':96, 'C':3}")

        self.path = path
        self.axis_size = dict(axis_size)
        self.dtype = dtype
        self.mmap_mode = mmap_mode

        self.items_per_image = 1
        for axis in source_order:
            self.items_per_image *= self.axis_size[axis]

        total_items = os.path.getsize(path) // numpy.dtype(dtype).itemsize
        total_images = total_items // self.items_per_image

        if data_size is not None:
            total_images = min(total_images, data_size)

        self.data_size = total_images

        # 生データをmemmapで持つ
        self.data = numpy.memmap(
            path,
            dtype=self.dtype,
            mode=self.mmap_mode,
            shape=(self.data_size, self.items_per_image)
        )

        super().__init__(
            item_list=list(range(self.data_size)), #self.item_list,
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
        memmapでもつデータのidxが指すflat画像1枚分 を取出して、
        source_order の指定に従って多次元配列にして返す
        """
        flat = self.data[idx]

        stored_shape = tuple(self.axis_size[axis] for axis in self.source_order[1:])
        x = flat.reshape(stored_shape)
        if self.resize is None:
            return x

        # PIL は HWC 前提なので、いったん HWC にしてから resize
        hwc = cf.change_axis_order(x[numpy.newaxis, ...], self.source_order, 'BHWC')[0]
        img = Image.fromarray(hwc)
        img = ImageOps.fit(img, self.resize, Image.LANCZOS)
        hwc = numpy.asarray(img, dtype=numpy.uint8)

        # 元の画像軸順に戻す
        x = cf.change_axis_order(hwc[numpy.newaxis, ...], 'BHWC', self.source_order)[0]
        return x


class CelebALoader(ImageLoader):
    """
    通常の画像ファイル群用ローダ
    読み出し直後は HWC
    """
    def __init__(self, file_path, batch_size=64, prefetch=8, shuffle=True,
                 resize=None, target_order='asis', normalize=None,
                 data_size=None):
        rawpath = os.path.normpath(file_path + os.sep + "*")
        file_list = glob.glob(rawpath)
        if data_size is not None:
            file_list = file_list[:data_size]

        self.data_size = len(file_list)

        super().__init__(
            item_list=file_list,
            batch_size=batch_size,
            prefetch=prefetch,
            shuffle=shuffle,
            resize=resize,
            source_order='HWC', 
            target_order=target_order,
            normalize=normalize,
        )

class STL10BinaryLoader(BinaryImageLoader):
    """
    STL10 用
    STL10 の bin は実質 CWH とみなす
    """
    def __init__(self, path, batch_size=64, prefetch=8, shuffle=True,
                 resize=None, source_order='CWH', target_order='CHW', normalize=None,
                 data_size=None):
        super().__init__(
            path=path,
            batch_size=batch_size,
            prefetch=prefetch,
            shuffle=shuffle,
            resize=resize,
            source_order=source_order,
            target_order=target_order,
            normalize=normalize,
            data_size=data_size,
            axis_size={'H': 96, 'W': 96, 'C': 3},
            dtype=numpy.uint8,
        )


if __name__ == '__main__':
    from pyaino import STL10

    # CelebA の例
    path = r'D:\Python\img_align_celeba\img_align_celeba'
    loader = CelebALoader(
        path,
        batch_size=50,
        resize=(48, 64),
        target_order='CHW',
        normalize=True,
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
    path = r'D:\Python\STL10\stl10_binary\train_X.bin'
    loader = STL10BinaryLoader(
        path,
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
