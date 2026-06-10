# data_loader
# 2026.06.08 A.Inoue

from pyaino.Config import *
#set_np('numpy'); np = Config.np
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
from PIL import Image, ImageOps
import glob, os, math, warnings
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
                 resize=None, source_order='HWC', target_order='asis',
                 cache=False, normalize=None, drop_last=False,
                 transform=None):
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
        self.target_order = f'B{target_order.upper()}' \
                            if target_order != 'asis' else 'asis'
        self.normalizer = cf.Normalize(normalize, np=numpy) \
                          if normalize is not None else None
        self.drop_last = drop_last
        self.transform = transform
        self.cache = cache
        self.cache_x, self.cache_y = None, None
        if self.cache:
            self.build_cache()

        if self.transform is not None and self.cache:
            warnings.warn("Cache=True and augmentation don't work well together.")

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

    def _batch_generator_bkup(self):
        indices = self._get_indices()
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self._load_item(idx) for idx in batch_indices]

            # (x, y) かどうか判定
            if isinstance(batch[0], tuple):
                xs, ys = zip(*batch)
                yield numpy.stack(xs), numpy.asarray(ys)
            else:
                yield numpy.stack(batch)


    def _batch_generator(self):
        indices = self._get_indices()

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            if self.cache_x is not None:
                if self.cache_y is None:
                    yield self.cache_x[batch_indices]
                else:
                    yield self.cache_x[batch_indices], self.cache_y[batch_indices]
                continue

            batch = [self._load_item(idx) for idx in batch_indices]

            if isinstance(batch[0], tuple):
                xs, ys = zip(*batch)
                xs = numpy.stack(xs)
                ys = numpy.asarray(ys)
                yield xs, ys
            else:
                yield numpy.stack(batch)

                

    def _batch_generator_bkup(self):
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

    def __next__bkup(self):
        batch = self.queue.get()
        if isinstance(batch, Exception):
            raise batch
        if batch is None:
            raise StopIteration

        # (x, y) 対応
        if isinstance(batch, tuple):
            x, y = batch
            x = self._formatting(x)
            x = np.asarray(x)
            y = np.asarray(y)
            return x, y
        else:
            batch = self._formatting(batch)
            batch = np.asarray(batch)
            return batch


    def __next__(self):
        batch = self.queue.get()
        if isinstance(batch, Exception):
            raise batch
        if batch is None:
            raise StopIteration

        if self.cache_x is not None:
            return batch

        if isinstance(batch, tuple):
            x, y = batch
            x = self._formatting(x)
            x = np.asarray(x)
            y = np.asarray(y)
            return x, y
        else:
            batch = self._formatting(batch)
            batch = np.asarray(batch)
            return batch


    def __next__bkup(self):
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

    def __getitem__bkup(self, idx):
        item = self._load_item(idx)

        if isinstance(item, tuple):
            x, y = item
            x = self._formatting(numpy.expand_dims(x, axis=0))[0]
            x = np.asarray(x)
            y = np.asarray(y)
            return x, y
        else:
            x = self._formatting(numpy.expand_dims(item, axis=0))[0]
            x = np.asarray(x)
            return x


    def __getitem__(self, idx):
        """
        与えられたidxに対応するデータを返す
        idxは環境依存のnpあるいは非依存のnumpyの両方に対応するが、内部の処理はnumpy

        """
        # --- cacheの場合 ---
        if self.cache_x is not None:
            if isinstance(idx, (int, numpy.integer, np.integer)):
                if self.cache_y is None:
                    return self.cache_x[int(idx)]
                else:
                    return self.cache_x[int(idx)], self.cache_y[int(idx)]

            elif isinstance(idx, slice):
                indices = list(range(*idx.indices(self.data_size)))
            elif isinstance(idx, (list, tuple)):
                indices = list(idx)
            elif isinstance(idx, (numpy.ndarray, np.ndarray)):
                indices = idx.tolist()
            else:
                raise TypeError(f"Unsupported index type: {type(idx)}")

            if self.cache_y is None:
                return self.cache_x[indices]
            else:
                return self.cache_x[indices], self.cache_y[indices]
       
        # --- 単一 index ---
        if isinstance(idx, (int, numpy.integer, np.integer)):
            item = self._load_item(int(idx))

            if isinstance(item, tuple):
                x, y = item
                x = self._formatting(numpy.expand_dims(x, axis=0))[0]
                x = np.asarray(x)
                y = np.asarray(y)
                return x, y
            else:
                x = self._formatting(numpy.expand_dims(item, axis=0))[0]
                x = np.asarray(x)
                return x

        # --- slice ---
        elif isinstance(idx, slice):
            indices = list(range(*idx.indices(self.data_size)))

        # --- list / tuple ---
        elif isinstance(idx, (list, tuple)):
            indices = list(idx)

        # --- ndarray ---
        elif isinstance(idx, (numpy.ndarray, np.ndarray)):
            indices = idx.tolist()

        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

        # --- 複数取得 ---
        batch = [self._load_item(i) for i in indices]

        # (x, y) 判定
        if isinstance(batch[0], tuple):
            xs, ys = zip(*batch)
            xs = numpy.stack(xs)
            ys = numpy.asarray(ys)

            xs = self._formatting(xs)
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            return xs, ys

        else:
            xs = numpy.stack(batch)
            xs = self._formatting(xs)
            xs = np.asarray(xs)
            return xs
   
    def __len__(self):
        return self.data_size

    def shutdown(self):
        self.stop_event.set()
        if self.future:
            self.future.cancel()
        self.executor.shutdown(wait=False)

    def build_cache(self):
        """
        全データを一度だけ読み込み、整形済み・正規化済みの形で保持する。
        以後の __iter__ / __getitem__ は _load_item() を呼ばず、
        cache_x / cache_y から取り出す。
        """
        batch = [self._load_item(i) for i in range(self.data_size)]

        if len(batch) == 0:
            raise ValueError("Cannot build cache: data_size is 0")

        if isinstance(batch[0], tuple):
            xs, ys = zip(*batch)
            xs = numpy.stack(xs)
            ys = numpy.asarray(ys)

            xs = self._formatting(xs)
            self.cache_x = np.asarray(xs)
            self.cache_y = np.asarray(ys)
        else:
            xs = numpy.stack(batch)
            xs = self._formatting(xs)

            self.cache_x = np.asarray(xs)
            self.cache_y = None


class CelebALoader(ImageLoader):
    """
    CelebA 画像ディレクトリ用ローダ

    data_source で指定されたディレクトリ内の画像ファイルを列挙し、
    index によってアクセスする。

    attr_source に anno/list_attr_celeba.txt を指定すると、
    画像と同時に属性ベクトルを返す。

    各画像はPIL.Imageで RGB (HWC) として読み込まれる。

    Parameters
    ----------
    data_source : str
        画像ディレクトリのパス
    attr_source : str or None
        CelebA の anno/list_attr_celeba.txt のパス
    """
    def __init__(self, data_source, attr_source=None, batch_size=64, prefetch=8, shuffle=True,
                 resize=None, target_order='asis', cache=False, normalize=None,
                 data_size=None, drop_last=False):

        if os.path.isdir(data_source):
            print(f"Get data from {data_source}")
        else:
            raise FileNotFoundError(f"フォルダが存在しません: {data_source}")

        rawpath = os.path.normpath(data_source + os.sep + "*")
        self.file_list = sorted(glob.glob(rawpath))

        self.attr_names = None
        self.attrs = None
        if attr_source is not None:
            self.attr_names, attr_dict = self._load_attr_file(attr_source)
            self.attrs = numpy.asarray(
                [attr_dict[os.path.basename(path)] for path in self.file_list],
                dtype=numpy.int8
            )
        
        if data_size is not None:
            self.file_list = self.file_list[:data_size]
            if self.attrs is not None:
                self.attrs = self.attrs[:data_size]
        self.data_size = len(self.file_list)    

        super().__init__(
            batch_size=batch_size,
            prefetch=prefetch,
            shuffle=shuffle,
            resize=resize,
            source_order='HWC',
            target_order=target_order,
            cache=cache,
            normalize=normalize,
            drop_last=drop_last,
        )

    def _load_attr_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()

        attr_names = lines[1].split()
        attr_dict = {}
        for line in lines[2:]:
            values = line.split()
            fname = values[0]
            attrs = numpy.asarray(values[1:], dtype=numpy.int8)
            attrs = (attrs > 0).astype(numpy.uint8)
            attr_dict[fname] = attrs

        return attr_names, attr_dict

    def _load_item(self, idx):
        file_path = self.file_list[idx]
        img = Image.open(file_path).convert('RGB')
        if self.resize is not None:
            img = ImageOps.fit(img, self.resize, Image.LANCZOS)
        x = numpy.asarray(img, dtype=numpy.uint8)

        if self.attrs is not None:
            y = self.attrs[idx]
            return x, y
        else:
            return x

    def label_names(self):
        return self.attr_names


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
    def __init__(self, data_source, label_source=None, batch_size=64, prefetch=8, shuffle=True,
                 resize=None, target_order='CHW', cache=False, normalize=None,
                 data_size=None, drop_last=False):

        self.data_source = data_source
        self.label_source = label_source
        self.axis_size = {'H': 96, 'W': 96, 'C': 3}
        self.dtype = numpy.uint8
        self.mmap_mode = 'r'
        source_order = 'CWH'

        self.source_shape = tuple(self.axis_size[axis] for axis in source_order)
        total_items = os.path.getsize(data_source) // numpy.dtype(self.dtype).itemsize
        self.data_size = total_items // math.prod(self.source_shape)

        if data_size is not None:
            self.data_size = min(self.data_size, data_size)

        if label_source is None:
            self.labels = None    
        else:
            self.labels = numpy.fromfile(label_source, dtype=numpy.uint8)
            self.labels = self.labels[:self.data_size]
        
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
            cache=cache,
            normalize=normalize,
            drop_last=drop_last,
        )

    def _load_item(self, idx):
        """
        idxが指すmemmapの画像1枚分を取出して、
        source_order の指定に従って多次元配列にして返す
        """
        x = self.data[idx]

        if self.resize is not None:
            x = cf.change_axis_order(x, self.source_order[1:], 'HWC')
            img = Image.fromarray(x)
            img = ImageOps.fit(img, self.resize, Image.LANCZOS)
            x = numpy.asarray(img, dtype=numpy.uint8)
            x = cf.change_axis_order(x, 'HWC', self.source_order[1:])

        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        else:
            return x

    def _load_item_bkup(self, idx):
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

    def label_names(self, file_path):
        with open(file_path, "r") as  f:
            names = f.read().splitlines() # 呼んでリスト形式に変換
        return names

class CIFAR10Loader(ImageLoader):
    """
    CIFAR-10 用ローダ

    data_source : CIFAR-10 のディレクトリ
        例: cifar-10-batches-py/

    train=True  -> data_batch_1〜5
    train=False -> test_batch
    """

    def __init__(self, data_source, train=True,
                 batch_size=64, prefetch=8, shuffle=True,
                 resize=None, target_order='CHW', cache=False, normalize=None,
                 data_size=None, drop_last=False,
                 transform=None):

        import pickle

        self.data = []
        self.labels = []

        if train:
            file_list = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            file_list = ["test_batch"]

        for fname in file_list:
            path = os.path.join(data_source, fname)
            with open(path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])     # (10000, 3072)
                self.labels.extend(entry['labels']) # list

        # concat
        self.data = numpy.concatenate(self.data, axis=0)
        self.labels = numpy.asarray(self.labels, dtype=numpy.uint8)

        # reshape to (N, C, H, W)
        self.data = self.data.reshape(-1, 3, 32, 32)

        self.data_size = len(self.data)

        if data_size is not None:
            self.data = self.data[:data_size]
            self.labels = self.labels[:data_size]
            self.data_size = data_size

        super().__init__(
            batch_size=batch_size,
            prefetch=prefetch,
            shuffle=shuffle,
            resize=resize,
            source_order='CHW',
            target_order=target_order,
            cache=cache,
            normalize=normalize,
            drop_last=drop_last,
            transform=transform, 
        )

    def _load_item(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.resize is None and self.transform is None:
            return x, y

        # 軸を入れ替えてPILimageに(CHW → HWC → PIL)
        x = cf.change_axis_order(x, 'CHW', 'HWC')
        img = Image.fromarray(x)

        if self.resize is not None:
            img = ImageOps.fit(img, self.resize, Image.LANCZOS)

        if self.transform is not None:
            img = self.transform(img)

        # numpyに戻し軸も元に戻す
        x = numpy.asarray(img, dtype=numpy.uint8)
        x = cf.change_axis_order(x, 'HWC', 'CHW')

        return x, y

    def label_names(self):
        names = ['airplane',   # 0
                 'automobile', # 1
                 'bird',       # 2
                 'cat',        # 3
                 'deer',       # 4
                 'dog',        # 5
                 'frog',       # 6
                 'horse',      # 7
                 'ship',       # 8
                 'truck']      # 9
        return names
    
class MNISTLoader(ImageLoader):
    """
    MNIST 用ローダ

    data_source : MNIST のディレクトリ
        例: MNIST/raw/ または ubyte ファイル群を含むディレクトリ

    train=True  -> train-images-idx3-ubyte / train-labels-idx1-ubyte
    train=False -> t10k-images-idx3-ubyte  / t10k-labels-idx1-ubyte
    """

    def __init__(self, data_source, train=True,
                 batch_size=64, prefetch=8, shuffle=True,
                 resize=None, target_order='CHW', normalize=None,
                 data_size=None, drop_last=False):

        import struct

        if train:
            image_file = 'train-images.idx3-ubyte'
            label_file = 'train-labels.idx1-ubyte'
        else:
            image_file = 't10k-images.idx3-ubyte'
            label_file = 't10k-labels.idx1-ubyte'

        image_path = os.path.join(data_source, image_file)
        label_path = os.path.join(data_source, label_file)

        with open(image_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError(f'Invalid MNIST image file: {image_path}')
            self.data = numpy.frombuffer(f.read(), dtype=numpy.uint8)
            self.data = self.data.reshape(num, 1, rows, cols)

        with open(label_path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError(f'Invalid MNIST label file: {label_path}')
            self.labels = numpy.frombuffer(f.read(), dtype=numpy.uint8)

        if len(self.data) != len(self.labels):
            raise ValueError('MNIST images and labels size mismatch')

        self.data_size = len(self.data)

        if data_size is not None:
            self.data = self.data[:data_size]
            self.labels = self.labels[:data_size]
            self.data_size = len(self.data)

        super().__init__(
            batch_size=batch_size,
            prefetch=prefetch,
            shuffle=shuffle,
            resize=resize,
            source_order='CHW',
            target_order=target_order,
            normalize=normalize,
            drop_last=drop_last,
        )

    def _load_item(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.resize is None:
            return x, y

        img = Image.fromarray(x[0], mode='L')
        img = ImageOps.fit(img, self.resize, Image.LANCZOS)
        x = numpy.asarray(img, dtype=numpy.uint8)
        x = numpy.expand_dims(x, axis=0)

        return x, y

    def label_names(self):
        return [str(i) for i in range(10)]

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
    cf.show_multi_samples(x) # celebAでも使える
    
    # STL10 の例
    data_source = r'D:\Python\STL10\stl10_binary\train_X.bin'
    label_source = r'D:\Python\STL10\stl10_binary\train_y.bin'
    label_names_file = r'D:\Python\STL10\stl10_binary\class_names.txt' 
   
    loader = STL10BinaryLoader(
        data_source,
        label_source,
        batch_size=50,
        resize=(48, 48),
        normalize=True,
        data_size=1000,
    )

    label_names = loader.label_names(label_names_file)

    count = 0
    for x, y in loader:
        count += 1
        print('STL10', type(x), x.shape, x.dtype, type(y), y.shape, y.dtype)
        if count > 2:
            break
    loader.shutdown()

    # STL10のlabelは0～9でなく1～10(0)
    cf.show_multi_samples(x, label_list=label_names, target=y-1)

    # CIFAR10
    data_source = r'C:\Python312\Lib\site-packages\cifar-10-batches-py'

    
    loader = CIFAR10Loader(
        data_source,
        train=True,
        batch_size=50,
        normalize=False
    )

    label_names = loader.label_names()

    count = 0
    for x, y in loader:
        count += 1
        print('CIFAR10', type(x), x.shape, x.dtype, type(y), y.shape, y.dtype)
        if count > 2:
            break
    loader.shutdown()
    
    cf.show_multi_samples(x, label_list=label_names, target=y)

    x, t = loader[123]
    l = label_names[int(t)]
    cf.show_sample(x, l, t)

    # MNIST
    data_source = r'C:\Python312\Lib\site-packages\MNIST'

    loader = MNISTLoader(
        data_source,
        train=True,
        batch_size=50,
        normalize=False
    )

    label_names = loader.label_names()

    count = 0
    for x, y in loader:
        count += 1
        print('MNIST', type(x), x.shape, x.dtype, type(y), y.shape, y.dtype)
        if count > 2:
            break
    loader.shutdown()

    cf.show_multi_samples(x, label_list=label_names, target=y)

