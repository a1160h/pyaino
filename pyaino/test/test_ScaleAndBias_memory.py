# inplace演算の確認
from pyaino.Config import *
from pyaino import Neuron
import sys

def get_total_size(obj, seen=None):
    """オブジェクトとその内部要素のメモリサイズを再帰的に計算"""
    
    # 計算済みのオブジェクトを記録する（循環参照対策）
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:  # すでに計算済みならスキップ
        return 0
    seen.add(obj_id)
    
    # 基本サイズ
    size = sys.getsizeof(obj)

    # 辞書型の場合（キーと値のサイズも加算）
    if isinstance(obj, dict):
        #print('### dict ###')
        for key, value in obj.items():
            size += get_total_size(key, seen)
            size += get_total_size(value, seen)
    
    # リスト、タプル、セットなどの場合（要素のサイズを加算）
    elif isinstance(obj, (list, tuple, set)):
        #print('### tuple ###')
        for item in obj:
            size += get_total_size(item, seen)

    elif isinstance(obj, np.ndarray):# and obj.ndim>=0:
        #print('### ndarray ###')
        n = 0
        for item in obj.reshape(-1):
            size += sys.getsizeof(item)
            n += 1
    
    # クラスのインスタンス（`__dict__` の中身を加算）
    elif hasattr(obj, '__dict__'):
        #print('### class ###')
        size += get_total_size(obj.__dict__, seen)
    
    return size

x = np.arange(1000000, dtype='f8').reshape(-1, 1000)
print(x.shape)

#tracemalloc.start()

func = Neuron.ScaleAndBias(axis=-1)
y = func(x)
print('x :', id(x), 'y :', id(y))  # xと別
func.backward()

print('func :', get_total_size(func)) 
print('x    :', get_total_size(x)) 
print('y    :', get_total_size(y)) 

print('total:', get_total_size([func, x, y])) # xとyはダブルカウントしているのが分る 
