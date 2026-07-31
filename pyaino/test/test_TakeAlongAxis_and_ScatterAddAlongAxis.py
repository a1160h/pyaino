from pyaino.Config import *
from pyaino import Functions as F


###### TakeAlongAxisのテスト
x = np.array(
    [[10, 11, 12, 13],
     [20, 21, 22, 23]],
    dtype=Config.dtype,
)
indices = np.array(
    [[3, 1],
     [0, 2]],
    dtype=np.int32,
)
model = F.TakeAlongAxis(indices, axis=1)

y = model(x)
gy = np.array([[1, 2], [3, 4]], dtype=Config.dtype)
gx = model.backward(gy)

print("\n----- TakeAlongAxis：通常のindex -----")
print("x =\n", x)
print("indices =\n", model.indices)
print("y（indicesの位置から取り出す）=\n", y)
print("gy =\n", gy)
print("gx（gyを元の位置へ加算する）=\n", gx)


# 同じ位置を複数回取り出す。
# backwardでは、その位置へ勾配が加算される。
model.indices = np.array(
    [[1, 1],
     [2, 2]],
    dtype=np.int32,
)

y = model(x)
gy = np.array([[1, 2], [3, 4]], dtype=Config.dtype)
gx = model.backward(gy)

print("\n----- TakeAlongAxis：重複するindex -----")
print("x =\n", x)
print("indices =\n", model.indices)
print("y（同じ位置を複数回取り出す）=\n", y)
print("gy =\n", gy)
print("gx（同じ位置の勾配を加算する）=\n", gx)


###### ScatterAddAlongAxisのテスト
x = np.array(
    [[10, 11],
     [20, 21]],
    dtype=Config.dtype,
)
indices = np.array(
    [[3, 1],
     [0, 2]],
    dtype=np.int32,
)
model = F.ScatterAddAlongAxis(
    indices,
    output_shape=(2, 4),
    axis=1,
)

y = model(x)
gy = np.array(
    [[1, 2, 3, 4],
     [5, 6, 7, 8]],
    dtype=Config.dtype,
)
gx = model.backward(gy)

print("\n----- ScatterAddAlongAxis：通常のindex -----")
print("x =\n", x)
print("indices =\n", model.indices)
print("y（indicesの位置へ加算配置する）=\n", y)
print("gy =\n", gy)
print("gx（indicesの位置から取り出す）=\n", gx)


# 複数の値を同じ位置へ配置する。
# forwardでは、その位置へ値が加算される。
model.indices = np.array(
    [[1, 1],
     [2, 2]],
    dtype=np.int32,
)

y = model(x)
gy = np.array(
    [[1, 2, 3, 4],
     [5, 6, 7, 8]],
    dtype=Config.dtype,
)
gx = model.backward(gy)

print("\n----- ScatterAddAlongAxis：重複するindex -----")
print("x =\n", x)
print("indices =\n", model.indices)
print("y（同じ位置へ値を加算する）=\n", y)
print("gy =\n", gy)
print("gx（同じ位置の勾配をそれぞれ取り出す）=\n", gx)
