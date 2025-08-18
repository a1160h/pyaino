from pyaino.Config import *
from pyaino import Neuron

def conv1dref(x, kernel):
    input_length = len(x)
    kernel_length = len(kernel)
    output_length = input_length - kernel_length + 1
    y = np.zeros(output_length)

    for i in range(output_length):
        y[i] = np.sum(x[i:i+kernel_length] * kernel)

    return y

                   # C, Iw, M, Fw, stride, pad
#conv1d = Neuron.Conv1dLayer(1, 6,  1, 3,  1,      0)
conv1d = Neuron.Conv1dLayer(1, 3)

# テストデータとカーネルを定義
x = np.array([[[1, 2, 3, 4, 5, 6]]])
kernel = np.array([[0.5], [1], [0.5]])
conv1d.w = kernel
conv1d.b = np.array([0])
print(conv1d.w, conv1d.b)
# 畳み込みを実行
y = conv1d.forward(x)
gx = conv1d.backward(np.ones_like(y))

print("入力信号:", x)
print("カーネル:", kernel)
print("出力信号:", y)
print("入力の勾配:", gx)

deconv1d = Neuron.DeConv1dLayer(1, 3)
z = deconv1d.forward(y)
print("逆変換:", z)
gy = deconv1d.backward(np.ones_like(z))
print("逆変換の逆伝播:", gy)

print("\n簡易実装による比較")
y = conv1dref(x.reshape(-1), kernel.reshape(-1))

print("入力信号:", x)
print("カーネル:", kernel)
print("出力信号:", y)


