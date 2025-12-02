from pyaino.Config import *
#set_np('numpy'); np=Config.np
from pyaino import Neuron
import matplotlib.pyplot as plt
from pyaino import common_function as cf

def show_image(x):
    if   x.ndim == 2:
        x = x.reshape(1, *x.shape)
    elif x.ndim == 3:
        pass
    elif x.ndim == 4:
        x = x[0]
    x_max = np.max(x)
    x_min = np.min(x)
    img = (x - x_min) / np.clip((x_max - x_min), 1e-7, None)
    plt.imshow(img.transpose(-2,-1,-3).tolist())
    plt.show()

x = np.random.randn(1, 3, 4, 4).astype(np.float32)
show_image(x)

# 最近傍（2倍）
#interpolate = Neuron.Interpolate2d(scale_factor=2)
#interpolate = Neuron.Interpolate2dNearestSimple(scale_factor=2)
interpolate = Neuron.Interpolate2dNearestGeneral(size=(5, 10))#scale_factor=2)
y = interpolate(x)
show_image(y)

gx = interpolate.backward()
show_image(gx)

# 数値微分と比較
gxr = cf.numerical_gradient(interpolate, x)
print("nearest grad check:", np.allclose(gx, gxr))

# bilinear で 1.5 倍にしてみる
interp_bilinear = Neuron.Interpolate2d(scale_factor=1.5, mode="bilinear")
y = interp_bilinear(x)
show_image(y)

gx = interp_bilinear.backward()
show_image(gx)

# 数値微分と比較
gxr = cf.numerical_gradient(interp_bilinear, x)
print("bilinear grad check:", np.allclose(gx, gxr))#, atol=1e-4, rtol=1e-3))
