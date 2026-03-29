from pyaino.Config import *
#set_np('numpy'); np=Config.np
from pyaino import Neuron
import matplotlib.pyplot as plt
from pyaino import common_function as cf

def show_image(x, title=None):
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
    plt.title(title)
    plt.show()

x = np.random.randn(1, 3, 4, 4).astype(np.float32)
show_image(x, 'x')

# 最近傍（倍率指定）
model = Neuron.Interpolate2d(scale_factor=2)
y = model(x)
title = model.impl.__class__.__name__
show_image(y, title)

gx = model.backward()
show_image(gx)

gxr = cf.numerical_gradient(model, x)
print("nearest grad check:", np.allclose(gx, gxr))

# 最近傍（サイズ指定）
model = Neuron.Interpolate2d(size=(5, 10))#scale_factor=2)
y = model(x)
title = model.impl.__class__.__name__
show_image(y, title)

gx = model.backward()
show_image(gx)

gxr = cf.numerical_gradient(model, x)
print("nearest grad check:", np.allclose(gx, gxr))

# bilinear で 1.5 倍にしてみる
model = Neuron.Interpolate2d(scale_factor=1.50,
                                       #mode="nearest",
                                       mode="bilinear",
                                       align_corners=True,
                                       #align_corners=False,
                                       )
y = model(x)
title = model.impl.__class__.__name__
show_image(y, title)

gx = model.backward()
show_image(gx)

# 数値微分と比較
gxr = cf.numerical_gradient(model, x)
print("bilinear grad check:", np.allclose(gx, gxr))#, atol=1e-4, rtol=1e-3))
