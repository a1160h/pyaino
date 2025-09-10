# Activators
# 2025.09.10 A.Inoue

from pyaino.Config import *
from pyaino.nucleus import Function
import copy

#### 活性化関数 ######################################################
class ActivatorBase(Function):
    def __init__(self, preserve_attr=False, **kwargs):
        """
         Neuron内で後に続く処理がinplace演算の場合に出力が書き換えられても
         self.yを壊されないようにpreserve_attr=Trueを指定

        """
        super().__init__(preserve_attr=preserve_attr)

    def forward(self, x, **kwargs):
        return super().forward(x)     # kwargsは無視する
    
    def update(self, *args, **kwargs): 
        pass                          # update()メソッドは何もしない 

class Identity(ActivatorBase):
    def __forward__(self, x):
        return x 
 
    def __backward__(self, gy): 
        return gy


class Step(ActivatorBase):
    def __init__(self, t=0):
        super().__init__()
        self.t = t
        
    def __forward__(self, x):
        y = x > self.t
        return y

    def __backward__(self, gy):
        raise Exception('__backward__ Method Not defined.')

def step(x):
    return Step()(x)

class Sigmoid(ActivatorBase):
    def __forward__(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def __backward__(self, gy):
        y = self.get_outputs()         
        gx = y * (1 - y) * gy
        return gx

def sigmoid(x):
    return Sigmoid().forward(x)

class SigmoidWithLoss(ActivatorBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.sumup = kwargs.pop('sumup', False)
        
    def __forward__(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def __backward__(self, t): # クロスエントロピー誤差との組合わせでの逆伝播(gyには正解値)
        y = self.get_outputs()         
        gx = (y - t)  
        return gx / len(t) if not self.sumup else gx

class SigmoidOut(ActivatorBase):
    def __init__(self, **kwargs):
        print('互換性のために維持、これは使わずに、y - t を外で作ってSigmoidを使用してください。')
        super().__init__()
        
    def __forward__(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def __backward__(self, t):
        y = self.get_outputs()         
        gx = (y - t) * y * (1 - y)
        return gx

class Tanh(ActivatorBase):
    def __forward__(self, x):
        y = np.tanh(x)
        return y

    def __backward__(self, gy):
        y = self.get_outputs()
        gx = gy * (1 - y * y)
        return gx

class ReLU(ActivatorBase):
    def __forward__(self, x):
        y = np.maximum(x, 0)
        return y
    
    def __backward__(self, gy):
        x, = self.inputs
        gx = gy * (x > 0)
        return gx
    
class ReLU_bkup(ActivatorBase):
    def __forward__(self, x):
        y = np.where(x<=0, 0, x)
        return y #.astype(Config.dtype)

    def __backward__(self, gy):
        x, = self.inputs
        gx = gy * np.where(x<=0, 0, 1)
        return gx #.astype(Config.dtype)

def relu(x):
    return ReLU().__forward__(x)

class LReLU(ActivatorBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.c = kwargs.pop('c', 0.01)
        
    def __forward__(self, x):
        y = np.maximum(x, 0) + np.minimum(x, 0) * self.c
        return y 

    def __backward__(self, gy):
        x, = self.inputs
        mask = x > 0
        gx = gy * (mask.astype(Config.dtype) + (~mask).astype(Config.dtype) * self.c)
        return gx
    
class LReLU_bkup(ActivatorBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.c = kwargs.pop('c', 0.01)
        
    def __forward__(self, x):
        y = np.where(x <= 0, self.c * x, x)
        return y #.astype(Config.dtype)

    def __backward__(self, gy):
        x, = self.inputs
        gx = gy * np.where(x<=0, self.c, 1)
        return gx #.astype(Config.dtype)

class ELU(ActivatorBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.c = kwargs.pop('c', 1.0)
        
    def __forward__(self, x):
        y = np.where(x<=0, self.c * (np.exp(x) - 1), x)
        return y #.astype(Config.dtype)

    def __backward__(self, gy):
        x, = self.inputs
        y = self.get_outputs()
        gx = gy * np.where(x<=0, (y + self.c), 1)
        return gx #.astype(Config.dtype)

class Swish(ActivatorBase):
    def __init__(self, eps=1e-7, **kwargs):
        super().__init__()
        self.beta = kwargs.pop('beta', 1.0)
        self.eps = eps
        
    def __forward__(self, x):
        beta_x = self.beta * x
        s = 1 / (1 + np.exp(-beta_x)) # sigmoid
        y = x * s
        return y

    def __backward__(self, gy):
        x, = self.inputs
        y  = self.get_outputs()
        gx = gy * (1 + self.beta * x - self.beta * y) * y / (x + self.eps)
        return gx

    def __backward__bkup(self, gy):
        s = self.s
        beta_x = self.beta_x
        gx = gy * (1 + beta_x - beta_x * s) * s
        return gx

def swish(x, beta=1.0):
    return Swish(beta).forward(x)

class Softplus(ActivatorBase):
    def __forward__(self, x):
        y = np.log(1 + np.exp(x))
        return y

    def __backward__(self, gy):
        x, = self.inputs         
        gx = gy / (1 + np.exp(-x))
        return gx

def softplus(x):
    return Softplus().forward(x)
    
class Mish(ActivatorBase):
    def __init__(self, eps=1e-7, **kwargs):
        super().__init__()
        self.eps = eps

    def __forward__(self, x):
        ts = np.tanh(np.log(1 + np.exp(x)))
        y = x * ts
        #self.ts = ts
        return y

    def __backward__(self, gy):
        x, = self.inputs
        y  = self.get_outputs()
        ts = y / (x + self.eps)
        gx = gy * (ts + (1 - np.square(ts)) * x / (1 + np.exp(-x))) 
        return gx

class GELU(Function):
    """ GELU "erf"(exact). """
    def __init__(self, eps=1e-7):
        super().__init__()
        # 関数erfを用意 
        try:        # cupy
            #raise Exception() # for debug 
            from np._cupyx.scipy.special import erf #as cupy_erf
            self.erf = erf
            print('Use cupyx.scipy.special for erf.')
        except:     # numpy
            try:    # scipy
                #raise Exception() # for debyg
                from scipy.special import erf
                self.erf = erf
                print('Use scipy for erf.')
            except: # math 
                try:
                    from math import erf
                    self.erf = np.vectorize(erf)
                    print('Use math for erf and is vectorized.')
                except:
                    raise Exception('No erf available on your computer.') 

        # 定数を用意
        self.c = np.array(np.sqrt(2.0 / np.pi), dtype=Config.dtype)   # √(2/π)
        self.inv_sqrt2 = np.array(1.0 / np.sqrt(2.0), dtype=Config.dtype)
        self.eps = eps

    def __forward__(self, x):
        z = self.erf(x * self.inv_sqrt2)
        y = 0.5 * x * (1.0 + z)
        #self.z = z
        return y

    def erf_backward(self, gy):
        x, = self.inputs
        return gy * (2.0 / np.sqrt(np.pi)) * np.exp(-x * x)

    def __backward__(self, gy):
        x, = self.inputs 
        y = self.get_outputs()
        z = 2.0 * (y / (x + self.eps)) - 1.0
        #z = self.z
        pdf = self.c * np.exp(-0.5 * x**2)
        dgelu_dx = 0.5 * (1.0 + z) + 0.5 * x * pdf
        return gy * dgelu_dx

class GELUap(Function):
    """ GELUap  "tanh" (Hendrycks & Gimpel approx) """
    def __init__(self, eps=1e-7):
        super().__init__()

        # 定数を dtype 付きで確定
        self.c = np.array(np.sqrt(2.0 / np.pi), dtype=Config.dtype)   # √(2/π)
        self.k = np.array(0.044715, dtype=Config.dtype)
        self.eps = eps

    def __forward__(self, x):
        u = self.c * (x + self.k * x**3)
        t = np.tanh(u)
        y = 0.5 * x * (1.0 + t)
        #self.t = t
        return y

    def __backward__(self, gy):
        x, = self.inputs
        y = self.get_outputs()
        t =  2.0 * (y / (x + self.eps)) - 1.0 
        #t  = self.t
        du_dx = self.c * (1.0 + 3.0 * self.k * x**2)
        dt_dx = (1.0 - t**2) * du_dx
        dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * dt_dx
        return gy * dgelu_dx

class Softmax(ActivatorBase):
    def __init__(self, temperature=1.0, **kwargs):
        super().__init__()
        self.temperature = temperature

    def __forward__(self, x):
        x = x / self.temperature   # 温度スケーリング
        max_x = np.max(x, axis=-1, keepdims=True) #if dimx>1 else np.max(x)
        exp_a = np.exp(x - max_x)  # オーバーフロー対策
        sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True) #if dimx>1 else np.sum(exp_a) 
        y = exp_a / (sum_exp_a + 1e-7)
        return y

    def __backward__(self, gy): # ソフトマックス本来の逆伝播
        y = self.get_outputs()
        gx = y * gy
        sumdx = np.sum(gx, axis=-1, keepdims=True)
        gx -= y * sumdx
        gx = gx / self.temperature # 温度スケーリング
        return gx

class Softmax2(ActivatorBase):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def __forward__(self, x):
        """Softmaxの順伝播"""
        x = x / self.temperature  # 温度スケーリング
        x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))  # オーバーフロー防止
        y = x_exp / np.sum(x_exp, axis=-1, keepdims=True)
        return y
    
    def __backward__(self, gy):
        """Softmaxの逆伝播"""
        y = self.get_outputs()
        batch_size, num_classes = y.shape
        dx = np.empty_like(gy)
        
        for i in range(batch_size):
            # Softmax のヤコビアン行列
            z = y[i].reshape(-1, 1)
            jacobian = np.diagflat(z) - np.dot(z, z.T)
            
            # 逆伝播の計算
            dx[i] = np.dot(jacobian, gy[i])
        return dx / self.temperature  # 温度の影響を考慮

class SoftmaxWithLoss(ActivatorBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.sumup = kwargs.pop('sumup', False)
        
    def __forward__(self, x):
        y = x - x.max(axis=-1, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=-1, keepdims=True)
        self.x_shape = x.shape  # B,T,V=x.shape
        return y
        
    def __backward__(self, t):
        y = self.get_outputs()
        vr = y.shape[-1]         # 値幅(出力ニューロン数)
        if t.shape == y.shape:   # tが出力と同形、即ちone-hotベクトルの場合
            t = t.argmax(axis=-1)
        else:
            t = np.array(t, dtype=int)
        #N = y.size // vr        # N=B*T
        dx = copy.deepcopy(y)    
        dx = dx.reshape(-1, vr)       # reshapeしても元の変数を参照
        t = t.reshape(-1)
        dx[np.arange(len(t)), t] -= 1 # tの指す所を-1 yも更新されることに注意　
        dx = dx.reshape(*self.x_shape)
        return dx / len(t) if not self.sumup else dx

class SoftmaxWithLossMasked(ActivatorBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.ignore_label = kwargs.pop('ignore',    -1)
        self.sumup        = kwargs.pop('sumup',  False)
        
    def __forward__(self, x):
        y = x - x.max(axis=-1, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=-1, keepdims=True)
        self.x_shape = x.shape  # B,T,V=x.shape
        return y

    def __backward__(self, t):
        y = self.get_outputs()
        vr = y.shape[-1]        # 値幅(出力ニューロン数)
        if t.shape == y.shape:  # tが出力と同形、即ちone-hotベクトルの場合
            t = t.argmax(axis=-1)
        else:
            t = np.array(t, dtype=int)
        #N = y.size // vr        # N=B*T
        dx = copy.deepcopy(y)    
        dx = dx.reshape(-1, vr)
        t = t.reshape(-1)
        mask = (t != self.ignore_label).reshape(-1)
        dx[np.arange(len(t)), t] -= 1     # tの指す所を-1
        if not self.sumup:
            dx /= mask.sum()         # 評価数(バッチ数の代わり)
        dx *= mask[:, np.newaxis]    # ignore_labelに該当するデータは勾配を0にする
        dx = dx.reshape(*self.x_shape)
        return dx
    
class SoftmaxWithLoss2(ActivatorBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.sumup   = kwargs.pop('sumup', False)

    def __forward__(self, x):
        #dimx = x.ndim
        max_x = np.max(x, axis=-1, keepdims=True) #if dimx>1 else np.max(x)
        exp_a = np.exp(x - max_x)  # オーバーフロー対策
        sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True) #if dimx>1 else np.sum(exp_a) 
        y = exp_a / (sum_exp_a + 1e-7)
         
        return y

    def __backward__(self, t): # クロスエントロピー誤差との組合わせでの逆伝播(gyには正解値)
        y = self.get_outputs()
        if t.shape != y.shape:  # tがone-hotベクトルでない場合
            t = cf.convert_one_hot(t, y.shape[-1])
        y = self.get_outputs()
        gx = y - t
        return gx / len(t) if not self.sumup else gx


if __name__=='__main__':
    import matplotlib.pyplot as plt

    Funcs = [Identity, Step, Sigmoid, Tanh, ReLU, LReLU, ELU, Softmax]
    Funcs += [Swish, Softplus, Mish, GELU, GELUap]
    Funcs += [SigmoidOut, SigmoidWithLoss, SoftmaxWithLoss, SoftmaxWithLossMasked] 

    x = np.linspace(-5, 5, 100) # 値の範囲を指定

    for Func in Funcs:
        func = Func()
        y = func.forward(x)
        plt.plot(x.tolist(), y.tolist())
        gy = np.ones_like(y)
        try:
            gx = func.backward(gy)
        except:
            print('backward failed', func.__class__.__name__)
            pass
        else:    
            plt.plot(x.tolist(), gx.tolist())
        plt.title(func.__class__.__name__)
        plt.show()


    
