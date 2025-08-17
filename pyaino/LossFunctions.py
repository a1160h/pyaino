# LossFunctions
# 2025.08.17 A.Inoue
from pyaino.Config import *
from pyaino import nucleus
from pyaino import Functions as F
import warnings

class LossFunctionBase(nucleus.Function):
    def __init__(self, axis=None, **kwargs):
        super().__init__()
        self.t = None
        self.k = None
        self.axis = axis # 指定した軸について損失を合算する
        pass

    def __forward__(self, y, t):
        self.t = t # 従来互換のため
        self.k = y.size if self.axis is None else y.size//y.shape[self.axis]
        l = self._forward(y, t)
        return l / self.k

    def __backward__(self, gl=1):
        y, t = self.inputs
        gy = self._backward(gl, y, t)
        gy /= self.k
        return gy.astype(Config.dtype)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class MeanSquaredError(LossFunctionBase):
    def __init__(self, axis=None, **kwargs):
        super().__init__(axis=axis, **kwargs)

    def _forward(self, y, t):
        return 0.5 * np.sum((y - t) ** 2)

    def _backward(self, gl, y, t):
        return gl * (y - t)

class CrossEntropyError(LossFunctionBase):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(axis=axis, **kwargs)

    def _forward(self, y, t):
        return - np.sum(t * np.log(y + 1e-7))        # メモリ使用量削減のため演算順序拘束

    def _backward(self, gl, y, t):
        return (- gl) * (t / (y + 1e-7))

class CrossEntropyError2(LossFunctionBase):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(axis=axis, **kwargs)

    def _forward(self, y, t):
        return -np.sum(t*np.log(y+1e-7)+(1-t)*np.log(1-y+1e-7))

    def _backward(self, gl, y, t):
        return gl * ((y - t) / (y * (1 - y) + 1e-7)) # メモリ使用量削減のため演算順序拘束

class CrossEntropyErrorMasked(LossFunctionBase):
    def __init__(self, axis=-1, ignore=None):
        super().__init__(axis=axis)
        self.ignore_label = ignore
        self.valid_mask = None
        self.valid_rate = None

    def _forward(self, y, t):
        """ y:probabilities確率、t:正解値ラベルから交差エントロピーを求める """
        if y.shape==t.shape: # tがone_hotの場合
            t = np.argmax(t, axis=self.axis)
            warnings.warn('Given target is one_hot. Target category recommended.')
        if y.ndim > 2:
            y = y.reshape(-1, y.shape[-1])
            t = t.reshape(-1)

        if self.ignore_label is not None:
            self.valid_mask = (t != self.ignore_label)
        else:
            self.valid_mask = np.ones_like(t, dtype=bool)

        valid_indices = np.where(self.valid_mask)[0]
        log_y = np.log(y[valid_indices, t[valid_indices]])
        l = -np.sum(log_y)
        valid_rate = self.valid_mask.sum() / self.valid_mask.size # 有効部分の比率
        self.valid_rate = valid_rate
        return l / valid_rate

    def _backward(self, gl, y, t):
        """ y:probabilitiesに対する勾配gyを返す """
        y_shape = y.shape
        if y.shape==t.shape: # tがone_hotの場合
            t = np.argmax(t, axis=self.axis)
        if y.ndim > 2:
            y = y.reshape(-1, y.shape[-1])
            t = t.reshape(-1)
        gy = np.zeros_like(y)
        gy[range(len(t)), t] = -1 / y[range(len(t)), t]
        gy[~self.valid_mask] = 0
        return gy.reshape(*y_shape) / self.valid_rate

class CrossEntropyErrorForLogits(LossFunctionBase):
    def __init__(self, axis=-1, ignore=None):
        super().__init__(axis=axis)
        self.ignore = ignore
        self.log_probs = None
        self.t = None
        self.mask = None
        self.safe_t = None
        self.valid_rate = None

    def _forward(self, y, t):
        if y.shape==t.shape: # tがone_hotの場合
            t = np.argmax(t, axis=self.axis)
            warnings.warn('Given target is one_hot. Target category recommended.')
        if y.ndim > 2:
            y = y.reshape(-1, y.shape[-1])
            t = t.reshape(-1)
        B, num_classes = y.shape

        max_y = np.max(y, axis=1, keepdims=True)
        y_stable = y - max_y
        logsumexp = np.log(np.sum(np.exp(y_stable), axis=1, keepdims=True)) + max_y
        self.log_probs = y - logsumexp  # log-softmax

        if self.ignore is not None:
            self.mask = (t != self.ignore).astype(np.float32)
            self.safe_t = np.where(t == self.ignore, 0, t)
        else:
            self.mask = np.ones_like(t, dtype=np.float32)
            self.safe_t = t

        nll = -self.log_probs[np.arange(B), self.safe_t] * self.mask
        loss = np.sum(nll) 
        self.valid_rate = np.sum(self.mask) / self.mask.size

        return loss / self.valid_rate

    def _backward(self, gl, y, t):
        y_shape = y.shape
        if y.shape==t.shape: # tがone_hotの場合
            t = np.argmax(t, axis=self.axis)
        if y.ndim > 2:
            y = y.reshape(-1, y.shape[-1])
            t = t.reshape(-1)
        grad = np.exp(self.log_probs)    # softmax 相当
        grad[range(len(t)), self.safe_t] -= 1.0
        grad *= self.mask[:, np.newaxis] # ignore ラベルの勾配を 0 に

        return grad.reshape(*y_shape) / self.valid_rate

class CrossEntropyErrorMasked_bkup(LossFunctionBase):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(axis=axis, **kwargs)
        self.ignore_label = kwargs.pop('ignore', -1)

    def _forward(self, y, t):
        vr = y.shape[-1]                  # 値幅(one_hotの幅)
        if t.ndim < y.ndim:
            t = np.eye(vr, dtype=bool)[t] # yの形状に合わせる(形状変化対応)
        y = y.reshape(-1, vr)
        t = t.reshape(-1, vr)
        if 0 <= self.ignore_label < vr:
            t[:, self.ignore_label] = 0
        return -np.sum(t * np.log(y + 1e-7))
           
    def _backward(self, gl, y, t):
        y_shape = y.shape
        vr = y_shape[-1]                  # 値幅(one_hotの幅)
        if t.ndim < y.ndim:
            t = np.eye(vr, dtype=bool)[t] # yの形状に合わせる(形状変化対応)
        y = y.reshape(-1, vr)
        t = t.reshape(-1, vr)
        #gl = gl.reshape(-1, vr)          # glも形状を合わせる
        gy = (- gl) * (t / (y + 1e-7))    # メモリ使用量削減のため演算順序拘束
        if 0 <= self.ignore_label < vr:
            gy[:, self.ignore_label] = 0
        return gy.reshape(*y_shape)    

class MeanStdDeviation(nucleus.Function):
    """ 平均と標準偏差をtargetに近づくようにする関数 """
    def __init__(self, mean=2.0, std=0.2, beta1=0, beta2=0, axis=-1):
        super().__init__()
        self.mean = F.Mean(axis=axis)
        self.std  = F.Std(axis=axis)
        self.loss_func1 = MeanSquaredError()
        self.loss_func2 = MeanSquaredError()
        self.target_mean = mean
        self.target_std  = std
        self.beta1 = beta1
        self.beta2 = beta2

    def __forward__(self, x):
        mu    = self.mean(x)
        sigma = self.std(x)
        loss_mean = self.loss_func1(mu, self.target_mean)
        loss_std  = self.loss_func2(sigma, self.target_std)
        loss = self.beta1*loss_mean + self.beta2*loss_std
        return loss  

    def __backward__(self, gl):
        gmu    = self.loss_func1.backward(gl)
        gsigma = self.loss_func2.backward(gl)
        gem = self.mean.backward(gmu)
        ges = self.std.backward(gsigma)
        gx = self.beta1*gem + self.beta2*ges
        return gx
    
class PairwiseGap(nucleus.Function):
    """ 末尾の軸のデータの並びの中の各ペアの差分をgapに近づける損失関数 """
    def __init__(self, gap=0.1, beta=1.0):
        super().__init__()
        self.target_gap = gap
        self.beta = beta

    def __forward__(self, x, gap=None):
        n = x.shape[-1] # ペアをとる末尾の軸
        d = np.expand_dims(x, -1) - np.expand_dims(x, -2)  # (..., n, n)
        self.diffs = d

        if gap is not None: # forwardの際に指定した場合
            self.gap = gap

        # マスク：対角成分を無視（== 0）
        eye = np.eye(n, dtype=bool)
        mask = eye.reshape((1,) * (x.ndim - 1) + (n, n)) # 上位の次元に1を並べる
        mask = np.broadcast_to(mask, d.shape)            # dと同じ形状にする
        
        self.gap_error = np.abs(d) - self.target_gap
        self.gap_error[mask] = 0

        loss = np.mean(self.gap_error ** 2) * n / (n - 1)
        return loss

    def __backward__(self, gl):
        x, = self.inputs
        n = x.shape[-1]
        sign = np.sign(self.diffs)
        grad = self.gap_error * sign

        dx = np.sum(grad, axis=-1) - np.sum(grad, axis=-2)

        # バッチスケール調整: (2 / n(n-1)) / batch_size
        batch_size = np.prod(np.array(x.shape[:-1]))
        scale = gl * (2 / (n * (n - 1))) / batch_size
        return self.beta * dx * scale


class PairwiseGap_bkup(nucleus.Function):
    """ 複数要素の中の各ペアの差分をgapに近づける損失関数 """
    def __init__(self, gap=0.1, beta=0):
        super().__init__()
        self.target_gap = gap
        self.beta = beta

    def __forward__(self, x):
        n = x.shape[0]                       
        d = x[:, None] - x[None, :]          # 各要素の差を並べたペアワイズ差分行列
        mask = ~np.eye(n, dtype=bool)        # 対角要素はFalse
        self.diffs = d[mask].reshape(n, n-1) # 自身を除く差
        self.gap_error = np.abs(self.diffs) - self.target_gap # 目標との乖離
        loss = np.mean(self.gap_error**2)    # 二乗平均
        return loss

    def __backward__(self, gy):
        x, = self.inputs
        n = x.shape[0]
        sign = np.sign(self.diffs)
        grad = gy * (2 / (n * (n - 1))) * np.sum(self.gap_error * sign, axis=1)
        return self.beta * grad

class KullbackLeiblerDivergence():
    def forward(self, mu, log_var):
        self.mu      = mu
        self.log_var = log_var
        loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
        return float(loss) / len(mu)

    def backward(self):
        mu = self.mu
        log_var = self.log_var
        dldmu = mu
        dldlog_var = -0.5 * (1 - np.exp(log_var))
        return dldmu, dldlog_var

    def __call__(self, mu, log_var):
        return self.forward(mu, log_var)

class MeanSquaredErrorForVAE:
    def __init__(self, sumup, enhance):
        self.sumup = sumup
        self.enhance = enhance
        
    def forward(self, y, t):
        y = y.reshape(*t.shape)
        self.y = y
        self.t = t
        rec_error = 0.5 * np.sum((y - t) ** 2) 
        return float(rec_error / len(y))

    def backward(self, gl=1):
        y = self.y
        t = self.t
        gy = gl * (y - t) 
        return gy if self.sumup else gy * self.enhance / len(y) 

class CrossEntropyErrorForVAE:
    '''
    -1 ～ +1 の範囲のCross Entropy(活性化関数tanhに対応)
    '''
    def __init__(self, sumup, enhance):
        self.sumup = sumup
        self.enhance = enhance

    def forward(self, y, t):
        y = y.reshape(*t.shape)
        self.y = y
        self.t = t
        rec_error = - 0.5 * np.sum((1 + t)*np.log(0.5 + 0.5*y + 1e-7) \
                                 + (1 - t)*np.log(0.5 - 0.5*y + 1e-7)) 
        return float(rec_error / len(y))

    def backward(self, gl=1):
        y = self.y
        t = self.t
        gy = - 0.5 * gl * ((1 + t)/(1 + y + 1e-7) - (1 - t)/(1 - y + 1e-7))
        return gy if self.sumup else gy * self.enhance / len(y) 

class CrossEntropyErrorForVAE2:
    '''
    0 ～ 1 の範囲のCross Entropy(活性化関数sigmoidやsoftmaxに対応)
    '''
    def __init__(self, sumup, enhance):
        self.sumup = sumup
        self.enhance = enhance

    def forward(self, y, t):
        y = y.reshape(*t.shape)
        self.y = y
        self.t = t
        rec_error = - np.sum(t*np.log(y+1e-7)+(1-t)*np.log(1-y+1e-7))
        return float(rec_error  / len(y))

    def backward(self, gl=1):
        y = self.y
        t = self.t
        gy = - gl * (t/(y + 1e-7)- (1 - t)/(1 - y +1e-7))
        return gy if self.sumup else gy * self.enhance / len(y) 

class LossFunctionForGAN:
    def __init__(self, sumup, enhance):
        self.sumup = sumup
        self.enhance = enhance

    def forward(self, y, t):
        self.y = y
        self.t = t
        loss = -np.sum(t * np.log(y + 1e-7) + (1 - t) * np.log(1 - y + 1e-7))
        return loss / len(y)

    def backward(self, gl=1):
        y = self.y 
        t = self.t 
        gy = - gl * (t / (y + 1e-7) - (1 - t) / (1 - y + 1e-7))
        return gy if self.sumup else gy * self.enhance / len(y)

    def forward_for_gen(self, y):
        self.y = y
        loss = -np.sum(np.log(1 - y + 1e-7))
        return loss / len(y)

    def backward_for_gen(self, gl=1):
        y = self.y
        gy = gl * ( - 1 / (1 - y + 1e-7))
        return gy if self.sumup else gy * self.enhance / len(y) 

class LossFunctionForGAN2:
    def __init__(self, sumup, enhance):
        self.sumup = sumup
        self.enhance = enhance

    def forward(self, y, t):
        self.y = y
        self.t = t
        loss = -np.sum(t * np.log(y + 1e-7) + (1 - t) * np.log(1 - y + 1e-7))
        return loss / len(y)

    def backward(self, gl=1):
        y = self.y 
        t = self.t 
        gy = - gl * (t / (y + 1e-7) - (1 - t) / (1 - y + 1e-7))
        return gy if self.sumup else gy * self.enhance / len(y)

    def forward_for_gen(self, y):
        self.y = y
        loss = -np.sum(np.log(y + 1e-7))
        return loss / len(y)

    def backward_for_gen(self, gl=1):
        y = self.y
        gy = -gl * (1 / (y + 1e-7))
        return gy if self.sumup else gy * self.enhance / len(y)

class LossFunctionForGAN3:
    def __init__(self, sumup, enhance):
        self.sumup = sumup
        self.enhance = enhance

    def forward(self, y, t):
        self.y = y
        self.t = t
        loss = -np.sum(t * np.log(y + 1e-7) + (1 - t) * np.log(1 - y + 1e-7))
        return loss / len(y)

    def backward(self, gl=1):
        y = self.y 
        t = self.t 
        gy = - gl * (t / (y + 1e-7) - (1 - t) / (1 - y + 1e-7))
        return gy if self.sumup else gy * self.enhance / (2 * len(y))

    def forward_for_gen(self, y):
        self.y = y
        loss = -np.sum(np.log(y + 1e-7) + np.log(1 - y + 1e-7))
        return loss / (2 * len(y))

    def backward_for_gen(self, gl=1):
        y = self.y
        gy = - gl / (y + 1e-7) - gl / (1 - y + 1e-7)
        return gy if self.sumup else gy * self.enhance / (2 * len(y))

class LossFunctionForGAN4:
    def __init__(self, sumup, enhance):
        self.sumup = sumup
        self.enhance = enhance

    def forward(self, y, t):
        self.y = y
        self.t = t
        loss = -np.sum(t * y - (1 - t) * y)
        return loss / len(y)

    def backward(self, gl=1):
        y = self.y 
        t = self.t 
        gy = - gl * (2 * t - 1)
        return gy if self.sumup else gy * self.enhance / (2 * len(y))

    def forward_for_gen(self, y):
        self.y = y
        loss = -np.sum(y)
        return loss / len(y)

    def backward_for_gen(self, gl=1):
        y = self.y
        gy = -gl * np.ones_like(y)
        return gy if self.sumup else gy * self.enhance / len(y)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    print('Test LossFunctions')

    loss_functions = (MeanSquaredError,
                      CrossEntropyError,
                      CrossEntropyError2,
                      ) 
    
    y = np.linspace(0.1, 0.9, 10)
    t = np.linspace(0.9, 0.1, 10)

    for lf in loss_functions:
        print(lf.__name__)        
        loss = lf()
        l = loss.forward(y, t)
        gy = loss.backward()
        print('loss =', l)         
        plt.plot(y.tolist(), label='y')
        plt.plot(t.tolist(), label='t')
        plt.plot(gy.tolist(), label='gy')
        plt.legend()
        plt.show()

    print('-- test CrossEntropyErrorMasked','-'*20)
    logits = np.arange(24).reshape(2,3,4)
    t = np.array([[0,1,2],[1,2,-1]])
    print("logits\n", logits)
    print("t\n", t)
    y = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    y /= np.sum(y, axis=-1, keepdims=True)
    print("y = softmax(logits)\n", y)

    
    loss_f = CrossEntropyErrorMasked(ignore=-1)
    print('\ntがターゲットラベルの場合')
    loss = loss_f.forward(y, t)
    print("loss:", loss)
    grad = loss_f.backward()
    print("grad:\n", grad)

    print('\ntがone hotの場合')
    loss = loss_f.forward(y, np.eye(y.shape[-1], dtype=int)[t])
    print("loss:", loss)
    grad = loss_f.backward()
    print("grad:\n", grad)
