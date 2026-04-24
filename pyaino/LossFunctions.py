# LossFunctions
# 2026.04.23 A.Inoue
from pyaino.Config import *
from pyaino import nucleus
from pyaino import safe_np as snp
from pyaino import Functions as F
import math, warnings

class LossFunctionBase(nucleus.Function):
    def __init__(self, reduction='mean', label2onehot=False, onehot2label=False):
        """
        reduction
        none     : return loss per element / per structure
        sum      : return an unnormalized global sum
        mean     : return a fully normalized mean (implementation-level)
        sample   : return E_x [ ℓ(x) ] (sample-level objective)

        """
        super().__init__()
        self.reduction = reduction
        self.t = None
        self.l_shape = None
        self.label2onehot = label2onehot
        self.onehot2label = onehot2label
        self._cached_denom = None
        self.sample_size = None
        pass

    def _mean_denominator(self, l):
        """reduction='mean' 時の分母。必要なら派生クラスで上書きする。"""
        if l.ndim == 0:
            return 1
        return math.prod(l.shape)

    def __forward__(self, y, t):
        # tの整形
        if self.label2onehot and y.shape != t.shape:   # tが正解ラベル 
            self.t = np.eye(y.shape[-1], dtype=Config.dtype)[t]
        elif self.onehot2label and y.shape == t.shape: # tがone_hot
            self.t = np.argmax(t, axis=-1)
            warnings.warn('Given target is one_hot. Target category recommended.')
        else:
            self.t = t

        # yのチェックと整形
        if y.ndim == 0:
            self.sample_size = 1
        else:
            self.sample_size = y.shape[0]

        if self.onehot2label and y.ndim > 2:
            y = y.reshape(-1, y.shape[-1])
            self.t = self.t.reshape(-1)

        # 損失
        l = self._forward(y, self.t)
        self.l_shape = l.shape

        # reduction
        if self.reduction == 'mean':
            denom = self._mean_denominator(l)
            self._cached_denom = np.array(denom, dtype=Config.dtype)
            return l.sum() / self._cached_denom
        if self.reduction == 'sum':
            return l.sum()
        if self.reduction == 'sample':
            return l.sum() / self.sample_size
        if self.reduction is None or self.reduction == 'none':
            return l
        raise ValueError(f'Invalid reduction {self.reduction}')

    def __backward__(self, gl=1):
        y, _ = self.inputs
        y_shape = y.shape
        # reduction の逆伝播
        if self.reduction is None or self.reduction == 'none':
            pass
        else:
            for ax in range(len(self.l_shape)):
                gl = np.expand_dims(gl, axis=ax).astype(Config.dtype)
            gl = np.broadcast_to(gl, self.l_shape)
            if self.reduction == 'mean':
                gl /= self._cached_denom
            if self.reduction == 'sample':
                gl /= self.sample_size
        # yの整形の再現
        if self.onehot2label and y.ndim > 2:
            y = y.reshape(-1, y.shape[-1])
        # 損失の逆伝播
        gy = self._backward(gl, y, self.t)
        # gyの整形
        gy = gy.reshape(*y_shape).astype(Config.dtype)
        return gy
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class MeanSquaredError(LossFunctionBase):
    def __init__(self, reduction='mean', **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def _forward(self, y, t):
        return 0.5 * (y - t) ** 2

    def _backward(self, gl, y, t):
        return gl * (y - t)

class CrossEntropyError(LossFunctionBase):
    def __init__(self, reduction='mean', label2onehot=True):
        super().__init__(reduction=reduction, label2onehot=label2onehot)

    def _forward(self, y, t):
        return - t * np.log(y + 1e-7)        # メモリ使用量削減のため演算順序拘束

    def _backward(self, gl, y, t):
        return (- gl) * (t / (y + 1e-7))

class CrossEntropyError2(LossFunctionBase):
    def __init__(self, reduction='mean', **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def _forward(self, y, t):
        return - t * np.log(y + 1e-7) - (1 - t) * np.log(1 - y + 1e-7)

    def _backward(self, gl, y, t):
        return gl * ((y - t) / (y * (1 - y) + 1e-7)) # メモリ使用量削減のため演算順序拘束

class CrossEntropyErrorMasked(LossFunctionBase):
    def __init__(self, reduction='mean', ignore=None, onehot2label=True):
        super().__init__(reduction=reduction, onehot2label=onehot2label)
        self.ignore_label = ignore
        self.valid_mask = None

    def _mean_denominator(self, l):
        """ 親クラスを上書き、有効要素数を返す """
        if self.valid_mask is None:
            return super()._mean_denominator(l)
        denom = int(snp.sum(self.valid_mask))
        return max(denom, 1)

    def _forward(self, y, t):
        """ y: probabilities, t: 正解ラベル """
        if self.ignore_label is not None:
            self.valid_mask = (t != self.ignore_label)
        else:
            self.valid_mask = np.ones_like(t, dtype=bool)

        l = np.zeros_like(t, dtype=Config.dtype)
        valid_indices = np.where(self.valid_mask)[0]
        if len(valid_indices) > 0:
            l[valid_indices] = -np.log(y[valid_indices, t[valid_indices]] + 1e-7)
        return l

    def _backward(self, gl, y, t):
        gy = np.zeros_like(y)
        valid_indices = np.where(self.valid_mask)[0]
        if len(valid_indices) > 0:
            gy[valid_indices, t[valid_indices]] = (
                -gl[valid_indices] / (y[valid_indices, t[valid_indices]] + 1e-7)
            )
        return gy
    
class CrossEntropyErrorForLogits(LossFunctionBase):
    def __init__(self, reduction='mean', ignore=None, onehot2label=True):
        super().__init__(reduction=reduction, onehot2label=onehot2label)
        self.ignore = ignore
        self.log_probs = None
        self.mask = None
        self.safe_t = None

    def _mean_denominator(self, l):
        """ 親クラスを上書き、有効要素数を返す """
        if self.mask is None:
            return super()._mean_denominator(l)
        denom = int(snp.sum(self.mask))
        return max(denom, 1)

    def _forward(self, y, t):
        B, num_classes = y.shape

        max_y = np.max(y, axis=1, keepdims=True)
        y_stable = y - max_y
        logsumexp = np.log(snp.sum(np.exp(y_stable), axis=1, keepdims=True)) + max_y
        self.log_probs = y - logsumexp  # log-softmax

        if self.ignore is not None:
            self.mask = (t != self.ignore)
            self.safe_t = np.where(self.mask, t, 0)
        else:
            self.mask = np.ones_like(t, dtype=bool)
            self.safe_t = t

        l = np.zeros(B, dtype=Config.dtype)
        valid_indices = np.where(self.mask)[0]
        if len(valid_indices) > 0:
            l[valid_indices] = -self.log_probs[valid_indices, self.safe_t[valid_indices]]
        return l

    def _backward(self, gl, y, t):
        grad = np.exp(self.log_probs)
        valid_indices = np.where(self.mask)[0]
        if len(valid_indices) > 0:
            grad[valid_indices, self.safe_t[valid_indices]] -= 1.0
        grad[~self.mask] = 0
        return grad * gl[:, np.newaxis]

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

        dx = snp.sum(grad, axis=-1) - snp.sum(grad, axis=-2)

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
        grad = gy * (2 / (n * (n - 1))) * snp.sum(self.gap_error * sign, axis=1)
        return self.beta * grad

class KullbackLeiblerDivergence():
    def forward(self, mu, log_var):
        self.mu      = mu
        self.log_var = log_var
        loss = -0.5 * snp.sum(1 + log_var - mu**2 - np.exp(log_var))
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
        rec_error = 0.5 * snp.sum((y - t) ** 2) 
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
        rec_error = - 0.5 * snp.sum((1 + t)*np.log(0.5 + 0.5*y + 1e-7) \
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
        rec_error = - snp.sum(t*np.log(y+1e-7)+(1-t)*np.log(1-y+1e-7))
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
        loss = -snp.sum(t * np.log(y + 1e-7) + (1 - t) * np.log(1 - y + 1e-7))
        return loss / len(y)

    def backward(self, gl=1):
        y = self.y 
        t = self.t 
        gy = - gl * (t / (y + 1e-7) - (1 - t) / (1 - y + 1e-7))
        return gy if self.sumup else gy * self.enhance / len(y)

    def forward_for_gen(self, y):
        self.y = y
        loss = -snp.sum(np.log(1 - y + 1e-7))
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
        loss = -snp.sum(t * np.log(y + 1e-7) + (1 - t) * np.log(1 - y + 1e-7))
        return loss / len(y)

    def backward(self, gl=1):
        y = self.y 
        t = self.t 
        gy = - gl * (t / (y + 1e-7) - (1 - t) / (1 - y + 1e-7))
        return gy if self.sumup else gy * self.enhance / len(y)

    def forward_for_gen(self, y):
        self.y = y
        loss = -snp.sum(np.log(y + 1e-7))
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
        loss = -snp.sum(t * np.log(y + 1e-7) + (1 - t) * np.log(1 - y + 1e-7))
        return loss / len(y)

    def backward(self, gl=1):
        y = self.y 
        t = self.t 
        gy = - gl * (t / (y + 1e-7) - (1 - t) / (1 - y + 1e-7))
        return gy if self.sumup else gy * self.enhance / (2 * len(y))

    def forward_for_gen(self, y):
        self.y = y
        loss = -snp.sum(np.log(y + 1e-7) + np.log(1 - y + 1e-7))
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
        loss = -snp.sum(t * y - (1 - t) * y)
        return loss / len(y)

    def backward(self, gl=1):
        y = self.y 
        t = self.t 
        gy = - gl * (2 * t - 1)
        return gy if self.sumup else gy * self.enhance / (2 * len(y))

    def forward_for_gen(self, y):
        self.y = y
        loss = -snp.sum(y)
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
        plt.title(type(loss).__name__)
        plt.legend()
        plt.show()

    print('-- test CrossEntropyErrorMasked','-'*20)
    logits = np.arange(24).reshape(2,3,4)
    t = np.array([[0,1,2],[1,2,3]])
    print("logits\n", logits)
    print("t\n", t)
    y = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    y /= np.sum(y, axis=-1, keepdims=True)
    print("y = softmax(logits)\n", y)
   
    loss_f = CrossEntropyErrorMasked(ignore=3)
    print('\ntがターゲットラベルの場合')
    loss = loss_f.forward(y, t)
    print("loss:", loss)
    grad = loss_f.backward()
    print("grad:\n", grad)

    print(loss_f.ignore_label)
    print(loss_f.valid_mask)

    print('\ntがone hotの場合')
    loss = loss_f.forward(y, np.eye(y.shape[-1], dtype=int)[t])
    print("loss:", loss)
    grad = loss_f.backward()
    print("grad:\n", grad)

    print(loss_f.ignore_label)
    print(loss_f.valid_mask)
