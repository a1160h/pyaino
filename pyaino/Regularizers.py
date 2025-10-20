# Regularizers
# 202510.20 A.Inoue
from pyaino.Config import *
from pyaino.nucleus import Function
from pyaino import common_function as cf
from pyaino import LossFunctions as lf
from pyaino import Functions as F
from pyaino import Optimizers


class EntropyUnit(Function):
    def __forward__(self, p):
        entropy = - p * np.log(p)
        return entropy  

    def __backward__(self, ge):
        p, = self.inputs
        gp = ge * (-np.log(p) - 1.0)
        return gp


class KLDivergenceUnit(Function):
    def __forward__(self, p, q):
        kld = p * np.log(p / q)
        return kld

    def __backward__(self, gy):
        p, q = self.inputs
        gp = gy * (np.log(p / q) + 1)
        gq = - gy * (p / q)
        return gp, gq

class SymmetricKLDivergenceUnit(Function):
    def __forward__(self, p, q):
        kld = p * np.log(p / q) + q * np.log(q / p)
        return 0.5 * kld 

    def __backward__(self, gy):
        p, q = self.inputs
        gp = 0.5 * gy * (np.log(p / q) + 1 - q / p)
        gq = 0.5 * gy * (np.log(q / p) + 1 - p / q)
        return gp, gq

class JSDivergenceUnit(Function):
    def __init__(self, log_base='e', eps=1e-9):
        super().__init__()
        self.log = np.log if log_base=='e' else np.log2
        self.eps = eps

    def __forward__(self, p, q):
        eps = self.eps
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        m = 0.5 * (p + q)
        self.m = m
        klp = p * self.log(p / m)
        klq = q * self.log(q / m)
        return 0.5 * (klp + klq)

    def __backward__(self, gy):
        p, q = self.inputs
        eps = self.eps
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        m = self.m  
        gp = 0.5 * gy * self.log(p / m)
        gq = 0.5 * gy * self.log(q / m)
        return gp, gq

class EntropyDivergence(Function):
    """ エントロピーの平均の隔たり """
    def __init__(self, axis1=-1, axis2=(0,2), keepdims=True, eps=1e-9):
        super().__init__()
        self.axis1 = axis1
        axis1 = (axis1,) if type(axis1) is not tuple else axis1 # 統計量算出軸
        if axis2 is None:
            self.axis = axis1
        else:    
            axis2 = (axis2,) if type(axis2) is not tuple else axis2 # 平均軸
            self.axis = axis2 + axis1
        self.unit = EntropyUnit()
        self.mean = F.Mean(axis=self.axis, keepdims=keepdims)
        self.eps = eps
        
    def __forward__(self, a):             # a : (B,h,Tq,Tk)
        self.Tk = a.shape[self.axis1]  
        ac = np.clip(a, self.eps, 1.0)
        entropy = self.unit(ac)
        entropy = self.mean(entropy) * self.Tk # 全軸mean->末尾の軸のみsum
        return entropy
    
    def __backward__(self, ge):
        ga = self.mean.backward(ge * self.Tk)
        ga = self.unit.backward(ga)
        return ga

class EntropyDivergence2(Function):
    """ エントロピーの平均の隔たり """
    def __init__(self, axis1=-1, axis2=0, axis3=None, keepdims=True, eps=1e-9):
        super().__init__()
        self.axis1 = axis1
        axis1 = (axis1,) if type(axis1) is not tuple else axis1 # 統計量算出軸
        if axis2 is None:
            self.axis = axis1
        else:    
            axis2 = (axis2,) if type(axis2) is not tuple else axis2 # 平均軸
            self.axis = axis2 + axis1
        self.unit = EntropyUnit()
        self.mean = F.Mean(axis=self.axis, keepdims=keepdims)
        if axis3 is not None:
            self.var  = F.Var(axis=axis3, keepdims=True)
        else:
            self.var = None
        self.eps = eps
        
    def __forward__(self, a):             # a : (B,h,Tq,Tk)
        self.Tk = a.shape[self.axis1]  
        ac = np.clip(a, self.eps, 1.0)
        entropy = self.unit(ac)
        entropy = self.mean(entropy) * self.Tk # 全軸mean->末尾の軸のみsum
        if self.var is not None:
            entropy = self.var(entropy)
        return entropy
    
    def __backward__(self, ge):
        if self.var is not None:
            ga = self.var.backward(ge)
        else:
            ga = ge
        ga = self.mean.backward(ga * self.Tk)
        ga = self.unit.backward(ga)
        return ga


class PairDivergence(Function):
    def __init__(self, unit, method='permutation', symmetric=False, 
                 axis0=1, axis1=-1, axis2=(0,2), keepdims=True, flatten=False,
                 log_base='e', eps=1e-9):
        """p: モデルからの出力, q: 目標分布"""
        super().__init__()
        #self.pairwise = F.Pairwise(axis=axis0, broadcast=True, diagonal_mask=True)  
        self.axis1 = axis1
        axis1 = (axis1,) if type(axis1) is not tuple else axis1 # 統計量算出軸
        if axis2 is None:
            self.axis = axis1
        else:    
            axis2 = (axis2,) if type(axis2) is not tuple else axis2 # 平均軸
            self.axis = axis2 + axis1
        self.flatten = flatten
        self.eps = eps
        self.take_pair = F.TakePair(axis0, method)
        self.unit = unit
        self.mean = F.Mean(axis=self.axis, keepdims=keepdims)
        self.Tk = None
        self.y_shape = None
        self.p, self.q = None, None

    def __forward__(self, a):
        self.Tk = a.shape[self.axis1]
        p, q = self.take_pair(a)
        p = np.clip(p, self.eps, 1.0)
        q = np.clip(q, self.eps, 1.0)
        y = self.unit(p, q)
        y = self.mean(y) * self.Tk # 一旦全てmeanをとってからTk軸はsumに戻す
        self.y_shape = y.shape
        if self.flatten:
            y = y.reshape(-1)
        #print('###', y.shape)    
        return y
    
    def __backward__(self, gy):
        if self.flatten:
            gy = gy.reshape(self.y_shape)
        gl = self.mean.backward(gy * self.Tk)
        gp, gq = self.unit.backward(gl)
        ga = self.take_pair.backward(gp, gq)
        return ga


class KLDivergence(PairDivergence):
    def __init__(self, **kwargs):
        symmetric = kwargs.pop('symmetric', False)
        eps       = kwargs.pop('eps',        1e-9)
        if symmetric:
            unit = SymmetricKLDivergenceUnit()#eps=eps)
        else:
            unit = KLDivergenceUnit()#eps=eps)
        super().__init__(unit, **kwargs)

class JSDivergence(PairDivergence):
    def __init__(self, **kwargs):
        eps       = kwargs.pop('eps',     1e-9)
        log_base  = kwargs.pop('log_base', 'e')
        unit = JSDivergenceUnit(log_base=log_base, eps=eps)
        super().__init__(unit, **kwargs)

class MeanVarDeviation(Function):
    """ 平均と標準偏差をtargetに近づくようにする関数 """
    def __init__(self, mean=2.0, var=0.2, beta1=0, beta2=0, axis=-1):
        super().__init__()
        self.mean = F.Mean(axis=axis)
        self.var  = F.Var(axis=axis)
        self.loss_func1 = lf.MeanSquaredError()
        self.loss_func2 = lf.MeanSquaredError()
        self.target_mean = mean
        self.target_var  = var
        self.beta1 = beta1
        self.beta2 = beta2

    def __forward__(self, x):
        mu    = self.mean(x)
        sigma = self.var(x)
        loss_mean = self.loss_func1(mu, self.target_mean)
        loss_var  = self.loss_func2(sigma, self.target_var)
        loss = self.beta1*loss_mean + self.beta2*loss_var
        return loss  

    def __backward__(self, gl):
        gmu    = self.loss_func1.backward(gl)
        gsigma = self.loss_func2.backward(gl)
        gem = self.mean.backward(gmu)
        ges = self.var.backward(gsigma)
        gx = self.beta1*gem + self.beta2*ges
        return gx
    
class MeanStdDeviation(Function):
    """ 平均と標準偏差をtargetに近づくようにする関数 """
    def __init__(self, mean=2.0, std=0.2, beta1=0, beta2=0, axis=-1):
        super().__init__()
        self.mean = F.Mean(axis=axis)
        self.std  = F.Std(axis=axis)
        self.loss_func1 = lf.MeanSquaredError()
        self.loss_func2 = lf.MeanSquaredError()
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
    
class PairwiseGap(Function):
    """ 指定する軸のデータの並びの中の各ペアの差分をgapに近づける損失関数 """
    def __init__(self, gap=0.1, beta=1.0, axis=1, method='combination'):
        super().__init__()
        self.target_gap = gap
        self.beta = beta
        self.axis = axis
        self.take_pair = F.TakePair(axis, method)
        self.square_mean = F.SquareMean()

    def __forward__(self, x, gap=None):
        if gap is not None: # forwardの際に指定した場合
            self.target_gap = gap
        p, q = self.take_pair(x)
        d = p - q
        self.diffs = d
        self.sign = np.sign(d)
        self.gap_error = np.abs(d) - self.target_gap
        loss = self.square_mean(self.gap_error)
        return loss

    def __backward__(self, gl):
        x, = self.inputs
        gx = self.square_mean.backward(gl)
        gx *= self.sign
        gx = self.take_pair.backward(gx, -gx)
        return self.beta * gx
    
class PairwiseGap_bkup(Function):
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
            self.target_gap = gap

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

    
class AttentionRegularizer(Function):
    """  """
    def __init__(self,
                 divergence1=EntropyDivergence(),
                 regularize1=None,
                 scheduler1=None, 
                 divergence2=None,
                 regularize2=None,
                 scheduler2=None, # reguiarize2のscheduler
                 divergence3=None,
                 regularize3=None,
                 scheduler3=None, # reguiarize3のscheduler
                 axis1=(0,2,3),   # divergence1の結果の平均軸
                 axis2=(0,2,3),   # divergence2の結果の平均軸
                 axis3=(0,2,3),   # divergence3の結果の平均軸
                 eta1=0,
                 eta2=0,
                 eta3=0,
                 ):
        super().__init__()
        if type(divergence1) == str:
            self.divergence1 = cf.eval_in_module(divergence1, None)
        else:    
            self.divergence1 = divergence1
        if type(regularize1) == str:
            self.regularize1 = cf.eval_in_module(regularize1, None)
        else:    
            self.regularize1 = regularize1
        if type(scheduler1) == str:
            self.scheduler1  = cf.eval_in_module(scheduler1, Optimizers)
        else:    
            self.scheduler1  = scheduler1
            
        if type(divergence2) == str:
            self.divergence2 = cf.eval_in_module(divergence2, None)
        else:    
            self.divergence2 = divergence2
        if type(regularize2) == str:
            self.regularize2 = cf.eval_in_module(regularize2, None)
        else:    
            self.regularize2 = regularize2
        if type(scheduler2) == str:
            self.scheduler2  = cf.eval_in_module(scheduler2, Optimizers)
        else:    
            self.scheduler2  = scheduler2

        if type(divergence3) == str:
            self.divergence3 = cf.eval_in_module(divergence3, None)
        else:    
            self.divergence3 = divergence3
        if type(regularize3) == str:
            self.regularize3 = cf.eval_in_module(regularize3, None)
        else:    
            self.regularize3 = regularize3
        if type(scheduler3) == str:
            self.scheduler3  = cf.eval_in_module(scheduler3, Optimizers)
        else:    
            self.scheduler3  = scheduler3

        self.axis1 = axis1
        self.axis2 = axis2
        self.axis3 = axis3
        
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3

        self.iter = 0

        print(self.__class__.__name__,
              '\ndivergence1:', self.divergence1.__class__.__name__,
              '\nregularize1:', self.regularize1.__class__.__name__,
              '\nscheduler1:',  self.scheduler1.__class__.__name__,
              '\ndivergence2:', self.divergence2.__class__.__name__,
              '\nregularize2:', self.regularize2.__class__.__name__,
              '\nscheduler2:', self.scheduler2.__class__.__name__,
              '\ndivergence3:', self.divergence3.__class__.__name__,
              '\nregularize3:', self.regularize3.__class__.__name__,
              '\nscheduler3:', self.scheduler3.__class__.__name__,
              )
        
        self.result1, self.result2, self.result3 = None, None, None 

    def __forward__(self, a, target=None):
        result1, result2, result3 = None, None, None

        if self.divergence1 is not None:
            result1 = self.divergence1(a)
        if self.divergence2 is not None:    
            result2 = self.divergence2(a)
        if self.divergence3 is not None:    
            result3 = self.divergence3(a)
        loss1 = 0 if self.regularize1 is None else self.regularize1(result1)
        loss2 = 0 if self.regularize2 is None else self.regularize2(result2)
        loss3 = 0 if self.regularize3 is None else self.regularize3(result3)
        # 以下は計測用、head毎の値は末尾の軸、それ以外の軸はバッチ軸など平均をとる
        #print(result1.shape, result2.shape, result3.shape)
        if self.axis1 is not None and self.divergence1 is not None:
            self.result1 = np.mean(result1, axis=self.axis1)
        else:
            self.result1 = result1
        if self.axis2 is not None and self.divergence2 is not None:
            self.result2 = np.mean(result2, axis=self.axis2)
        else:
            self.result2 = result2
        if self.axis3 is not None and self.divergence3 is not None:
            self.result3 = np.mean(result3, axis=self.axis3)
        else:
            self.result3 = result3
        #print(self.result1.shape, self.result2.shape, self.result3.shape)    
        return loss1 + loss2 + loss3

    def __backward__(self, gl):
        if  self.regularize1 is None \
        and self.regularize2 is None \
        and self.regularize3 is None:
            return 0
        
        if self.regularize1 is None:
            ga1 = 0
        else:
            gy1 = self.regularize1.backward(gl)
            ga1 = self.divergence1.backward(gy1)
        if self.scheduler1 is not None:
            eta1 = self.eta1 * self.scheduler1(self.iter)
        else:
            eta1 = self.eta1
            
        if self.regularize2 is None:
            ga2 = 0
        else:
            gy2 = self.regularize2.backward(gl)
            ga2 = self.divergence2.backward(gy2) 
        if self.scheduler2 is not None:
            eta2 = self.eta2 * self.scheduler2(self.iter)
        else:
            eta2 = self.eta2

        if self.regularize3 is None:
            ga3 = 0
        else:
            gy3 = self.regularize3.backward(gl)
            ga3 = self.divergence3.backward(gy3) 
        if self.scheduler3 is not None:
            eta3 = self.eta3 * self.scheduler3(self.iter)
        else:
            eta3 = self.eta3

        self.iter += 1    

        return ga1 * eta1 + ga2 * eta2 + ga3 * eta3
   
