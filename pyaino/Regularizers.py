# Regularizers
# 20260902 A.Inoue
from pyaino.Config import *
from pyaino.nucleus import Function
from pyaino import common_function as cf
from pyaino import LossFunctions as lf
from pyaino import Functions as F
from pyaino import Optimizers
import itertools


class EntropyUnit(Function):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def __forward__(self, p):
        eps = self.eps
        p = np.clip(p, eps, 1.0)
        entropy = - p * np.log(p)
        return entropy  

    def __backward__(self, ge):
        p, = self.inputs
        eps = self.eps
        p = np.clip(p, eps, 1.0)
        gp = ge * (-np.log(p) - 1.0)
        return gp


class KLDivergenceUnit(Function):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def __forward__(self, p, q):
        eps = self.eps
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        kld = p * np.log(p / q)
        return kld

    def __backward__(self, gy):
        p, q = self.inputs
        eps = self.eps
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        gp = gy * (np.log(p / q) + 1)
        gq = - gy * (p / q)
        return gp, gq

class SymmetricKLDivergenceUnit(Function):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def __forward__(self, p, q):
        eps = self.eps
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        kld = p * np.log(p / q) + q * np.log(q / p)
        return 0.5 * kld 

    def __backward__(self, gy):
        p, q = self.inputs
        eps = self.eps
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
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
        klp = p * self.log(p / m)
        klq = q * self.log(q / m)
        return 0.5 * (klp + klq)

    def __backward__(self, gy):
        p, q = self.inputs
        eps = self.eps
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        m = 0.5 * (p + q) 
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


class PairRoundRobin:
    """ round robin順に指定された2つのhead pairを取り出す """

    def __init__(self, n_head, axis=1):
        self.axis = axis
        self.head_pairs = list(
            itertools.combinations(range(n_head), 2)
        )
        self.schedule = self.make_schedule()

    def make_schedule(self):
        players = self.head_pairs.copy()

        if len(players) % 2:
            players.append(None)

        schedule = []

        for _ in range(len(players) - 1):
            for i in range(len(players) // 2):
                p = players[i]
                q = players[-1 - i]

                if p is not None and q is not None:
                    schedule.append((p, q))

            players = [players[0], players[-1]] + players[1:-1]

        return schedule

    def get_schedule(self):
        return self.schedule

    def forward(self, x, index):
        indices = self.schedule[index]
        self.take = F.Take(self.axis, indices)
        y = self.take(x)
        y = np.moveaxis(y, self.axis + 1, 0)
        return y[0], y[1]

    def backward(self, gp, gq):
        gy = np.stack([gp, gq], axis=self.axis + 1)
        return self.take.backward(gy)

    def __call__(self, *args):
        return self.forward(*args)
    
class PairDivergence(Function):
    """ unitで数学的に与えられるKLDやJSDを4軸のattention weightの測定や制御に供する """
    def __init__(self, unit, method='permutation', symmetric=False,
                 round_robin=False, n_head=None,
                 axis0=1, axis1=-1, axis2=(0,2), keepdims=True, flatten=False,
                 log_base='e', eps=1e-9):
        """p: モデルからの出力, q: 目標分布"""
        super().__init__()
        self.axis0 = axis0
        self.method = method
        self.round_robin = round_robin
        self.n_head = n_head
        self.index = 0

        if round_robin:
            self.take_pair = PairRoundRobin(n_head, axis0)
        else:
            self.take_pair = F.TakePair(axis0, method)

        self.axis1 = axis1
        axis1 = (axis1,) if type(axis1) is not tuple else axis1 # 統計量算出軸
        if axis2 is None:
            self.axis = axis1
        else:
            axis2 = (axis2,) if type(axis2) is not tuple else axis2 # 平均軸
            self.axis = axis2 + axis1
        self.flatten = flatten
        self.eps = eps
        self.unit = unit
        self.mean = F.Mean(axis=self.axis, keepdims=keepdims)
        self.Tk = None
        self.y_shape = None
        self.p, self.q = None, None

    def get_schedule(self):
        if self.round_robin:
            return self.take_pair.get_schedule()
        return None

    def __forward__(self, a):
        self.Tk = a.shape[self.axis1]

        if self.round_robin:
            p, q = self.take_pair(a, self.index)
        else:
            p, q = self.take_pair(a)

        y = self.unit(p, q)
        y = self.mean(y) * self.Tk # 一旦全てmeanをとってからTk軸はsumに戻す
        self.y_shape = y.shape
        if self.flatten:
            y = y.reshape(-1)
        return y

    def __backward__(self, gy):
        if self.flatten:
            gy = gy.reshape(self.y_shape)
        gl = self.mean.backward(gy * self.Tk)
        gp, gq = self.unit.backward(gl)
        ga = self.take_pair.backward(gp, gq)

        if self.round_robin:
            self.index = (self.index + 1) % len(self.take_pair.get_schedule())

        return ga

class KLDivergence(PairDivergence):
    def __init__(self, **kwargs):
        symmetric = kwargs.pop('symmetric', False)
        eps       = kwargs.pop('eps',        1e-9)
        if symmetric:
            kwargs.setdefault('method', 'combination')
            unit = SymmetricKLDivergenceUnit(eps=eps)
        else:
            unit = KLDivergenceUnit(eps=eps)
        super().__init__(unit, **kwargs)

class JSDivergence(PairDivergence):
    def __init__(self, **kwargs):
        eps       = kwargs.pop('eps',     1e-9)
        log_base  = kwargs.pop('log_base', 'e')
        kwargs.setdefault('method', 'combination')
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
    
   
class AttentionRegularizer(Function):
    """ MultiHeadAttentionへの組込み機構 """
    def __init__(self,
                 divergence1=EntropyDivergence(),
                 regularize1=None,
                 scheduler1=None,
                 divergence2=None,
                 regularize2=None,
                 scheduler2=None,
                 divergence3=None,
                 regularize3=None,
                 scheduler3=None,
                 axis1=(0,2,3),
                 axis2=(0,2,3),
                 axis3=(0,2,3),
                 eta1=0,
                 eta2=0,
                 eta3=0,
                ):
        super().__init__()

        settings = [
            (divergence1, regularize1, scheduler1, axis1, eta1),
            (divergence2, regularize2, scheduler2, axis2, eta2),
            (divergence3, regularize3, scheduler3, axis3, eta3),
        ]

        self.settings = []

        for divergence, regularize, scheduler, axis, eta in settings:
            if type(divergence) == str:
                divergence = cf.eval_in_module(divergence, None)
            if type(regularize) == str:
                regularize = cf.eval_in_module(regularize, None)
            if type(scheduler) == str:
                scheduler = cf.eval_in_module(scheduler, Optimizers)

            round_robin = (
                divergence is not None and
                getattr(divergence, 'round_robin', False)
            )

            record_schedule = None
            record_rr_index = None
            record_index = None
            record_seen = None

            if round_robin:
                n_head = divergence.n_head
                record_schedule = divergence.get_schedule()
                record_rr_index = 0
                record_index = {
                    pair: i for i, pair in enumerate(
                        itertools.combinations(range(n_head), 2))
                }
                record_seen = set()

            self.settings.append({
                'divergence': divergence,
                'regularize': regularize,
                'scheduler': scheduler,
                'axis': axis,
                'eta': eta,
                'round_robin': round_robin,
                'record_schedule': record_schedule,
                'record_rr_index': record_rr_index,
                'record_index': record_index,
                'record_buffer': None,
                'record_seen': record_seen,
                'result': None,
                'record': [],
            })

        self.iter = 0

        print(self.__class__.__name__)
        for i, setting in enumerate(self.settings, 1):
            divergence = setting['divergence']
            regularize = setting['regularize']
            scheduler = setting['scheduler']
            print(f'[{i}]',
                   '\ndivergence:',
                   None if divergence is None else divergence.__class__.__name__,
                   '\nregularize:',
                   None if regularize is None else regularize.__class__.__name__,
                   '\nscheduler:',
                   None if scheduler is None else scheduler.__class__.__name__,
            )

    def get_record1(self):
        return self.settings[0]['record']

    def get_record2(self):
        return self.settings[1]['record']

    def get_record3(self):
        return self.settings[2]['record']

    def get_records(self):
        return [s['record'] for s in self.settings]

    def __forward__(self, a, target=None):
        loss = 0

        for setting in self.settings:
            divergence = setting['divergence']
            regularize = setting['regularize']
            axis = setting['axis']

            result = None if divergence is None else divergence(a)

            if regularize is not None:
                loss += regularize(result)

            # 計測用、head毎の値は末尾の軸、
            # それ以外の軸はバッチ軸など平均をとる
            if axis is not None and result is not None:
                result = np.mean(result, axis=axis)

            setting['result'] = result

        return loss

    def _record_result(self, setting):
        result = setting['result']
        if result is None:
            return

        if not setting['round_robin']:
            setting['record'].append(result.copy())
            return

        rr_index = setting['record_rr_index']
        selected = setting['record_schedule'][rr_index]

        head_pairs = selected
        values = result.reshape(-1)

        if setting['record_buffer'] is None:
            setting['record_buffer'] = np.empty(
                len(setting['record_index']), dtype=values.dtype)

        for pair, value in zip(head_pairs, values):
            slot_index = setting['record_index'][pair]
            setting['record_buffer'][slot_index] = value
            setting['record_seen'].add(slot_index)

        setting['record_rr_index'] = (
            rr_index + 1
        ) % len(setting['record_schedule'])

        if len(setting['record_seen']) == len(setting['record_index']):
            setting['record'].append(setting['record_buffer'].copy())
            setting['record_seen'].clear()

    def __backward__(self, gl):
        # backwardまで到達したforwardだけを学習記録として残す
        for setting in self.settings:
            self._record_result(setting)

        if all(setting['regularize'] is None for setting in self.settings):
            return 0

        ga = 0

        for setting in self.settings:
            divergence = setting['divergence']
            regularize = setting['regularize']
            scheduler = setting['scheduler']
            eta = setting['eta']

            if scheduler is not None:
                eta = eta * scheduler(self.iter)

            if regularize is None or eta == 0:
                continue

            gy = regularize.backward(gl)
            gx = divergence.backward(gy)
            ga += gx * eta

        self.iter += 1

        return ga

