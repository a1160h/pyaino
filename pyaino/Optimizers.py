# Optimizers
# 2025.10.15 A.Inoue

from pyaino.Config import *
import copy

#### 最適化関数の共通機能 ############################################
class OptimizerBase:
    '''
    最適化関数の基底クラスであって、以下の関連機能を纏め、共通機能を付与する
      weight decay
      gradient clipping
      learning rate scheduler
      dynamic adjustment of learning rate
      spectral normalization(更新時にパラメタのリプシッツ制約を行う)   
      weight clipping

    '''
    
    def __init__(self, **kwargs):
        # biasか？ -> 以下対象外: weight decay, weight clipping, spectral normalization 　　
        self.bias = kwargs.pop('bias', False)         
        
        # weight decay : 0で無効
        self.w_decay = 0 if self.bias else kwargs.pop('w_decay', 0)  
        
        # 勾配クリッピングは更新の際の指定による
        self.gradient_clipping = GradientClipping() 

        # 以下はlearning rate scheduler, 
        self.scheduler = None
        if kwargs.pop('anneal', False) and self.scheduler is None:
            self.scheduler = Annealing(**kwargs) 
        if kwargs.pop('exponential_decay', False) and self.scheduler is None:
            self.scheduler = ExponentialDecay(**kwargs)
        if kwargs.pop('cos_decay', False) and self.scheduler is None:
            self.scheduler = CosineDecay(**kwargs)
        if kwargs.pop('linear_grow_cos_decay', False) and self.scheduler is None:
            self.scheduler = LinearGrowCosineDecay(**kwargs)
        if kwargs.pop('sin_grow_cos_decay', False) and self.scheduler is None:
            self.scheduler = SineGrowCosineDecay(**kwargs)
            
        # DynamicAdjust(学習率をindicatorの変化に応じて動的に調整)
        if kwargs.pop('dynamic',   False):
            self.dynamic_adjust = DynamicAdjust()
        else:
            self.dynamic_adjust = None

        # SpectralNormalizationのksnを数値で指定
        spctrnorm = kwargs.pop('spctrnorm', None)
        if spctrnorm is not None and not self.bias:
            self.spctrnorm = SpectralNormalization(power_iterations=spctrnorm)
        else:
            self.spctrnorm = None

        # Weight Clipping
        wghtclpng  = kwargs.pop('wghtclpng',  None)
        wghtclpng2 = kwargs.pop('wghtclpng2', None)
        self.wghtclpng = None
        if self.bias:
            pass
        elif wghtclpng  is not None: # wghtclpng > wghtclpng2
            self.wghtclpng = WeightClipping(wghtclpng)
        elif wghtclpng2 is not None:
            self.wghtclpng = WeightClipping2(wghtclpng2)
            
        self.iter = 0    
            
    def update(self, parameter, gradient, eta=0.001, **kwargs): # parameter追加
        # w_decay操作を勾配による更新とは独立に行う
        if self.w_decay!=0:
            parameter *= (1 - eta * self.w_decay)

        # 勾配クリッピング 無指定では__call__()しない
        if any(kwargs): # kwargsは勾配クリッピングの指定のみ
            eta *= self.gradient_clipping(gradient, **kwargs)
            
        # 学習率調整の諸手法 learning rate scheduler　　
        if self.scheduler is not None: 
            eta *= self.scheduler(self.iter)

        # 学習率の動的調整
        if self.dynamic_adjust is not None:
            g_l2n = np.sqrt(np.sum(np.square(gradient))) # 仮実装20250320AI
            eta *= self.dynamic_adjust(g_l2n)            #

        # 最適化関数に従い勾配にもとづく更新
        self.eta = eta
        self.iter += 1 # AdamTでも使う
        parameter -= self.__call__(gradient, eta)

        # Spectral Normalization 
        if self.spctrnorm is not None: 
            self.spctrnorm(parameter)

        # Weight Clipping
        if self.wghtclpng is not None:
            self.wghtclpng(parameter)
            
        return parameter

class GradientClipping:
    def __init__(self):
        self.g_l2n_ppl = 1.0
        
    def __call__(self, gradient, **kwargs):
        g_clip   = kwargs.pop('g_clip',   None)
        g_clip_a = kwargs.pop('g_clip_a', None)
        g_clip_b = kwargs.pop('g_clip_b', None)
        g_l2n = np.sqrt(np.sum(np.square(gradient)))           # 勾配のL2ノルム
        if g_clip is not None:     # g_clip が有効 
            rate = g_clip / (g_l2n + 1e-6) # 上限値に対する逆比(1で丁度、大きいほど余裕あり)
            return rate if rate < 1.0 else 1.0
        
        # 以下 g_clip_a または g_clip_b の指定あり
        if g_clip_a is not None:   # g_clip_a が有効
            rate = g_clip_a * self.g_l2n_ppl / (g_l2n + 1e-6)
            if rate < 1.0:
                pass
            else:
                self.g_l2n_ppl -= 0.01 * (self.g_l2n_ppl - g_l2n) # 移動平均 advanced 用
                rate = 1.0
            return rate
        
        # g_clip_b が有効
        rate = 1 / (g_l2n + 1 / g_clip_b + 1e-6)
        return rate

class Annealing:
    def __init__(self, **kwargs):
        self.warmup   = kwargs.pop('warmup',      100)  # warmupサイクル数
        self.max_iter = kwargs.pop('max_iter',   1000)  # 着地点
        self.hold     = kwargs.pop('hold',        100)  # 温度保持サイクル数

    def __call__(self, iteration):
        z1 = (iteration+1) / (self.warmup + 1)   
        z2 = (iteration - self.max_iter + 1) / (self.warmup + self.hold - self.max_iter) 
        z  = max(min(1, z1, z2), 1/self.max_iter)
        return z

class ExponentialDecay:
    """ 指数逓減関数 """
    def __init__(self, **kwargs):
        self.decay = kwargs.pop('decay_rate',    1.0) 
        self.start = kwargs.pop('decay_start',     0)
        self.scale = kwargs.pop('time_scale',   1000)

    def __call__(self, iteration):
        if iteration < self.start:
            return 1.0
        exponent =  1 + (iteration - self.start) / self.scale
        decay = self.decay / (1 - (1 - self.decay) ** exponent)
        return decay

class CosineDecay:
    """　Cosine decay関数 """
    def __init__(self, **kwargs):
        self.rate  = kwargs.pop('decay_rate',    1.0)
        self.start = kwargs.pop('decay_start',     0)
        self.end   = kwargs.pop('decay_end',  100000) 
        
    def __call__(self, iteration):
        """ 1からrateまで指定ステップ数に応じてなだらかに減衰 """
        if iteration < self.start: # 減衰開始前
            return 1.0       
        elif iteration > self.end: # 減衰終了後
            return self.rate 
        # 減衰中はcos曲線で
        progress = (iteration - self.start) / (self.end - self.start)  # 時間軸
        cos2range = (1 - self.rate) * np.cos(np.pi * progress) # 振れ幅
        decay = 0.5 + 0.5 * self.rate + 0.5 * cos2range
        return decay

class LinearGrowCosineDecay:
    """　Linear grow & Cosine decay関数 """
    def __init__(self, **kwargs):
        self.warmup      = kwargs.pop('warmup',       1000)
        self.decay_rate  = kwargs.pop('decay_rate',    1.0)
        self.decay_start = kwargs.pop('decay_start',  1000)
        self.decay_end   = kwargs.pop('decay_end',  100000) 
   
    def __call__(self, iteration):
        if iteration < self.warmup:
            return iteration / self.warmup
        elif iteration < self.decay_start:
            return 1.0
        elif iteration > self.decay_end:
            return self.decay_rate
        progress = (iteration - self.decay_start) / (self.decay_end - self.decay_start)  # 時間軸
        cos2range = (1 - self.decay_rate) * np.cos(np.pi * progress) # 振れ幅
        decay = 0.5 + 0.5 * self.decay_rate + 0.5 * cos2range
        return decay
            

class SineGrowCosineDecay:
    """　Sine grow & Cosine decay関数 """
    def __init__(self, **kwargs):
        self.initial     = kwargs.pop('initial',       1.0)
        self.grow_start  = kwargs.pop('grow_start',      0)
        self.grow_end    = kwargs.pop('grow_end',     1000)
        self.decay_rate  = kwargs.pop('decay_rate',    1.0)
        self.decay_start = kwargs.pop('decay_start',  1000)
        self.decay_end   = kwargs.pop('decay_end',  100000) 
        
    def __call__(self, iteration):
        """ 指定ステップに応じて、initialから1まで成長した後、rateまでなだらかに減衰 """
        if iteration < self.grow_start:  # 成長開始前　　
            return self.initial
        if iteration < self.grow_end:
            progress = (iteration - self.grow_start) / (self.grow_end - self.grow_start)
            sin2range = (1 - self.initial) * np.sin(np.pi * (progress -0.5))
            grow = 0.5 + 0.5 * self.initial + 0.5 * sin2range
            return grow
        if iteration < self.decay_start: # 減衰開始前
            return 1.0       
        if iteration > self.decay_end:   # 減衰終了後
            return self.decay_rate 
        progress = (iteration - self.decay_start) / (self.decay_end - self.decay_start)  # 時間軸
        cos2range = (1 - self.decay_rate) * np.cos(np.pi * progress) # 振れ幅
        decay = 0.5 + 0.5 * self.decay_rate + 0.5 * cos2range
        return decay

class DynamicAdjust: # 仮実装20250320AI
    def __init__(self, rate=0.5):
        self.rate = rate
        self.adjust = 1
        self.memory = 1

    def __call__(self, indicator=None):
        if indicator is not None and indicator > self.memory:
            self.adjust *= self.rate
        else:
            self.adjust = 1
        self.memory = indicator
        return self.adjust
    
#### 以下、各種最適化関数 ##################################################
class SGD(OptimizerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, gradient, eta):
        return eta * gradient

class Momentum(OptimizerBase):
    def __init__(self, **kwargs): 
        self.momentum = kwargs.pop('momentum', 0.9)
        super().__init__(**kwargs)
        self.vlcty = None
        
    def __call__(self, gradient, eta):
        if self.vlcty is None:
            self.vlcty = np.zeros_like(gradient)
        self.vlcty -= (1 - self.momentum) * (self.vlcty - gradient) # 移動平均　
        return eta * self.vlcty
        
class Momentum2(OptimizerBase):
    def __init__(self, **kwargs): 
        self.momentum = kwargs.pop('momentum', 0.9)
        super().__init__(**kwargs)
        self.vlcty = None
        
    def __call__(self, gradient, eta):
        if self.vlcty is None:
            self.vlcty = np.zeros_like(gradient)
        self.vlcty = - eta * gradient + self.momentum * self.vlcty # 次回の前回更新量
        return - self.vlcty
    
class RMSProp(OptimizerBase):
    def __init__(self, **kwargs): 
        self.decayrate = kwargs.pop('decayrate', 0.9)
        super().__init__(**kwargs)
        self.hstry = None

    def __call__(self, gradient, eta):
        if self.hstry is None:
            self.hstry = np.ones_like(gradient)
        self.hstry -= (1 - self.decayrate) * (self.hstry - gradient ** 2) # 移動平均
        return eta * gradient / (np.sqrt(self.hstry) + 1e-7)
    
class AdaGrad(OptimizerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hstry = None

    def __call__(self, gradient, eta):
        if self.hstry is None:
            self.hstry = np.zeros_like(gradient)
        self.hstry += gradient ** 2
        return eta * gradient / (np.sqrt(self.hstry) + 1e-7)

class Adam(OptimizerBase):
    def __init__(self, **kwargs): 
        self.momentum  = kwargs.pop('momentum',  0.9)
        self.decayrate = kwargs.pop('decayrate', 0.999)
        super().__init__(**kwargs)
        self.vlcty = None
        self.hstry = None

    def __call__(self, gradient, eta):
        if self.vlcty is None:
            self.vlcty = np.zeros_like(gradient)
        if self.hstry is None:
            self.hstry = np.zeros_like(gradient) # 初期値1->0変更20250414AI            
         
        self.vlcty -= (1 - self.momentum)  * (self.vlcty - gradient)
        self.hstry -= (1 - self.decayrate) * (self.hstry - gradient ** 2)

        return eta * self.vlcty / (np.sqrt(self.hstry) + 1e-7)

class AdamT(OptimizerBase):
    def __init__(self, **kwargs): 
        self.momentum  = kwargs.pop('momentum',  0.9)   # beta1
        self.decayrate = kwargs.pop('decayrate', 0.999) # beta2
        super().__init__(**kwargs)
        self.vlcty = None
        self.hstry = None

    def __call__(self, gradient, eta):
        if self.vlcty is None:
            self.vlcty = np.zeros_like(gradient)
        if self.hstry is None:
            self.hstry = np.zeros_like(gradient) # 初期値1->0変更20250414AI
         
        self.vlcty -= (1 - self.momentum)  * (self.vlcty - gradient)
        self.hstry -= (1 - self.decayrate) * (self.hstry - gradient ** 2)

        #self.vlcty = self.momentum  * self.vlcty + (1 - self.momentum)  * gradient
        #self.hstry = self.decayrate * self.hstry + (1 - self.decayrate) * gradient * gradient

        vlcty_hat = self.vlcty / (1 - self.momentum  ** self.iter)
        hstry_hat = self.hstry / (1 - self.decayrate ** self.iter)

        return eta * vlcty_hat / (np.sqrt(hstry_hat) + 1e-8)

class AdamTS(OptimizerBase):
    def __init__(self, **kwargs): 
        self.momentum    = kwargs.pop('momentum',  0.9)   # beta1
        self.decayrate   = kwargs.pop('decayrate', 0.999) # beta2
        self.switch_time = kwargs.pop('switch_time', 100)
        super().__init__(**kwargs)
        self.vlcty = None
        self.hstry = None
        self.t = 0

    def __call__(self, gradient, eta):
        if self.vlcty is None:
            self.vlcty = np.zeros_like(gradient)
        if self.hstry is None:
            self.hstry = np.zeros_like(gradient) # 初期値1->0変更20250414AI
         
        self.vlcty -= (1 - self.momentum)  * (self.vlcty - gradient)
        self.hstry -= (1 - self.decayrate) * (self.hstry - gradient ** 2)

        #self.vlcty = self.momentum  * self.vlcty + (1 - self.momentum)  * gradient
        #self.hstry = self.decayrate * self.hstry + (1 - self.decayrate) * gradient * gradient

        vlcty_hat = self.vlcty / (1 - self.momentum  ** self.t + 1)
        hstry_hat = self.hstry / (1 - self.decayrate ** self.t + 1)

        if self.iter > self.switch_time:
            self.t +=1

        return eta * vlcty_hat / (np.sqrt(hstry_hat) + 1e-8)

class AdamTracking(OptimizerBase):
    def __init__(self, **kwargs): 
        self.momentum  = kwargs.pop('momentum',  0.9)   # beta1
        self.decayrate = kwargs.pop('decayrate', 0.999) # beta2
        super().__init__(**kwargs)
        self.vlcty = None
        self.hstry = None
        self.t = 0
        self.diff_trck_vlcty = []
        self.diff_trck_hstry = []

    def __call__(self, gradient, eta):
        if self.vlcty is None:
            self.vlcty = np.zeros_like(gradient)
        if self.hstry is None:
            self.hstry = np.zeros_like(gradient) # 初期値1->0変更20250414AI
         
        prev_vlcty = self.vlcty.copy()
        prev_hstry = self.hstry.copy()

        self.t += 1
        #gradient2 = gradient.astype('f8')
        self.vlcty -= (1 - self.momentum)  * (self.vlcty - gradient)
        self.hstry -= (1 - self.decayrate) * (self.hstry - np.square(gradient))

        self.diff_trck_vlcty.append(np.mean(np.abs(self.vlcty - prev_vlcty)))
        self.diff_trck_hstry.append(np.mean(np.abs(self.hstry - prev_hstry)))

        vlcty_hat = self.vlcty / (1 - np.power(self.momentum,  self.t))
        hstry_hat = self.hstry / (1 - np.power(self.decayrate, self.t))

        return eta * vlcty_hat / (np.sqrt(hstry_hat) + 1e-8)

class Adam2(OptimizerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time_attn = kwargs.pop('time_attn',      True)
        decayrate = 0.999 if self.time_attn else 0.9
        self.momentum  = kwargs.pop('momentum',        0.9)
        self.decayrate = kwargs.pop('decayrate', decayrate)
        self.vlcty = None  # First moment
        self.hstry = None  # Second moment
        self.t = 0         # Time step

    def __call__(self, gradient, eta):
        if self.vlcty is None:
            self.vlcty = np.zeros_like(gradient)
        if self.hstry is None:
            self.hstry = np.zeros_like(gradient)
        self.t += 1
        self.vlcty = self.momentum  * self.vlcty + (1 - self.momentum)  * gradient
        self.hstry = self.decayrate * self.hstry + (1 - self.decayrate) * (gradient ** 2)
        if self.time_attn:
            vlcty_hat = self.vlcty / (1 - self.momentum  ** self.t)
            hstry_hat = self.hstry / (1 - self.decayrate ** self.t)
        else:
            vlcty_hat = self.vlcty
            hstry_hat = self.hstry
        return eta * vlcty_hat / (np.sqrt(hstry_hat) + 1e-8)

class Adam2_bkup:
    def __init__(self, **kwargs): 
        self.momentum  = kwargs.pop('momentum',  0.9)
        self.decayrate = kwargs.pop('decayrate', 0.99)
        self.vlcty = None
        self.hstry = None

    def __call__(self, gradient, eta):
        if self.vlcty is None:
            self.vlcty = np.zeros_like(gradient)
        if self.hstry is None:
            self.hstry = np.ones_like(gradient)
         
        self.vlcty -= (1 - self.momentum)  * (self.vlcty - gradient)
        self.hstry -= (1 - self.decayrate) * (self.hstry - gradient ** 2)

        vlcty = self.vlcty / (1 - self.momentum)
        hstry = self.hstry / (1 - self.decayrate)
        
        return eta * vlcty / (np.sqrt(hstry) + 1e-7)

class Momentum3:
    def __init__(self, **kwargs): 
        print('従来互換')
        self.momentum = kwargs.pop('momentum', 0.9)
        self.vlcty = None
        
    def __call__(self, gradient, eta):
        if self.vlcty is None:
            self.vlcty = np.zeros_like(gradient)
        self.vlcty = self.momentum * self.vlcty - eta * gradient
        return -self.vlcty

# 以下は従来との互換性のため
class RMSProp2:
    def __init__(self, **kwargs):
        print('従来互換')
        self.decayrate = kwargs.pop('decayrate', 0.9)
        self.hstry = None

    def __call__(self, gradient, eta):
        if self.hstry is None:
            self.hstry = np.zeros_like(gradient)
        self.hstry = self.decayrate * self.hstry \
                     + (1 - self.decayrate) * (gradient ** 2)              
        return eta * gradient / (np.sqrt(self.hstry) + 1e-7)
    
class Adam3:
    def __init__(self, **kwargs): 
        print('従来互換')
        self.momentum  = kwargs.pop('momentum',  0.9)
        self.decayrate = kwargs.pop('decayrate', 0.999)
        self.hstry = None
        self.vlcty = None
        self.iter = 0

    def __call__(self, gradient, eta):
        if self.hstry is None:
            self.hstry = np.zeros_like(gradient)
        if self.vlcty is None:
            self.vlcty = np.zeros_like(gradient)
        self.iter += 1    
        eta_t = 0.1 * eta * np.sqrt(1.0 - self.decayrate ** self.iter) \
                                 / (1.0 - self.momentum  ** self.iter)
        self.hstry += (1 - self.momentum)  * (gradient - self.hstry)
        self.vlcty += (1 - self.decayrate) * (gradient ** 2 - self.vlcty)
        return eta_t * self.hstry / (np.sqrt(self.vlcty) + 1e-7)

#### 最適化に関連する、その他の関数 ####################################
class SpectralNormalization:
    """ 入力WにSpectral Normalizationを適用して更新する """
    def __init__(self, power_iterations=1, alpha=1.0, eps=1e-12):
        # power_iterations: パワー反復の回数（デフォルトは1）
        self.power_iterations = int(power_iterations)
        print(self.__class__.__name__, power_iterations)
        self.u = None
        self.alpha = alpha # 緩和係数 0.5～0.8で表現力を残しつつ安定化
        self.eps = eps
   
    def __call__(self, W):
        """ W(重み行列)を更新する """
        m, n = W.shape
        if self.u is None: 
            self.u = np.random.randn(n, 1).astype(W.dtype) # 初期のランダムベクトル u
        u = self.u    
        for _ in range(self.power_iterations):
            v = np.dot(W, u)
            v = v / (np.linalg.norm(v, axis=0, keepdims=True) + self.eps) # vをノルムで正規化
            u = np.dot(W.T, v)
            u = u / (np.linalg.norm(u, axis=0, keepdims=True) + self.eps) # uをノルムで正規化
        sigma = np.dot(v.T, np.dot(W, u))    # Wのスペクトラルノルム
        W /= sigma ** self.alpha + self.eps  # Wを更新
        self.u = u
        #print('sigma =', sigma)
        

def spectral_normalization(W, power_iterations=1):
    return SpectralNormalization(power_iterations)(W)
    
class SpectralNormalization2:
    def __init__(self, ksn=2.5, power_iterations=1):
        self.u = None
        self.ksn = ksn
        self.power_iterations = power_iterations
        print('spectral_normalization ksn =', ksn)

    def init_parameters(self, n):    
        self.u = np.random.randn(n).astype('f4')
        print(self.__class__, 'init_parameters', n)

    def __call__(self, W):
        if self.u is None:
            self.init_parameters(W.shape[0])
        u = self.u
        ksn = self.ksn
        
        for i in range(self.power_iterations):
            # v = l2normalize of W.T @ u
            Wu = np.dot(W.T, u)
            l2n_Wu = np.sqrt(np.sum(Wu**2))
            v = Wu / l2n_Wu
            # u = l2normalize of W @ v
            Wv = np.dot(W, v)
            l2n_Wv = np.sqrt(np.sum(Wv**2))
            u = Wv / l2n_Wv
            
        # sigma = u.T @ W @ v 
        sigma = np.dot(u.T, np.dot(W, v))
        K = 1 if ksn > sigma else ksn / sigma 
        W = W * K 
        self.u = u
        return W

class WeightClipping:
    def __init__(self, clip=(-1, 1)):
        ''' clip W by specified range '''
        if type(clip) in (tuple, list) and len(clip)==2:
            self.clip = clip
        elif type(clip) in (float, int):
            self.clip = -clip, clip
        else:
            raise Exception('Wrong specification of clip value.')
        print(self.__class__.__name__, clip)

    def __call__(self, W):
        """ Wを更新する """
        clip_l = self.clip[0]
        clip_h = self.clip[1]
        W[...] = np.where(W < clip_l, clip_l, W)
        W[...] = np.where(W > clip_h, clip_h, W)
        
class WeightClipping2:
    def __init__(self, clip=10):
        ''' clip W by L2norm of W '''
        if type(clip) in (float, int):
            self.clip = clip
        else:
            raise Exception('Wrong specification of clip value.')
        print(self.__class__.__name__, clip)

    def __call__(self, W):
        """ Wを更新する """
        #w_l2n = np.sqrt(np.sum(W **2))    # Wのl2ノルムだがW**2で新たな配列が作られる
        w_l2n = np.linalg.norm(W) 
        rate = self.clip / (w_l2n + 1e-6)
        #W[...] = W * rate if rate < 1 else W # W*rateが新たに作られるためインプレース演算にならない　
        W[...] = np.where(rate < 1, W * rate, W)
        

# Example usage
if __name__ == "__main__":

    from pyaino import Functions as F
    import matplotlib.pyplot as plt

    func = F.Square()

    opts = SGD, Momentum, RMSProp, AdaGrad, Adam, AdamT, AdamTS#, Adam2
    eta = 0.01

    plt.figure(figsize=(10, 6))
    for o in opts:
        opt = o()
        name = opt.__class__.__name__
        print(name)
        x = np.array([[5.0]])
        trajectory = [x.reshape(-1).tolist()]
        for i in range(200):
            y  = func(x)
            gx = func.backward()
            x = opt.update(x, gx, eta=eta)
            trajectory.append(x.reshape(-1).tolist())
        plt.plot(trajectory, label=name)

    plt.xlabel('Iteration')
    plt.ylabel('x')
    plt.title('Optimizer Behavior for Minimizing $f(x) = x^2$')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

    ###################################################################
    funcs = (Annealing(),
             ExponentialDecay(decay_start=50, decay_rate=0.1, time_scale=100),
             CosineDecay(decay_rate=0.11, decay_start=100, decay_end=500),
             LinearGrowCosineDecay(warmup=110, decay_rate=0.12, decay_start=130, decay_end=500),
             SineGrowCosineDecay(initial=0.2, grow_start=100, grow_end=200, decay_rate=0.13, decay_start=300, decay_end=500)
             )
    eta = 0.01
    iters = range(0, 1000)
    plt.figure(figsize=(10, 6))
    for f in funcs:
        name = f.__class__.__name__
        print(name)
        etas = []
        for t in iters:
            decay = f(t)
            decay = float(decay)
            etas.append(decay)

        plt.plot(iters, etas, label=name)
        
    plt.title('Behaviers of decay functions')
    plt.legend()
    plt.show()
        
