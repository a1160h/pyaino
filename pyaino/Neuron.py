# Neuron
# 2025.11.14 A.Inoue

import copy
import warnings
from pyaino.Config import *
from pyaino.nucleus import Function, CompositFunction
from pyaino import Activators
from pyaino import Optimizers
from pyaino import common_function as cf
from pyaino import LossFunctions as lf
from pyaino import Functions as F
from pyaino.nucleus import HDArray
from pyaino import Regularizers
from pyaino.Initializer import init_weight

class Sequential:
    """ 複数の層を積み上げて一括して扱う """
    def __init__(self, *layers, **kwargs):
        self.layers = [l for l in layers]
        print(self.layers)

    def forward(self, x, **kwargs):
        y = x
        for l in self.layers:
            y = l.forward(y, **kwargs)
        return y

    def backward(self, gy=None):
        if gy is None:
            gx = self.layers[-1].backward()
        else:
            gx = self.layers[-1].backward(gy)
        for l in reversed(self.layers[:-1]):
            gx = l.backward(gx)
        return gx

    def update(self, eta=0.001, **kwargs):
        for l in self.layers:
            if hasattr(l, 'update'): 
                l.update(eta=eta, **kwargs)
                
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def summary(self):
        for l in self.layers:
            print(l.__class__.__name__, end=' ')
        if hasattr(l, 'config'):
            print(l.config)
        else:
            print('\n')

#### ニューロンの基本機能 ##############################################
class AffineParameters:
    """ Affine変換のパラメータ管理 """
    def __init__(self, layer, **kwargs):
        print('Initialize', self.__class__.__name__)
        self.layer      = layer
        self.bias       = kwargs.pop('bias',        True) # dot_linearのbias有無
        optimize        = kwargs.get('optimize',   'SGD') # kwargsに残してBNに渡す
        self.width      = kwargs.pop('width',       None) # 重みの初期値の広がりを指定
        self.debug_mode = kwargs.pop('debug_mode', False) # 重みを一律に初期化
        self.scale      = kwargs.pop('scale',      False) # ReParameteraization

        self.w, self.b, self.gamma = None, None, None
        self.grad_w, self.grad_b, self.ggamma = None, None, None
        self.w_bkup, self.b_bkup, self.gamma_bkup = None, None, None

        self.optimizer_w = cf.eval_in_module(optimize, Optimizers, **kwargs)  # 最適化関数
        if self.bias:
            self.optimizer_b = cf.eval_in_module(optimize, Optimizers, bias=True, **kwargs)
        if self.scale:
            self.optimizer_g = cf.eval_in_module(optimize, Optimizers, bias=True, **kwargs)

    def __call__(self):
        if self.w is None:
            self.init_parameter()
        return self.w, self.b, self.gamma

    def init_parameter(self):
        m, n = self.layer.get_parameter_size()
        if m is None or n is None:
            raise Exception('Configuration is not fixed.', self.__class__.__name__)
        if hasattr(self.layer, 'activator'):
            activator = self.layer.activator
        else:
            activator = None
        self.w = init_weight((m, n),
                             width=self.width,
                             activator=activator,
                             debug_mode=self.debug_mode)        
        if self.bias:
            self.b = np.zeros(n, dtype=Config.dtype)
        if self.scale:
            self.gamma = np.array(1.0, dtype=Config.dtype)
        
    def update(self, eta=0.001, bkup=False, **kwargs):
        if bkup:
            self.backup()
        self.optimizer_w.update(self.w, self.grad_w, eta, **kwargs) # 戻り値=更新量
        if self.bias:
            self.optimizer_b.update(self.b, self.grad_b, eta, **kwargs) # 戻り値=更新量
        if self.scale:
            self.optimizer_g.update(self.gamma, self.ggamma, eta, **kwargs)
        
    def backup(self):
        if self.w_bkup is not None:
            self.w_bkup[...] = self.w
        else:
            self.w_bkup = copy.deepcopy(self.w)
        if self.bias:
            if self.b_bkup is not None:
                self.b_bkup[...] = self.b
            else:
                self.b_bkup = copy.deepcopy(self.b)
        if self.scale:        
            if self.gamma_bkup is not None:
                self.gamma_bkup[...] = self.gamma
            else:
                self.gamma_bkup = copy.deepcopy(self.gamma)
        
    def recover(self):
        self.w[...] = self.w_bkup
        if self.bias:
            self.b[...] = self.b_bkup
        if self.scale:
            self.gamma[...] = self.gamma_bkup

    def accommodate(self):
        if self.config[1] <= self.w.shape[1]:
            return
        print(self.__class__.__name__, 'expand the size of w to accommodate new vocabulary.')
        m, n = self.config
        xpcn = n - self.w.shape[1] # 拡張する列数
        center_w = np.mean(self.w, axis=1, keepdims=True)
        new_colums = center_w + np.random.normal(0, 0.01, size=(m, xpcn), dtype=Config.dtype)
        print('new colums of w =', new_colums.shape)
        self.w = np.concatenate([self.w, new_colums], axis=1)
        if self.bias:
            center_b = np.mean(self.b)
            new_bias = center_b + np.random.normal(0, 0.01, size=(xpcn,), dtype=Config.dtype)
            print('new bias =', new_bias.shape)
            self.b = np.concatenate([self.b, new_bias])

    def set_gradient(self, *grads, flush=True):
        if flush:
            self.grad_w = grads[0]
        else:
            self.grad_w += grads[0]
        if self.bias:
            if flush:
                self.grad_b = grads[1]
            else:
                self.grad_b += grads[1]
        if self.scale:
            if flush:
                self.ggamma = grads[-1]
            else:
                self.ggamma += grads[-1]

    def flush_gradient(self):
        self.grad_w = np.zeros_like(self.w, dtype=Config.dtype)
        if self.bias:
            self.grad_b = np.zeros_like(self.b, dtype=Config.dtype)
        if self.scale:
            self.ggamma = np.array(1.0, dtype=Config.dtype)

#### ニューロンの基本機能 ##############################################
class LinearLayer(Function):
    """ ニューロンの基本機能(Pytorch互換機能提供) """
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        if   len(configuration) == 2:
            m, n = configuration
        elif len(configuration) == 1:
            m = None; n, = configuration
        else:
            m, n = None, None
        self.config = m, n                                # m:入力幅、n:ニューロン数
        print('Initialize', self.__class__.__name__, self.config)
        matmul          = kwargs.pop('matmul',     False) # MatMulLinearを使う
        self.bias       = kwargs.get('bias',        True) # dot_linearのbias有無
        self.scale      = kwargs.get('scale',      False) # ReParameteraization
        self.parameters = AffineParameters(self, **kwargs)
        self.dot_linear = F.ScaleDotLinear(matmul, self.bias, self.scale)

    def update(self, eta=0.001, **kwargs):
        self.parameters.update(eta=eta, **kwargs) 
        
    def fix_configuration(self, shape):
        if self.dot_linear.__class__.__name__=='MatMulLinear':
            m = shape[-1]
        else:
            m = 1
            for i in shape[1:]:                       # バッチ軸以外の積
                m *= i
        self.config = m, self.config[1]
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def __forward__(self, x, **kwargs):           # kwargsは使わない
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        w, b, gamma = self.parameters()    
        y = self.dot_linear.forward(x, w, b, gamma)
        return y 
        
    def __backward__(self, gy):
        gx, gw, gb, ggamma = self.dot_linear.backward(gy)
        self.parameters.set_gradient(gw, gb, ggamma) 
        return gx

    def get_parameter_size(self):
        return self.config

    def accommodate(self):
        self.parameters.accommodate()
            
#### ニューロンの基本機能 ##############################################
class LinearLayerCrossEntropy(LinearLayer):
    """ Softmaxそして損失まで一気に算出するニューロンの基本機能 """
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.tile_size = kwargs.pop('tile_size', None) #
        self.selector = TileTargetScanner()
       
    def __forward__(self, x, t=None, **kwargs):       # kwargsは使わない
        
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        w, b, gamma = self.parameters()

        m, n = self.config
        if self.tile_size is None:
            self.tile_size = n

        leading_shape = x.shape[:-1] # (B,)や(B,T)
        self.leading_size = 1.0      # leading_shapeの積
        for i in leading_shape:
            self.leading_size *= i 
        
        # 全体の最大logits値とその位置の初期値(バッチサイズ分並べる)
        max_logit = np.full(leading_shape, -np.inf, dtype=x.dtype) # 現時点の最大logit
        max_index = np.full(leading_shape, -1)                     # その語彙ID
        sum_exp = np.zeros(leading_shape, dtype=x.dtype)        # 逐次 exp 累積
        if t is not None:
            zt = np.zeros(leading_shape, dtype=x.dtype)    # 正解値の指すlogitのロジット

        for start in range(0, n, self.tile_size):
            # タイル毎にlogitsを算出
            end = min(start + self.tile_size, n)
            tile_w = w[:, start:end]
            tile_b = b[start:end]
            tile_z = self.dot_linear.forward(x, tile_w, tile_b, gamma)      # (B, Vt)

            last_max_logit = max_logit.copy() # 更新前のmax_logit  

            # タイル内の最大位置を求め、そのlogits値を得る
            tile_max_index = np.argmax(tile_z, axis=-1)            # (B,)
            tile_max_logit = (np.take_along_axis(tile_z, tile_max_index[..., None], axis=-1)
                              .squeeze(-1))

            # 全体最大を更新(タイルの最大が全体の最大より大きいものについて処理)
            mask = tile_max_logit > max_logit
            if mask.any():
                max_logit[mask] = tile_max_logit[mask]
                max_index[mask] = start + tile_max_index[mask]  # 全体での位置=語彙ID 
            # 最新と以前の最大値の補正をしながらsum_exp を更新
            sum_exp = (sum_exp * np.exp(last_max_logit - max_logit) # 補正項　
                     + np.sum(np.exp(tile_z - max_logit[..., None]), axis=-1)) # 更新値
            # tがtileに含まれる場合だけztを更新(zt:正解値tの指すlogit)
            if t is not None:
                zt = self.selector.gather(zt, t, tile_z, (start, end))

        # 予測だけ欲しい（推論）場合
        if t is None:
            return max_index, max_logit

        # 逆伝播用に保存
        self.sum_exp = sum_exp
        self.max_logit = max_logit

        # 確定した値で損失計算
        log_sum_exp = np.log(sum_exp) + max_logit     # 補正項
        loss = log_sum_exp - zt                       # CrossEntropy算出
        loss = np.sum(loss) / self.leading_size       # 平均
        # 学習時の返りは慣習的に (pred, loss) にしておく
        return max_index, loss
        
    def __backward__(self, *args): # argsは使わない
        m, n = self.config
        x, t = self.inputs 
        w, b, gamma = self.parameters()

        grad_w = np.zeros_like(w)
        grad_b = np.zeros_like(b)
        grad_x = np.zeros_like(x)
        ggamma = np.zeros_like(gamma) if self.scale else None

        for start in range(0, n, self.tile_size):
            # タイル毎にlogitsを算出
            end = min(start + self.tile_size, n)
            tile_w = w[:, start:end]
            tile_b = b[start:end]
            tile_z = self.dot_linear.forward(x, tile_w, tile_b, gamma) # (B, Vt)

            # Softmaxでlogit->確率 
            tile_y = np.exp(tile_z - self.max_logit[..., None]) / self.sum_exp[..., None]

            tile_gz = self.selector.scatter_add(tile_y.copy(), (start, end))   
            tile_gz /= self.leading_size    # 順伝播のloss/leading_sizeに合わせる
              
            # dot_linearの逆伝播
            tile_gx, tile_gw, tile_gb, tile_gg = self.dot_linear.backward(tile_gz)    
            grad_x += tile_gx
            grad_w[:, start:end] = tile_gw
            if self.bias:
                grad_b[start:end] = tile_gb
            if self.scale:
                ggamma += tile_gg
                
        self.parameters.set_gradient(grad_w, grad_b, ggamma)        
        return grad_x

class TileTargetScanner:
    """ LinearLayerCrossEntropyのtile処理用の選択器 """
    def __init__(self):
        # 環境に応じた関数の選択
        try:
            self.add_at = np.add.at
        except:
            try:
                self.add_at = np.scatter_add
            except:
                try:
                    self.add_at = np._cupyx.scatter_add
                except:
                    def f(x, y, z): # xのyの位置にzを加算する
                        for i, idx in enumerate(y):
                            x[idx] += z[i]
                    self.add_at = f        
    
    def gather2(self, zt, t, tile_z, window):
        """ tが処理窓内の時、tile_zからtに対応するlogitを選んでztにセットする """
        start, end = window
        t_in_tile = (start <= t) & (t < end) # バッチごとの該当非該当
        self.t = t  
        if not t_in_tile.any():
            return zt
        coords = np.where(t_in_tile)         # 対象バッチ番号
        idx_in_tile = (t[coords] - start).astype(np.int64)

        #print(start, '->', end, 'zt =', zt, 't =', t,
        #      'coords =', coords, '\ntile_z\n', tile_z, '\n', tile_z[coords])
        
        zt[coords] = (
            np.take_along_axis(tile_z[coords], idx_in_tile[..., None], axis=-1)
            .squeeze(-1))
        self.t = t
        return zt

    def gather(self, zt, t, tile_z, window):
        """ tが処理窓内の時、tile_zからtに対応するlogitを選んでztにセットする """
        start, end = window
        t_in_tile = (start <= t) & (t < end)
        self.t = t
        if not t_in_tile.any():
            return zt
        coords = np.where(t_in_tile)                # 先行軸の座標タプル
        idx_in_tile = (t[coords] - start).astype(np.int64)
        zt[coords] = tile_z[coords + (idx_in_tile,)] # 多次元インデクスとして結合
        return zt

    def scatter_add2(self, tile_gz, window, value=-1):
        """ tが処理窓内の時、tile_gzのtに対応する場所に値を加える """
        t = self.t
        start, end = window
        t_in_tile = (start <= t) & (t < end)
        if not t_in_tile.any():
            return tile_gz
        coords = np.where(t_in_tile)
        idx_in_tile = (t[coords] - start).astype(np.int64)
        index_tuple = coords + (idx_in_tile,)
        self.add_at(tile_gz, index_tuple, value)
        return tile_gz

    def scatter_add(self, tile_gz, window, value=-1):
        """ tが処理窓内の時、tile_gzのtに対応する場所に値を加える """
        t = self.t
        start, end = window
        t_in_tile = (start <= t) & (t < end)
        if not t_in_tile.any():
            return tile_gz
        coords = np.where(t_in_tile)
        idx_in_tile = (t[coords] - start).astype(np.int64)
        tile_gz[coords + (idx_in_tile,)] += value
        return tile_gz



#### ニューロン関連共通部分 ##############################################
# NeuronLayer(Affine),
# ConvLayer, DeconvLayer, MaskedExpansionLayer
# LatentLayer
class BaseLayer(Function): # ニューロンの基本機能 
    def __init__(self, **kwargs):
        super().__init__()
        print('Initialize', self.__class__.__name__, self.config)
        matmul          = kwargs.pop('matmul',       False) # MatMulLinearを使う 
        self.bias       = kwargs.get('bias',          True) # dot_linearのbias有無
        self.scale      = kwargs.get('scale',        False) # ReParameteraization
        self.parameters = AffineParameters(self, **kwargs)
        self.dot_linear = F.ScaleDotLinear(matmul, self.bias, self.scale)

        activate        = kwargs.pop('activate','Identity') # 恒等関数
        activate_option = kwargs.copy()                     # 残りは活性化のオプション
        self.activator  = cf.eval_in_module(activate, Activators, **activate_option)  # 活性化関数

        dropout         = kwargs.pop('dropout',      False) # ドロップアウト可否(forwardで指定)
        self.DO = Dropout() if dropout else None

        batchnorm       = kwargs.pop('batchnorm',    False) # バッチ正規化の適用有無
        layernorm       = kwargs.pop('layernorm',    False) # 層正規化の適用有無

        if batchnorm:
            self.Norm = BatchNormalization(**kwargs)
        elif layernorm:
            self.Norm = LayerNormalization(**kwargs)
        else:
            self.Norm = None

    def fix_configuration(self, shape):
        raise NotImplementedError('fix_configuration method for BaseLayer')
        
    def update(self, eta=0.001, **kwargs):
        self.parameters.update(eta=eta, **kwargs)
        if self.Norm:
            self.Norm.update(**kwargs)
        
    def flush_gradient(self):
        self.parameters.flush_gradient()

    def backup(self):
        self.parameters.backup()
        
    def recover(self):
        self.parameters.recover()
        
    def __forward__(self, y, *, train=False, dropout=0.0):
        if self.Norm:
            y = self.Norm.forward(y, train=train)          # バッチor層ノーマライゼーション
        y = self.activator.forward(y)                      # 活性化関数
        if self.DO:
            y = self.DO.forward(y, dropout=dropout)        # ドロップアウト
        return y    
        
    def __backward__(self, grad_y, **kwargs):
        if self.DO:
            grad_y = self.DO.backward(grad_y)              # ドロップアウト
        grad_y = self.activator.backward(grad_y, **kwargs) # 活性化関数
        if self.Norm:
            grad_y = self.Norm.backward(grad_y)            # バッチor層ノーマライゼーション
        return grad_y

    def get_parameter_size(self):    
        raise Exception('Invalid configuration')
    

#### Affine層 #######################################################
# m:上流のニューロン数、n:自身のニューロン数、activate:活性化関数、optimize:最適化、
# eta:学習係数、width:広がり係数、loss_f:損失関数、w_decay:L2正則化項の係数
class NeuronLayer(BaseLayer): # ニューロンの基本機能 
    def __init__(self, *configuration, **kwargs):
        if   len(configuration) == 2:
            m, n = configuration
        elif len(configuration) == 1:
            m = None; n, = configuration
        else:
            m, n = None, None
        self.config = m, n
        self.full_cnnt  = kwargs.pop('full_connection', False) # 全結合層を明示
        super().__init__(**kwargs)

    def fix_configuration(self, shape):
        m = 1
        for i in shape[1:]:                               # バッチ軸以外の積
            m *= i
        self.config = m, self.config[1]
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def get_parameter_size(self):
        m, n = self.config
        return m, n

    def __forward__(self, x, *, train=False, dropout=0.0):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        w, b, gamma = self.parameters()
            
        # 画像の全結合ではバッチ数維持し各ベクトル化、一方、時系列データではバッチ数と時系列長を維持
        m, n = self.config   # m:入力数、n:ニューロン数         
        # 画像、時系列データにも対応するためxをreshape、yの形状は (-1, n)
        y = self.dot_linear.forward(x.reshape(-1, m), w, b, gamma)
        y = super().__forward__(y, train=train, dropout=dropout)
        # 形状を入力と合致させる
        y_shape = (-1, n) if self.full_cnnt is True else (-1,) + x.shape[1:-1] + (n,)  
        y = y.reshape(*y_shape)
        return y 
        
    def __backward__(self, grad_y, flush=True, **kwargs):
        x, = self.inputs
        # 順伝播の際の形状 (-1, n) に
        m, n = self.config                                 # m:入力数、n:ニューロン数
        grad_y = grad_y.reshape(-1, n)                     # 畳込み層からの逆伝播で必要                   
        grad_y = super().__backward__(grad_y, **kwargs)
        grad_x, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)
        self.parameters.set_gradient(grad_w, grad_b, ggamma, flush=flush)
        grad_x = grad_x.reshape(*x.shape)                  # 形状を合致させる 　
        return grad_x

    def accommodate(self):
        self.parameters.accommodate()
    
### 畳み込み層 #####################################################
class Conv1dLayer(BaseLayer):
    # B:バッチサイズ, C:入力チャンネル数, Iw:入力画像幅
    # M:フィルタ数, Fw:フィルタ幅
    # stride:ストライド幅, pad:パディング幅
    # 出力チャンネル数=フィルタ数M, Ow:出力幅
    # w_decay:L2正則化項の係数
    
    def __init__(self, *configuration, **kwargs):
        if len(configuration) == 6:
            C, Iw, M, Fw, stride, pad = configuration
        if len(configuration) == 4:
            C = None; Iw = None; M, Fw, stride, pad = configuration
        if len(configuration) == 2:
            C = None; Iw = None; M, Fw = configuration; stride = 1; pad = 0 
        Ow = None
        self.params = self.config = C, Iw, M, Fw, stride, pad, Ow
        super().__init__(**kwargs)
        
    def fix_configuration(self, shape):
        C, Iw, M, Fw, stride, pad, Ow = self.config
        if len(shape) >= 2:
            Iw = shape[-1] 
            C = shape[1] if len(shape)==3 else 1
        elif C is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
           
        Ow = (Iw - Fw + 2*pad) // stride + 1   # 出力幅
        self.config = C, Iw, M, Fw, stride, pad, Ow

    def get_parameter_size(self):
        C, Iw, M, Fw, stride, pad, Ow = self.config
        m = C*Fw  # 入力チャネル数とフィルタサイズ
        n = M     # フィルタ数
        return m, n

    def __forward__(self, x, *, train=False, dropout=0.0):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        self.x_shape = x.shape 
        C, Iw, M, Fw, stride, pad, Ow = self.config
        x = x.reshape(-1, C, Iw)    # (B,C,Iw)  
        # '0'パディング B軸    C軸  Iw左Iw右
        x = np.pad(x, [(0,0),(0,0),(pad,pad)])
        # 入力画像を行列に変換 (B,C,Ih+2*pad,Iw+2*pad)->(C*Fh*Fw,B*Oh*Ow)
        cols = Vec2col(C, Iw+2*pad, M, Fw, stride)(x)
        # Affine変換: (B*Ow,C*Fw)×(C*Fw,M)->(B*Ow,M)
        w, b, gamma = self.parameters()    
        y = self.dot_linear.forward(cols, w, b, gamma)
        y = y.reshape(-1, Ow, M).transpose(0, 2, 1)       # u.shape=(B,M,Ow) 
        y = super().__forward__(y, train=train, dropout=dropout)
        return y
    
    def __backward__(self, grad_y, flush=True):
        C, Iw, M, Fw, stride, pad, Ow = self.config
        #grad_y = grad_y.reshape(-1, M, Ow)               # grad_y.shape=(B,M,Ow)
        grad_y = super().__backward__(grad_y) # 20250401AI            
        grad_y = grad_y.transpose(0, 2, 1).reshape(-1, M)   #grad_y.shape=(B*Ow,M)
        # Affineの逆伝播 grad_cols.shape=(B*Ow,C*Fw)
        grad_cols, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)
        self.parameters.set_gradient(grad_w, grad_b, ggamma, flush=flush)
        # 行列を画像に変換 (B*Oh*Ow,C*Fh*Fw)->(B,C,Ih,Iw)  　
        grad_x = Col2vec(M, Ow, C, Fw, stride)(grad_cols)
        # パディング分を外して元の画像データに戻す        
        grad_x = grad_x[:,:,pad:pad+Iw]
        grad_x = grad_x.reshape(self.x_shape)
        return grad_x

### 逆畳み込み層 #####################################################
class DeConv1dLayer(BaseLayer):
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # M:フィルタ数, Fh:フィルタ高さ, Fw:フィルタ幅
    # stride:ストライド幅, pad:パディング幅
    # 出力チャンネル数=フィルタ数M, Oh:出力高さ, Ow:出力幅
    # w_decay:L2正則化項の係数
    
    def __init__(self, *configuration, **kwargs):
        if len(configuration) == 6:
            C, Iw, M, Fw, stride, pad = configuration
        if len(configuration) == 4:
            C = None; Iw = None; M, Fw, stride, pad = configuration
        if len(configuration) == 2:
            C = None; Iw = None; M, Fw = configuration; stride = 1; pad = 0 
        Ow = None
        self.params = self.config = C, Iw, M, Fw, stride, pad, Ow
        super().__init__(**kwargs)

    def fix_configuration(self, shape):
        C, Iw, M, Fw, stride, pad, Ow = self.config
        if len(shape) >= 2:
            Iw = shape[-1] 
            C  = shape[1] if len(shape)==3 else 1
        elif C is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
        Ow = (Iw - 1) * stride + Fw - 2 * pad  # 出力幅
        self.config = C, Iw, M, Fw, stride, pad, Ow

    def get_parameter_size(self):
        C, Iw, M, Fw, stride, pad, Ow = self.config
        m = C              # 入力チャネル数「要注意」
        n = M*Fw           # フィルタ数とフィルタサイズ「要注意」
        return m, n

    def __forward__(self, x, *, train=False, dropout=0.0):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        self.x_shape = x.shape 
        C, Iw, M, Fw, stride, pad, Ow = self.config
        x = x.reshape(-1, C, Iw).transpose(0,2,1).reshape(-1,C) # (B*Iw,C)  
        # Affine変換 (B*Iw,C)×(C,M*Fw)->(B*Iw,M*Fw)   
        w, b, gamma = self.parameters()    
        cols = self.dot_linear.forward(x, w, b, gamma)
        # 行列を画像に変換 cols.T:(M*Fw,B*Iw)->(B,M,Ow)  　
        y = Col2vec(C, Iw, M, Fw, stride)(cols)
        # 画像調整 トリミング
        y = y[:,:,pad:pad+Ow]                     # y.shape=(B,M,Ow)
        y = super().__forward__(y, train=train, dropout=dropout)
        return y

    def __backward__(self, grad_y, flush=True):
        C, Iw, M, Fw, stride, pad, Ow = self.config
        #grad_y = grad_y.reshape(-1, M, Ow)       # grad_y.shape=(B,M,Ow)
        grad_y = super().__backward__(grad_y) # 20250401AI
        #  '0'パディング
        grad_y = np.pad(grad_y, [(0,0), (0,0), (pad, pad)])
        # 画像の勾配を行列に変換 grad_y.shape=(M*Fh*Fw,B*Ih*Iw)に変換
        grad_y = Vec2col(M, Ow+2*pad, C, Fw, stride)(grad_y)
        # Affineの逆伝播
        grad_x, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)
        self.parameters.set_gradient(grad_w, grad_b, ggamma, flush=flush)
        grad_x = grad_x.reshape(-1,Iw,C).transpose(0,2,1) # (B,C,Iw)
        grad_x = grad_x.reshape(self.x_shape)
        return grad_x

class Vec2col:
    """ vec.shape = (B, C, Iw) -> col.shape = (B*Ow, C*Fw) """
    def __init__(self, C, Iw, M, Fw, stride):
        # 出力画像のサイズ
        Ow = (Iw - Fw) // stride + 1        # 出力幅
        # パラメータをまとめる(class内での変数受渡しのため)
        self.params = (C, Iw, M, Fw, stride, Ow)

    def __call__(self, vec):
        C, Iw, M, Fw, stride, Ow = self.params
        B = vec.size // (C*Iw)
        col = np.empty((B, C, Fw, Ow), dtype=Config.dtype) # メモリ節約のためzerosでなくempty 
        # vecからstride毎のデータを取ってきて、colsにOwになるまで並べる
        # それをFh,Fwを満たすまで繰返す
        for fw in range(Fw):
            w_lim = fw + stride*Ow
            col[:,:,fw,:] = vec[:,:,fw:w_lim:stride]
        # 軸の入替と変形     B  Ow C  Fw 
        col = col.transpose(0, 3, 1, 2).reshape(B*Ow, C*Fw)
        return col

class Col2vec:
    """ col.shape = (B*Iw, M*Fw) -> vec.shape = (B, M, Ow)  """
    def __init__(self, C, Iw, M, Fw, stride):
        # 出力画像のサイズ
        Ow = (Iw - 1) * stride + Fw 
        # パラメータをまとめる(class内での変数受渡しのため)
        self.params = (C, Iw, M, Fw, stride, Ow)

    def __call__(self, col):
        C, Iw, M, Fw, stride, Ow = self.params
        B = col.size // (M*Fw*Iw)
        col = col.reshape(B,Iw,M,Fw).transpose(0,2,3,1) # col.shape=(B,M,Fw,Iw)
        vec = np.zeros((B, M, Ow), dtype=Config.dtype)
        # colからstride,Ow個のデータを取ってきて、vecにstride毎に並べる
        # それをFh,Fwを満たすまで繰返す
        for fw in range(Fw):
            w_lim = fw + stride*Iw
            vec[:,:,fw:w_lim:stride] += col[:,:,fw,:]
        return vec


### 畳み込み層 #####################################################
class Conv2dLayer(BaseLayer):
    """ 二次元畳込み層 """
    # B:バッチサイズ,
    # C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # M:フィルタ数, Fh:フィルタ高さ, Fw:フィルタ幅
    # Sh/_w:ストライド高さ/幅, pad:パディング幅
    # 出力チャンネル数=フィルタ数M, Oh:出力高さ, Ow:出力幅
    # w_decay:L2正則化項の係数
    
    def __init__(self, *configuration, **kwargs):
        if len(configuration) == 6:
            C, image_size, M, kernel_size, stride, pad = configuration
        if len(configuration) == 4:
            C = None; image_size = None; M, kernel_size, stride, pad = configuration
        if len(configuration) == 2:
            C = None; image_size = None; M, kernel_size = configuration; stride = 1; pad = 0 
        Oh = None; Ow = None

        Ih, Iw = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        Fh, Fw = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        Sh, Sw = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        
        self.params = self.config = C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow
        super().__init__(**kwargs)
        
    def fix_configuration(self, shape):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        if len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
            
        Oh = (Ih - Fh + 2*pad) // Sh + 1   # 出力高さ
        Ow = (Iw - Fw + 2*pad) // Sw + 1   # 出力幅
        self.config = C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow

    def get_parameter_size(self):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        m = C*Fh*Fw  # 入力チャネル数とフィルタサイズ
        n = M        # フィルタ数
        return m, n

    def __forward__(self, x, *, train=False, dropout=0.0):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        self.x_shape = x.shape
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        x = x.reshape(-1, C, Ih, Iw)    # (B,C,Ih,Iw)  
        # '0'パディング
        x = np.pad(x, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        # 入力画像を行列に変換 (B,C,Ih+2*pad,Iw+2*pad)->(C*Fh*Fw,B*Oh*Ow) 
        cols = Im2col(C, Ih+2*pad, Iw+2*pad, M, Fh, Fw, Sh, Sw)(x)
        # Affine変換: (B*Oh*Ow,C*Fh*Fw)×(C*Fh*Fw,M)->(B*Oh*Ow,M)
        w, b, gamma = self.parameters()    
        y = self.dot_linear.forward(cols, w, b, gamma)
        y = y.reshape(-1, Oh, Ow, M).transpose(0, 3, 1, 2) # u.shape=(B,M,Oh,Ow) 
        y = super().__forward__(y, train=train, dropout=dropout)
        return y
    
    def __backward__(self, grad_y, flush=True):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        #grad_y = grad_y.reshape(-1, M, Oh, Ow)       # grad_y.shape=(B,M,Oh,Ow)
        grad_y = super().__backward__(grad_y)
        grad_y = grad_y.transpose(0, 2, 3, 1).reshape(-1, M) # grad_y.shape=(B*Oh*Ow,M)
        # Affineの逆伝播 grad_cols.shape=(B*Oh*Ow,C*Fh*Fw)
        grad_cols, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)
        self.parameters.set_gradient(grad_w, grad_b, ggamma, flush=flush)
        # 行列を画像に変換 (B*Oh*Ow,C*Fh*Fw)->(B,C,Ih,Iw)  　
        #grad_x = Col2im(M, Oh, Ow, C, Fh, Fw, Sh, Sw)(grad_cols.T)
        grad_x = Col2im(M, Oh, Ow, C, Fh, Fw, Sh, Sw)(grad_cols) # 20251107AI
        # パディング分を外して元の画像データに戻す        
        grad_x = grad_x[:,:,pad:pad+Ih,pad:pad+Iw]
        grad_x = grad_x.reshape(self.x_shape)
        return grad_x

class ConvLayer(Conv2dLayer):
    pass

### 逆畳み込み層 #####################################################
class DeConv2dLayer(BaseLayer):
    """ 二次元逆畳込み """
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # M:フィルタ数, Fh:フィルタ高さ, Fw:フィルタ幅
    # Sh/_w:ストライド高さ/幅, pad:パディング幅
    # 出力チャンネル数=フィルタ数M, Oh:出力高さ, Ow:出力幅
    # w_decay:L2正則化項の係数
    
    def __init__(self, *configuration, **kwargs):
        if len(configuration) == 6:
            C, image_size, M, kernel_size, stride, pad = configuration
        if len(configuration) == 4:
            C = None; image_size = None; M, kernel_size, stride, pad = configuration
        if len(configuration) == 2:
            C = None; image_size = None; M, kernel_size = configuration; stride = 1; pad = 0 
        Oh = None; Ow = None

        Ih, Iw = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        Fh, Fw = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        Sh, Sw = stride if isinstance(stride, (tuple, list)) else (stride, stride)

        self.params = self.config = C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow
        super().__init__(**kwargs)

    def fix_configuration(self, shape):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        if len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C  = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
            
        Oh = (Ih - 1) * Sh + Fh - 2 * pad  # 出力高さ
        Ow = (Iw - 1) * Sw + Fw - 2 * pad  # 出力幅
        self.config = C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow

    def get_parameter_size(self):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        m = C              # 入力チャネル数「要注意」
        n = M*Fh*Fw        # フィルタ数とフィルタサイズ「要注意」
        return m, n

    def __forward__(self, x, *, train=False, dropout=0.0):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        self.x_shape = x.shape
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        x = x.reshape(-1, C, Ih, Iw).transpose(0,2,3,1).reshape(-1,C) # (B*Ih*Iw,C)  

        #print('## x.shape', x.shape, '->', self.x.shape)
        # Affine変換 (B*Ih*Iw,C)×(C,M*Fh*Fw)->(B*Ih*Iw,M*Fh*Fw)
        w, b, gamma = self.parameters()    
        cols = self.dot_linear.forward(x, w, b, gamma)

        #print('## cols.shape', cols.shape)
        # 行列を画像に変換 cols.T:(M*Fh*Fw,B*Ih*Iw)->(B,M,Oh,Ow)  　
        #y = Col2im(C, Ih, Iw, M, Fh, Fw, Sh, Sw)(cols.T)
        y = Col2im(C, Ih, Iw, M, Fh, Fw, Sh, Sw)(cols) # 20251107AI
        # 画像調整 トリミング
        y = y[:,:,pad:pad+Oh,pad:pad+Ow]              # y.shape=(B,M,Oh,Ow)
        y = super().__forward__(y, train=train, dropout=dropout)
        return y

    def __backward__(self, grad_y, flush=True):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        #grad_y = grad_y.reshape(-1, M, Oh, Ow)       # grad_y.shape=(B,M,Oh,Ow)
        grad_y = super().__backward__(grad_y)
        #  '0'パディング
        grad_y = np.pad(grad_y, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        # 画像の勾配を行列に変換 grad_y.shape=(M*Fh*Fw,B*Ih*Iw)に変換
        grad_y = Im2col(M, Oh+2*pad, Ow+2*pad, C, Fh, Fw, Sh, Sw)(grad_y)
        # Affineの逆伝播
        grad_x, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)
        self.parameters.set_gradient(grad_w, grad_b, ggamma, flush=flush)
        grad_x = grad_x.reshape(-1,Ih,Iw,C).transpose(0,3,1,2) # (B,C,Ih,Iw)
        grad_x = grad_x.reshape(self.x_shape)
        return grad_x

class DeConvLayer(DeConv2dLayer):
    pass

class Im2col:
    """
    img.shape=(B,C,Ih,Iw) → cols.shape=(B,C,Fh,Fw,Oh,Ow) -> (B*Oh*Ow, C*Fh*Fw)

    """
    def __init__(self, C, Ih, Iw, M, Fh, Fw, Sh, Sw):
        # 出力画像のサイズ
        Oh = (Ih - Fh) // Sh + 1        # 出力高さ
        Ow = (Iw - Fw) // Sw + 1        # 出力幅
        # パラメータをまとめる(class内での変数受渡しのため)
        self.params = (C, Ih, Iw, M, Fh, Fw, Sh, Sw, Oh, Ow)

    def __call__(self, img):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, Oh, Ow = self.params
        B = img.size // (C*Ih*Iw)
        col = np.empty((B, C, Fh, Fw, Oh, Ow), dtype=Config.dtype) # メモリ節約のためzerosでなくempty 
        # imgからstride毎のデータを取ってきて、colsにOh,Owになるまで並べる
        # それをFh,Fwを満たすまで繰返す
        for fh in range(Fh):
            h_lim = fh + Sh*Oh
            for fw in range(Fw):
                w_lim = fw + Sw*Ow
                col[:,:,fh,fw,:,:] = img[:,:,fh:h_lim:Sh,fw:w_lim:Sw]
        # 軸の入替と変形     B  Oh Ow C Fh Fw 
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(B*Oh*Ow, C*Fh*Fw)
        return col

class Col2im:
    """
    col.shape=(B*Ih*Iw, M*Fh*Fw)->(B,Ih,Iw,M,Fh,Fw)->(B,M,Fh,Fw,Ih,Iw)→img.shape=(B,M,Oh,Ow)

    """
    def __init__(self, C, Ih, Iw, M, Fh, Fw, Sh, Sw):
        # Im2col 側の Oh,Ow (= ここでの Ih,Iw) から復元後の画像サイズを計算
        Oh = (Ih - 1) * Sh + Fh
        Ow = (Iw - 1) * Sw + Fw
        self.params = (C, Ih, Iw, M, Fh, Fw, Sh, Sw, Oh, Ow)

    def __call__(self, col):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, Oh, Ow = self.params
        B = col.size // (M*Ih*Iw*Fh*Fw)
        col = col.reshape(B, Ih, Iw, M, Fh, Fw).transpose(0, 3, 4, 5, 1, 2)
        img = np.zeros((B, M, Oh, Ow), dtype=Config.dtype)
        # colからstride*Ih,Ow個のデータを取ってきて、imgにstride毎に並べる
        # それをFh,Fwを満たすまで繰返す
        for fh in range(Fh):
            h_lim = fh + Sh * Ih
            for fw in range(Fw):
                w_lim = fw + Sw * Iw
                img[:, :, fh:h_lim:Sh, fw:w_lim:Sw] += col[:, :, fh, fw, :, :]
        return img

class Col2im_bkup:
    # col.shape = (M*Fh*Fw, B*Ih*Iw) -> img.shape = (B, M, Oh, Ow)
    def __init__(self, C, Ih, Iw, M, Fh, Fw, Sh, Sw):
        # 出力画像のサイズ
        Oh = (Ih - 1) * Sh + Fh
        Ow = (Iw - 1) * Sw + Fw 
        # パラメータをまとめる(class内での変数受渡しのため)
        self.params = (C, Ih, Iw, M, Fh, Fw, Sh, Sw, Oh, Ow)

    def __call__(self, col):
        """ col.shape=(B,M,Fh,Fw,Ih,Iw) → img.shape=(B,M,Oh,Ow)  """
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, Oh, Ow = self.params
        B = col.size // (M*Fh*Fw*Ih*Iw)
        col = col.reshape(M,Fh,Fw,B,Ih,Iw).transpose(3,0,1,2,4,5) # col.shape=(B,M,Fh,Fw,Ih,Iw)
        img = np.zeros((B, M, Oh, Ow), dtype=Config.dtype)
        # colからstride*Ih,Ow個のデータを取ってきて、imgにstride毎に並べる
        # それをFh,Fwを満たすまで繰返す
        for fh in range(Fh):
            h_lim = fh + Sh*Ih
            for fw in range(Fw):
                w_lim = fw + Sw*Iw
                img[:,:,fh:h_lim:Sh,fw:w_lim:Sw] += col[:,:,fh,fw,:,:]
        return img

### プーリング層 ####################################################
class Pooling1dLayer(Function):  
    # B:バッチサイズ, C:入力チャンネル数, Iw:入力幅
    # pool:プーリング領域のサイズ, pad:パディング幅
    # C:出力チャンネル数, Ow:出力幅
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        method = None
        if   len(configuration) == 5:
            C, Iw, pool, pad, method = configuration
        elif len(configuration) == 4:
            C, Iw, pool, pad = configuration
        elif len(configuration) == 3:
            C = None; Iw = None; pool, pad, method = configuration
        elif len(configuration) == 2:
            C = None; Iw = None; pool, pad = configuration
        elif len(configuration) == 1:
            C = None; Iw = None; pool = configuration; pad = 0
        else:
            C = None; Iw = None; pool = 2; pad = 0 
        Ow = None
        self.params = self.config = C, Iw, pool, pad, Ow
        self.method = kwargs.pop('method', 'max') if method is None else method
        print('Initialize', self.__class__.__name__, self.config, self.method)
        self.max_index = None
        self.DO = Dropout() if kwargs.pop('dropout', False) else None
        
    def fix_configuration(self, shape):
        C, Iw, pool, pad, Ow = self.config
        B  = shape[0]
        if len(shape) >= 2:
            Iw = shape[-1] 
            C = shape[1] if len(shape)==3 else 1
        elif C is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
        Ow = (Iw + 2 * pad + pool - 1) // pool # 端数は切捨て
        self.config = C, Iw, pool, pad, Ow
            
    def __forward__(self, x, *, train=False, dropout=0.0):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        C, Iw, pool, pad, Ow = self.config
        #B = x.size // (C*Ih*Iw)
        x = x.reshape(-1, C, Iw)                     # 入力の形状 ex. (C, Iw)に対応   
        pdw = Ow * pool - Iw - pad                   # サイズの端数に対応
        # 画像調整            B      C      Iw左 Iw右　ゼロパディング   
        img_pad = np.pad(x, [(0,0), (0,0), (pad, pdw)], 'constant')
        y, self.max_index = Pooling1d(pool, Ow, self.method)(img_pad)
        if self.DO:
            y = self.DO.forward(y, dropout=dropout)  # 形状は(B,C,Oh,Ow)
        return y

    def __backward__(self, grad_y):
        C, Iw, pool, pad, Ow = self.config  
        B = grad_y.size // (C*Ow)                    # B = grad_y.shape[0] = len(grad_y)
        self.grad_y = grad_y.reshape(B, C, Ow)       # ドロップアウトへの入力形状は順伝播時と同じ
        if self.DO:
            self.grad_y = self.DO.backward(self.grad_y)  # ドロップアウト
        grad_x = UnPooling1d(pool, self.method)(self.grad_y, self.max_index)
        # 画像調整 トリミング
        grad_x = grad_x[:, :, pad:pad+Iw]            # grad_x.shape=(B,C,Iw) 
        return grad_x

### 逆プーリング層 ####################################################
class UnPooling1dLayer(Function):  
    # B:バッチサイズ, C:入力チャンネル数, Iw:入力幅
    # pool:プーリング領域のサイズ, pad:パディング幅
    # C:出力チャンネル数, Ow:出力幅
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        method = None
        if   len(configuration) == 5:
            C, Iw, pool, pad, method = configuration
        elif len(configuration) == 4:
            C, Iw, pool, pad = configuration
        elif len(configuration) == 3:
            C = None; Iw = None; pool, pad, method = configuration
        elif len(configuration) == 2:
            C = None; Iw = None; pool, pad = configuration
        elif len(configuration) == 1:
            C = None; Iw = None; pool = configuration; pad = 0
        else:
            C = None; Iw = None; pool = 2; pad = 0
        Ow = None
        self.params = self.config = C, Iw, pool, pad, Ow
        self.method = kwargs.pop('method', 'max') if method is None else method
        print('Initialize', self.__class__.__name__, self.config, self.method)
        self.max_index = None
        self.DO = Dropout() if kwargs.pop('dropout', False) else None
        
    def fix_configuration(self, shape):
        C, Iw, pool, pad, Ow = self.config
        B  = shape[0]
        if len(shape) >= 2:
            Iw = shape[-1] 
            C = shape[1] if len(shape)==3 else 1
        elif C is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
        Ow = Iw * pool - 2 * pad
        self.config = C, Iw, pool, pad, Ow
            
    def __forward__(self, x, *, train=False, dropout=0.0, max_index=None):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        C, Iw, pool, pad, Ow = self.config
        #B = x.size // (C*Iw)
        x = x.reshape(-1, C, Iw)                     # 入力の形状 ex. (C,Ih*Iw)に対応   
        #print('img_pad', img_pad.shape, self.config)
        y = UnPooling1d(pool, self.method)(x, max_index)
        # 画像調整 トリミング
        y = y[:, :, pad:pad+Ow]                      # y.shape=(B,C,Oh,Ow) 
        if self.DO:
            y = self.DO.forward(y, dropout=dropout)  # 形状は(B,C,Oh,Ow)
        return y

    def __backward__(self, grad_y):
        C, Iw, pool, pad, Ow = self.config   # パラメタ
        B = grad_y.size // (C*Ow)                    # B = grad_y.shape[0] = len(grad_y)
        grad_y = grad_y.reshape(B, C, Ow)            # ドロップアウトへの入力形状は順伝播時と同じ
        if self.DO:
            self.grad_y = self.DO.backward(self.grad_y)  # ドロップアウト
        pdw = Iw*pool - Ow - pad                     # 画像サイズの端数を調整
        # 画像調整                 B      C     Iw左 Iw右　 ゼロパディング   
        grad_y = np.pad(grad_y, [(0,0), (0,0), (pad, pdw)], 'constant')
        grad_x, _ = Pooling1d(pool, Iw, self.method)(grad_y)
        return grad_x

class Pooling1d:  
    def __init__(self, pool, Ow, method):
        self.config = pool, Ow, method
            
    def __call__(self, x):
        pool, Ow, method = self.config
        B, C, Iw = x.shape
                        # (0,  1, 2,  3   )
        quarry = x.reshape(-1, C, Ow, pool)          # poolはaxis=3   
                        
        if method == 'average': # averageプーリング　
            y = np.mean(quarry, axis=3)              # poolの軸で平均値
            max_index = None
        else:     # 'max': 通常は Maxプーリング
            y = np.max (quarry, axis=3)              # poolの軸で最大値
            max_index = np.argmax(quarry, axis=3)    # インデクス記録
        return y, max_index

class UnPooling1d:  
    def __init__(self, pool, method=None):
        self.config = pool, method

    def __call__(self, x, max_index=None):
        pool, method = self.config   # パラメタ
        B, C, Iw = x.shape
        quarry = np.zeros((B*C*Iw, pool), dtype='f4') # 初期値0
        # 各行に勾配を入れる
        if method == 'average' or max_index is None: 
            quarry[np.arange(B*C*Iw)] \
                =  np.repeat((x/pool).reshape(-1, 1), pool, axis=1)
        else:     #  'max':    各行の最大値であった列の要素にのみ出力の勾配を入れる
            max_index = max_index.reshape(-1)
            quarry[np.arange(B*C*Iw), max_index] = x.reshape(-1)
                      #     (B*C*Iw,  pool)
        y = quarry.reshape  (B, C, Iw*pool)
        return y


### プーリング層 ####################################################
class Pooling2dLayer(Function):
    """ 二次元プーリング層 """
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # pool:プーリング領域のサイズ, pad:パディング幅
    # C:出力チャンネル数, Oh:出力高さ, Ow:出力幅
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        method = None
        if   len(configuration) == 5:
            C, image_size, pool, pad, method = configuration
        elif len(configuration) == 4:
            C, image_size, pool, pad = configuration
        elif len(configuration) == 3 and isinstance(configuration[-1], str):
            C = None; image_size = None; pool, pad, method = configuration
        elif len(configuration) == 3:
            C, image_size, pool = configuration; pad = 0
        elif len(configuration) == 2 and isinstance(configuration[-1], str):
            C = None; image_size = None; pool, method = configuration; pad = 0
        elif len(configuration) == 2:    
            C = None; image_size = None; pool, pad = configuration
        elif len(configuration) == 1:
            C = None; image_size = None; pool = configuration; pad = 0
        else:
            C = None; image_size = None; pool = 2; pad = 0

        Ih, Iw = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        pool_h, pool_w = pool if isinstance(pool, (tuple, list)) else (pool, pool)
        Oh = Ow = None

        self.params = self.config = C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow
        self.method = kwargs.pop('method', 'max') if method is None else method
        print('Initialize', self.__class__.__name__, self.config, self.method)
        self.max_index = None
        self.DO = Dropout() if kwargs.pop('dropout', False) else None
        
    def fix_configuration(self, shape):
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config
        B  = shape[0]
        if len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None or Iw is None:
            raise Exception(self.__class__.__name__, 'cannot fix configuration.')

        Oh = (Ih + 2 * pad + pool_h - 1) // pool_h # 端数は切捨て
        Ow = (Iw + 2 * pad + pool_w - 1) // pool_w # 端数は切捨て
        self.config = C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow
            
    def __forward__(self, x, *, train=False, dropout=0.0):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config
        #B = x.size // (C*Ih*Iw)
        x = x.reshape(-1, C, Ih, Iw)                 # 入力の形状 ex. (C,Ih*Iw)に対応   
        pdh = Oh * pool_h - Ih - pad                 # 画像サイズの端数に対応
        pdw = Ow * pool_w - Iw - pad                 # 画像サイズの端数に対応
        # 画像調整            B      C     Ih上　Ih下   Iw左 Iw右　ゼロパディング   
        img_pad = np.pad(x, [(0,0), (0,0), (pad, pdh), (pad, pdw)], 'constant')
        y, self.max_index = Pooling2d(pool_h, pool_w, Oh, Ow, self.method)(img_pad)
        if self.DO:
            y = self.DO.forward(y, dropout=dropout)  # 形状は(B,C,Oh,Ow)
        return y

    def __backward__(self, grad_y):
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config   # パラメタ
        B = grad_y.size // (C*Oh*Ow)                 # B = grad_y.shape[0] = len(grad_y)
        self.grad_y = grad_y.reshape(B, C, Oh, Ow)   # ドロップアウトへの入力形状は順伝播時と同じ
        if self.DO:
            self.grad_y = self.DO.backward(self.grad_y)  # ドロップアウト
        grad_x = UnPooling2d(pool_h, pool_w, self.method)(self.grad_y, self.max_index)
        # 画像調整 トリミング
        grad_x = grad_x[:, :, pad:pad+Ih, pad:pad+Iw] # grad_x.shape=(B,C,Ih,Iw) 
        return grad_x

class PoolingLayer(Pooling2dLayer):
    pass

### 逆プーリング層 ####################################################
class UnPooling2dLayer(Function):  
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # pool:プーリング領域のサイズ, pad:パディング幅
    # C:出力チャンネル数, Oh:出力高さ, Ow:出力幅
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        method = None
        if   len(configuration) == 5:
            C, image_size, pool, pad, method = configuration
        elif len(configuration) == 4:
            C, image_size, pool, pad = configuration
        elif len(configuration) == 3 and isinstance(configuration[-1], str):
            C = None; image_size = None; pool, pad, method = configuration
        elif len(configuration) == 3:
            C, image_size, pool = configuration; pad = 0
        elif len(configuration) == 2 and isinstance(configuration[-1], str):
            C = None; image_size = None; pool, method = configuration; pad = 0
        elif len(configuration) == 2:    
            C = None; image_size = None; pool, pad = configuration
        elif len(configuration) == 1:
            C = None; image_size = None; pool = configuration; pad = 0
        else:
            C = None; image_size = None; pool = 2; pad = 0

        Ih, Iw = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        pool_h, pool_w = pool if isinstance(pool, (tuple, list)) else (pool, pool)
        Oh = Ow = None

        self.params = self.config = C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow
        self.method = kwargs.pop('method', 'max') if method is None else method
        print('Initialize', self.__class__.__name__, self.config, self.method)
        self.max_index = None
        self.DO = Dropout() if kwargs.pop('dropout', False) else None
        
    def fix_configuration(self, shape):
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config
        B  = shape[0]
        if len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None or Iw is None:
            raise Exception(self.__class__.__name__, 'cannot fix configuration.')

        Oh = Ih * pool_h - 2 * pad
        Ow = Iw * pool_w - 2 * pad
        self.config = C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow
            
    def __forward__(self, x, *, train=False, dropout=0.0, max_index=None):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config
        #B = x.size // (C*Ih*Iw)
        x = x.reshape(-1, C, Ih, Iw)                 # 入力の形状 ex. (C,Ih*Iw)に対応   
        #print('img_pad', img_pad.shape, self.config)
        y = UnPooling2d(pool_h, pool_w, self.method)(x, max_index)
        # 画像調整 トリミング
        y = y[:, :, pad:pad+Oh, pad:pad+Ow]          # y.shape=(B,C,Oh,Ow) 
        if self.DO:
            y = self.DO.forward(y, dropout=dropout)  # 形状は(B,C,Oh,Ow)
        return y

    def __backward__(self, grad_y):
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config   # パラメタ
        B = grad_y.size // (C*Oh*Ow)                 # B = grad_y.shape[0] = len(grad_y)
        grad_y = grad_y.reshape(B, C, Oh, Ow)        # ドロップアウトへの入力形状は順伝播時と同じ
        if self.DO:
            self.grad_y = self.DO.backward(self.grad_y)  # ドロップアウト
        pdh = Ih*pool_h - Oh - pad                   # 画像サイズの端数を調整
        pdw = Iw*pool_w - Ow - pad                   # 画像サイズの端数を調整
        # 画像調整            B      C     Ih上　Ih下   Iw左 Iw右　ゼロパディング   
        grad_y = np.pad(grad_y, [(0,0), (0,0), (pad, pdh), (pad, pdw)], 'constant')
        grad_x, _ = Pooling2d(pool_h, pool_w, Ih, Iw, self.method)(grad_y)
        return grad_x

class UnPoolingLayer(UnPooling2dLayer):
    pass

class Pooling2d:  
    def __init__(self, pool_h, pool_w, Oh, Ow, method):
        self.config = pool_h, pool_w, Oh, Ow, method
            
    def __call__(self, x):
        pool_h, pool_w, Oh, Ow, method = self.config
        B, C, Ih, Iw = x.shape
                        #   (0,  1, 2,  3,    4,   5   )
        quarry = x.reshape  (-1, C, Oh, pool_h, Ow,  pool_w) \
                  .transpose(0,  1, 2,  4,      3,   5)      \
                  .reshape  (-1, C, Oh, Ow,   pool_h*pool_w)
                        #   (0, 1, 2, 3,    4)  pool_h*pool_wはaxis=4 
        if method == 'average': # averageプーリング　
            y = np.mean(quarry, axis=4)   # pool*poolの軸で平均値
            max_index = None
        else:     # 'max': 通常は Maxプーリング
            y = np.max (quarry, axis=4)   # pool*poolの軸で最大値
            max_index = np.argmax(quarry, axis=4)   # インデクス記録
        return y, max_index

class UnPooling2d:  
    def __init__(self, pool_h, pool_w, method=None):
        self.config = pool_h, pool_w, method

    def __call__(self, x, max_index=None):
        pool_h, pool_w, method = self.config   # パラメタ
        B, C, Ih, Iw = x.shape
        quarry = np.zeros((B*C*Ih*Iw,pool_h*pool_w), dtype='f4')  # 初期値0
        # 各行に勾配を入れる
        if method == 'average' or max_index is None: 
            quarry[np.arange(B*C*Ih*Iw)] \
                =  np.repeat((x/(pool_h*pool_w)).reshape(-1, 1), pool_h*pool_w, axis=1)
        else:     #  'max':    各行の最大値であった列の要素にのみ出力の勾配を入れる
            max_index = max_index.reshape(-1)
            quarry[np.arange(B*C*Ih*Iw), max_index] = x.reshape(-1)
                      #     (B*C*Ih*Iw,    pool*pool)
        y = quarry.reshape  (B*C, Ih, Iw,  pool_h, pool_w)   \
                  .transpose(0,   1,  3,   2,      4)      \
                  .reshape  (B,C, Ih*pool_h, Iw*pool_w) 
                      #     (B,C, Oh,        Ow) 
        return y

### globalAveragePooling層 ####################################################
class GlobalAveragePooling(Function):
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        if len(configuration) == 3:
            C, Ih, Iw = configuration
        else:
            C = None; Ih = None; Iw = None
        self.config = C, Ih, Iw
        print('Initialize', self.__class__.__name__, self.config)
        self.DO = Dropout() if kwargs.pop('dropout', False) else None

    def fix_configuration(self, shape):
        C, Ih, Iw = self.config
        B = shape[0]
        if len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None or Iw is None:
            raise Exception(self.__class__.__name__, 'cannot fix configuration.')
        self.config = C, Ih, Iw

    def __forward__(self, x, *, train=False, dropout=0.0):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        y = np.mean(x, axis=(2, 3))
        if self.DO:
            y = self.DO.forward(y, dropout=dropout)       
        return y
        
    def __backward__(self, grad_y):
        C, Ih, Iw = self.config
        B = grad_y.size//C
        if self.DO:
            grad_y = self.DO.backward(grad_y) 
        grad_x = grad_y / (Ih*Iw)
        grad_x = np.broadcast_to(grad_x.reshape(B, C, 1, 1), (B, C, Ih, Iw))
        return grad_x

    def __backward2__(self, grad_y):
        C, Ih, Iw = self.config
        #grad_x = self.DO.backward(grad_y) # 
        grad_x /= Ih*Iw
        grad_x = np.repeat(grad_x.reshape(-1, 1), Ih*Iw, axis=1)
        grad_x = grad_x.reshape(-1, C, Ih, Iw)
        return grad_x

### マスク展開層 = 逆畳込みもどき ###########################################
class MaskedExpansionLayer(BaseLayer):
    # マスクに入力をかけて展開する
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # M:フィルタ数は入力チャネル数と同じ, Fh:フィルタ高さ, Fw:フィルタ幅
    # 出力チャンネル数=入力チャネル数, Oh:出力高さ, Ow:出力幅
    # w_decay:L2正則化項の係数
    
    def __init__(self, *configuration, **kwargs):
        if len(configuration) == 8:
            C, Ih, Iw, M, Fh, Fw, stride, pad = configuration
        if len(configuration) == 6:
            C, Ih, Iw, M, Fh, Fw = configuration
        if len(configuration) == 5:
            C, Ih, Iw, Fh, Fw = configuration; M = 1
        if len(configuration) == 3:
            C = None; Ih = None; Iw = None; M, Fh, Fw = configuration
        if len(configuration) == 2:
            C = None; Ih = None; Iw = None; M = 1; Fh, Fw = configuration
        # stride, pad は未対応
        Oh = Ow = None    
        self.config = (C, Ih, Iw, M, Fh, Fw, Oh, Ow)
        super().__init__(**kwargs)

    def fix_configuration(self, shape):
        C, Ih, Iw, M, Fh, Fw, Oh, Ow = self.config
        if len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None:
            raise Exception('MaskedExpansionLayer cannot fix configuration.')
            
        self.Oh = Oh = Ih * Fh
        self.Ow = Ow = Iw * Fw
        self.config = C, Ih, Iw, M, Fh, Fw, Oh, Ow
            
    def get_parameter_size(self):
        C, Ih, Iw, M, Fh, Fw, Oh, Ow = self.config
        m = M              # 入力チャネル数とフィルタサイズ
        n = C*Fh*Fw        # フィルタ数
        return m, n

    def __forward__(self, x, *, train=False, dropout=0.0):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        C, Ih, Iw, M, Fh, Fw, Oh, Ow = self.config
        B = x.size // (C*Ih*Iw)                        # B = x.shape[0] = len(x)
        w, b, gamma = self.parameters()    
        w = w.reshape(M, C, Fh*Fw).transpose(1,0,2) # (C,M,Fh*Fw) 
        b = b.reshape(C, 1, Fh*Fw)
        x = x.reshape(B,C,Ih,Iw).transpose(1,0,2,3).reshape(C,B*Ih*Iw,1)
        self.x = x
        z = np.tile(x, (1, 1, M)) # 同じ要素をフィルタ数分繰返す(C,B*Ih*Iw,M)

        # 出力の計算: 各チャネル毎　
        y = np.empty((C, B*Ih*Iw, Fh*Fw), dtype='f4')
        for c in range(C):
            y[c] = np.dot(z[c], w[c]) + b[c]  # u[c].shape=(B*Ih*Iw,Fh*Fw)
        y = y.reshape(C,B,Ih,Iw,Fh,Fw).transpose(1,0,2,4,3,5).reshape(B,C,Oh,Ow)
        y = super().__forward__(y, train=train, dropout=dropout)
        return y
    
    def __backward__(self, grad_y):
        #print('畳み込み層逆伝播grad_yの形は', grad_y.shape)
        x, = self.inputs
        C, Ih, Iw, M, Fh, Fw, Oh, Ow = self.config
        B = grad_y.size // (C*Oh*Ow)                  # B = grad_y.shape[0] = len(grad_y)
        grad_y = grad_y.reshape(B, C, Oh, Ow)
        grad_y = super().__backward__(grad_y)
        grad_y = grad_y.reshape(B,C,Ih,Fh,Iw,Fw).transpose(1,0,2,4,3,5).reshape(C,B*Ih*Iw,Fh*Fw)
        
        # フィルタとバイアスの勾配
        w = self.w.reshape(M,C,Fh*Fw).transpose(1,0,2) # (C,M,Fh*Fw) 
        b = self.b.reshape(C, 1, Fh*Fw)
        grad_w = np.empty((C,M,Fh*Fw), dtype='f4')
        grad_b = np.empty((C,1,Fh*Fw), dtype='f4')
        grad_z = np.empty((C,B*Ih*Iw,M), dtype='f4')
        z = np.tile(x, (1, 1, M)) # 同じ要素をフィルタ数分繰返す(C,B*Ih*Iw,M)
        for c in range(C):
            grad_w[c] = np.dot(z[c].T, grad_y[c])       # (M,Fh*Fw)
            grad_b[c] = np.sum(grad_y[c], axis=0)
            grad_z[c] = np.dot(grad_y[c], w[c].T)       # (B*Ih*Iw,M)

        grad_w = grad_w.transpose(1,0,2).reshape(M,C*Fh*Fw)
        grad_b = grad_b.reshape(-1)
        self.parameters.set_gradient(grad_w, grad_b, None)
        grad_z = grad_z.reshape(C,B,Ih,Iw,M).transpose(1,0,2,3,4) # (B,C,Ih,Iw,M)
        grad_x = np.sum(grad_z, axis=4)
        return grad_x

# -- 潜在変数をサンプリングする層 -- 20240701
class LatentSampling(Function):
    def __init__(self, rate=1.0, kld=None, mil=None, **kwargs):# kwargsは他の層に対する指定を無視するために必要
        super().__init__()
        print('Initialize', self.__class__.__name__)
        self.sampling = MuVarSampling()             # サンプリングの関数
        self.rate = rate                            # サンプリングの広がり
        if kld and kld>0:
            self.r_kld = kld                        # 混ぜ具合 
            self.kld = KullbackLeiblerDivergence()  # カルバック・ライブラー情報量関数
        else:
            self.kld = None
        if mil and mil>0:
            self.r_mil = mil                        # 混ぜ具合
            self.mil = MutualInformationLoss()      # 相互情報量
        else:
            self.mil = None
        
    def __forward__(self, x, *, epsilon=None):
        # -- xを半分ずつmuとlog_varに振り分ける --
        x = x.reshape(len(x), -1) # バッチサイズ×ベクトル
        nz = x.shape[-1]//2       # 半分
        mu = x[:, :nz]
        log_var = x[:, nz:2*nz]
        # -- epsilonを決める --
        if epsilon is None:
            epsilon = (self.rate * np.random.randn(*log_var.shape)).astype(Config.dtype)
        # -- サンプリングとカルバック・ライプラー --    
        z = self.sampling.forward(mu, log_var, epsilon=epsilon)

        if not (self.kld or self.mil):
            return z
        if self.kld:
            kll = self.r_kld * self.kld.forward(mu, log_var)
        else:
            kll = 0
        if self.mil:
            mi  = self.r_mil * self.mil.forward(z, mu, log_var)
        else:
            mi = 0
        return z, kll, mi
    
    def __backward__(self, gz, gkll=1, gmi=1):
        gmu, glog_var = 0, 0

        if self.mil:
            gz0, gmu0, glog_var0 = self.mil.backward(gmi * self.r_mil)
            gz += gz0
            gmu += gmu0
            glog_var += glog_var0

        if self.kld:
            gmu0, glog_var0 = self.kld.backward(gkll * self.r_kld) 
            gmu += gmu0
            glog_var += glog_var0

        gmu0, glog_var0 = self.sampling.backward(gz)
        gmu += gmu0
        glog_var += glog_var0

        gx = np.hstack((gmu, glog_var))           # forwardで振り分けたことに対応

        return gx
    
# -- 潜在変数をサンプリングする層 -- 2023.07.17
class LatentSamplingZ:
    def __init__(self, rate=1.0, **kwargs): # kwargsは使わないが他の層に対する指定を無視するために必要
        super().__init__()
        print('Initialize', self.__class__.__name__)
        self.sampling = MuVarSampling()         # サンプリングの関数　  
        self.kld = KullbackLeiblerDivergence()  # カルバック・ライブラー情報量関数
        self.rate = rate                        # サンプリングの広がり
        #self.r_kld = kwargs.pop('r_kld', 1.0)  # kldの混ぜ具合 
        
    def forward(self, mu, log_var, epsilon=None):
        if epsilon is None:
            epsilon = np.random.randn(*log_var.shape) * self.rate
            epsilon = epsilon.astype(Config.dtype)
        z = self.sampling.forward(mu, log_var, epsilon=epsilon)
        kll = self.kld.forward(mu, log_var)
        return z, kll
    
    def backward(self, gz, gkll):
        gmu1, glog_var1 = self.sampling.backward(gz)
        gmu2, glog_var2 = self.kld.backward(gkll)    # gkllの効果は不明
        gmu = gmu1 + gmu2
        glog_var = glog_var1 + glog_var2
        return gmu, glog_var

class MuVarSampling(Function):
    def __forward__(self, mu, log_var, *, epsilon=None):
        self.epsilon = epsilon # 逆伝播に備えて覚える
        z = mu + self.epsilon * np.exp(log_var/2)
        return z

    def __backward__(self, gz=1):
        mu, log_var = self.inputs
        epsilon = self.epsilon # 順伝播から引き継ぐ 
        gmu = np.broadcast_to(gz, mu.shape) 
        glog_var = 0.5 * gz * epsilon * np.exp(log_var/2)
        return gmu, glog_var

class MuVarSampling2(Function):
    """ mu, log_var, epsilonのすべてが位置変数のサンプリング """
    def __forward__(self, mu, log_var, epsilon):
        z = mu + epsilon * np.exp(log_var/2)
        return z

    def __backward__(self, gz=1):
        mu, log_var, epsilon = self.inputs
        gmu = np.broadcast_to(gz, mu.shape) 
        gepsilon = gz * np.exp(log_var/2)
        #glog_var = 0.5 * gz * epsilon * np.exp(log_var/2)
        glog_var = 0.5 * gepsilon * epsilon
        return gmu, glog_var, gepsilon

class KullbackLeiblerDivergence(Function):
    def __forward__(self, mu, log_var):
        #self.inputs = mu, log_var
        kll = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
        return kll / len(mu)

    def __backward__(self, gkll=1):
        mu, log_var = self.inputs
        gmu = gkll * mu
        glog_var = gkll * (-0.5) * (1 - np.exp(log_var))
        return gmu/len(mu), glog_var/len(log_var)

class KullbackLeiblerDivergence2(CompositFunction):
    def _forward(self, mu, log_var):
        kll = -0.5 * F.sum(1 + log_var - mu**2 - F.exp(log_var))
        return kll / len(mu)

class MutualInformationLoss(CompositFunction):
    def _forward(self, z, mu, log_var):
        log_qz_cond_x = self.log_normal_density(z, mu, log_var)
        log_pz = self.log_standard_normal(z)
        mi_loss = F.mean(log_qz_cond_x - log_pz, axis=-1)
        return mi_loss

    def log_normal_density(self, z, mu, log_var):
        normalization = -0.5 * (F.log(2 * np.pi) + log_var)
        log_density = normalization - 0.5 * ((z - mu) ** 2 / F.exp(log_var))
        return F.sum(log_density, axis=-1)

    def log_standard_normal(self, z):
        log_density = -0.5 * z ** 2 - 0.5 * F.log(2 * np.pi)
        return F.sum(log_density, axis=-1)


# -- 潜在変数をサンプリングする層 仮版
class LatentLayer(BaseLayer):
    def __init__(self, *configuration, **kwargs):
        if len(configuration) == 2:
            m, n = configuration
        if len(configuration) == 1:
            m = None; n, = configuration
        self.config = m, n
        super().__init__(**kwargs)
        self.kl_loss = lf.KullbackLeiblerDivergence()
        self.rate      = kwargs.pop('rate', 1.0)      # サンプリングの広がり        
        self.r_kl_loss = kwargs.pop('r_kl_loss', 1.0) # kl_lossの混ぜ具合

    def fix_configuration(self, shape):
        m = 1
        for i in shape[1:]:
            m *= i
        self.config = m, self.config[1]

    def get_parameter_size(self):
        m, n = self.config
        return m, n*2 # m×2n 大きさに注意 

    def forward(self, x, train=False, kl_loss=False):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        m, n = self.config
        self.x = x
        w, b, gamma = self.parameters()    
        y = self.dot_linear.forward(self.x, w, b, gamma)
        y = super().__forward__(y, train=train, dropout=dropout)
        self.mu = self.y[:, :n]
        self.log_var = self.y[:, n:]
        self.epsilon = np.random.randn(*self.log_var.shape) * self.rate
        self.z = self.mu + self.epsilon * np.exp(self.log_var/2)
        kll = self.kl_loss.forward(self.mu, self.log_var) 
        if not kl_loss:
            return self.z
        else:
            return self.z, kll
    
    def backward(self, grad_z, flush=True):
        dldmu, dldlog_var = self.kl_loss.backward()
        # デコーダからのgrad_zとkl_lossの逆伝播の和
        delta_mu      = grad_z \
                      + dldmu * self.r_kl_loss
        delta_log_var = 0.5 * grad_z * self.epsilon * np.exp(self.log_var/2) \
                      + dldlog_var * self.r_kl_loss
        grad_y = np.hstack((delta_mu, delta_log_var))
        grad_y = super().__backward__(grad_y)
        grad_x, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)        
        self.parameters.set_gradient(grad_w, grad_b, ggamma, flush=flush)
        return grad_x

#### 時系列データをまとめて処理するRNN層(Truncated BPTT方式) ##################
# m:Vector_size(入力数)、n:hidden_size(ニューロン数)

class AffineParametersForRNN:
    """ RNNの重みやバイアスを管理する """
    def __init__(self, layer, **kwargs):
        self.layer       = layer
        self.init_method = kwargs.pop('method', 'Orthogonal') # 重みの初期化手段
        self.width       = kwargs.pop('width',       None) # 重みの初期値の広がりを指定
        self.debug_mode  = kwargs.pop('debug_mode', False) # 重みを一律に初期化
        optimize         = kwargs.pop('optimize',   'SGD') # 最適化関数の指定 

        self.optimizer_w = cf.eval_in_module(optimize, Optimizers, **kwargs)
        self.optimizer_v = cf.eval_in_module(optimize, Optimizers, **kwargs)
        self.optimizer_b = cf.eval_in_module(optimize, Optimizers, bias=True, **kwargs)

        self.w, self.v, self.b = None, None, None 
        
    def __call__(self):
        if self.w is None:
            self.init_parameter()
        return self.w, self.v, self.b    

    def init_parameter(self): 
        l, m, n = self.layer.get_parameter_size() # l:戻りパス、m:入力、n:ニューロン数
        if l is None or m is None or n is None:
            raise Exception('Configuration is not fixed.', self.__class__.__name__)

        kwargs = {'method':self.init_method, 'debug_mode':self.debug_mode}     

        if self.layer.__class__.__name__=='RNN':
            self.w = init_weight((m, n), **kwargs)
            self.v = init_weight((l, n), **kwargs)
            self.b = np.zeros(n).astype(Config.dtype)
            
        elif self.layer.__class__.__name__=='LSTM':    
            wgf = init_weight((m, n), **kwargs) # 忘却
            wgi = init_weight((m, n), **kwargs) # 入力
            wgo = init_weight((m, n), **kwargs) # 出力
            wnm = init_weight((m, n), **kwargs) # 新記憶
            vgf = init_weight((l, n), **kwargs) # 忘却
            vgi = init_weight((l, n), **kwargs) # 入力
            vgo = init_weight((l, n), **kwargs) # 出力
            vnm = init_weight((l, n), **kwargs) # 新記憶
            bgf = np.ones(n).astype(Config.dtype)
            bgi = np.zeros(n).astype(Config.dtype)
            bgo = np.zeros(n).astype(Config.dtype)
            bnm = np.zeros(n).astype(Config.dtype)
            self.w = np.concatenate([wgf, wgi, wgo, wnm], axis=1)
            self.v = np.concatenate([vgf, vgi, vgo, vnm], axis=1)
            self.b = np.concatenate([bgf, bgi, bgo, bnm])
             
        elif self.layer.__class__.__name__=='GRU':    
            wgz = init_weight((m, n), **kwargs) # 忘却
            wgr = init_weight((m, n), **kwargs) # 入力
            wnm = init_weight((m, n), **kwargs) # 新記憶
            vgz = init_weight((l, n), **kwargs) # 忘却
            vgr = init_weight((l, n), **kwargs) # 入力
            vnm = init_weight((l, n), **kwargs) # 新記憶
            bgz = np.ones(n).astype(Config.dtype)
            bgr = np.zeros(n).astype(Config.dtype)
            bnm = np.zeros(n).astype(Config.dtype)
            self.w = np.concatenate([wgz, wgr, wnm], axis=1)
            self.v = np.concatenate([vgz, vgr, vnm], axis=1)
            self.b = np.concatenate([bgz, bgr, bnm])

        else:
            raise NotImplementedError(self.layer.__class__.__name__+' is not valid.')
        
    def update(self, eta=0.001, **kwargs):
        self.optimizer_w.update(self.w, self.grad_w, eta, **kwargs)
        self.optimizer_v.update(self.v, self.grad_v, eta, **kwargs)
        self.optimizer_b.update(self.b, self.grad_b, eta, **kwargs)

    def set_gradient(self, grad_w, grad_v, grad_b, flush=True):
        if flush:
            self.grad_w = grad_w
            self.grad_v = grad_v
            self.grad_b = grad_b
        else:
            self.grad_w += grad_w
            self.grad_v += grad_v
            self.grad_b += grad_b
            
    def flush_gradient(self):
        self.grad_w = np.zeros_like(self.w, dtype=Config.dtype)
        self.grad_v = np.zeros_like(self.v, dtype=Config.dtype)
        self.grad_b = np.zeros_like(self.b, dtype=Config.dtype)
        
class RnnBaseLayer(Function):
    """
    Layerは時系列の展開を担う．重みやバイアスはAffineParametersForRNNに委託
    時系列の展開に際して機能cellをインスタンス化してLayer内に蓄積する
    いっぽう機能cellではforwardの際の入出力や内部の状態を自身と一体で保持し、
    backwardの際にLayerに蓄積されたcellを呼出せばforwardの際の変数がそのまま使える
    すなわち、Layer側では時刻と対応するcellを関係づけるのみで良く、
    他方cell側では時系列を気にせずにforwardでの変数をbackwardで呼出すだけで良い　

    """
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        self.cell = None
        stateful = False
        if len(configuration) == 3:
            m, n, stateful = configuration
        elif len(configuration) == 2:
            m, n = configuration
        elif len(configuration) == 1:
            m = None; n, = configuration
        elif len(configuration) == 0:
            m = None; n = 100
        else:
            raise Exception('Wrong configuration specified.')
        stateful        = kwargs.pop('stateful', stateful)
        self.config = m, n, stateful 
        print('Initialize', self.__class__.__name__,
                            self.config[:2], 'stateful', self.config[2])
        
        self.parameters = AffineParametersForRNN(self, **kwargs)
        
        self.cell_normalization = kwargs.pop('cell_normalization', False)
                                                # セル正規化(層正規化相当)の適用有無
        self.CPT = Capture()
        self.DO  = Dropout()

        self.last_state = None
        self.r0, self.c0 = None, None
        self.grad_r0, self.grad_c0 = None, None # seq2seqなどの場合に外部から参照
        self.r0_given, self.c0_given = False, False
        self.mask = None    
        
    def fix_configuration(self, shape):
        self.config[0] = shape[-1]
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def get_parameter_size(self):
        # 実際のサイズはCellの種類による
        m, n, stateful = self.config
        return n, m, n 

    def set_state(self, r0, c0=None):
        self.r0, self.c0 = r0, c0     # last_state
        #self.flush_gradient()
        #self.ys = []

    def reset_state(self):
        self.r0, self.c0 = None, None # last_state
        #self.flush_gradient()
        
    def flush_gradient(self):
        self.parameters.flush_gradient()
        
    def update(self, eta=0.001, **kwargs):
        self.parameters.update(eta=eta, **kwargs)

    def __forward__(self, x, r0=None, c0=None, *, mask=None, CPT=None, dropout=0.0):
        """
        順伝播で一時刻に一つcellのインスタンスを生成しlayerを形成する
        順伝播の際、入出力や内部状態はcellと一体で時刻毎にlayerに記憶される
        r0/c0は初回は零で、statefulならば2回目以降はそのまま使う
        最後の時刻のrt/ctをr0/c0に保存し、statefulならば次回の順伝播で使う

        """
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)

        w, v, b = self.parameters()
        
        # -- r0とc0を初期化し、yの器と、cellの器layerを用意
        B, T, m = x.shape          # B:バッチサイズ、T:時系列長
        _, n, stateful = self.config
        if r0 is not None and r0.shape==(B, n):    # 外から設定
            self.r0_given = True
        elif r0 is None and (not stateful or self.r0 is None): # 初期値を設定
            r0 = np.zeros((B, n), dtype=Config.dtype)
        elif stateful and self.r0 is not None and self.r0.shape==(B, n): # 前の値を継承
            r0 = self.r0
        else:
            print(self.config, x.shape, self.r0.shape)
            raise Exception(self.__class__.__name__+' state held inconsistent. May need reset_state().')
        if c0 is not None and c0.shape==(B, n):    # 外から設定
            self.c0_given = True
        elif c0 is None and (not stateful or self.c0 is None): # 初期値を設定
            c0 = np.zeros((B, n), dtype=Config.dtype)
        elif stateful and self.c0 is not None and self.c0.shape==(B, n): # 前の値を継承
            c0 = self.c0
        else:
            print(self.config, x.shape, self.c0.shape)
            raise Exception(self.__class__.__name__+' state held inconsistent. May need reset_state().')
        y = np.empty((B, T, n), dtype=Config.dtype)
        self.layer = []

        # 仮テスト
        #mask = np.ones((B, T), dtype=bool)
 
        if mask is not None:
            if x.shape[:-1]==mask.shape:
                self.mask = mask.astype(x.dtype)
            else:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match input shape {x.shape}" \
                    + self.__class__.__name__)
        
        # -- 時系列を展開して順伝播 --
        rt = r0.copy()
        ct = c0.copy()
        for t in range(T):
            cell = self.cell(self.cell_normalization) # 選択したユニットのインスタンス生成
            xt = x[:, t, :]
            rtt, ctt = cell.forward(xt, rt, ct, w, v, b) # rt,ct上書き
            if mask is None:
                rt, ct = rtt, ctt
            else:
                mt = self.mask[:, t, None]
                rt = mt * rtt + (1 - mt) * rt
                ct = mt * ctt + (1 - mt) * ct
            y[:, t, :] = rt
            self.layer.append(cell)

        self.r0 = rt                       # 最後の時刻の出力　
        self.c0 = ct                       # 最後の時刻の出力
        #self.r0[...] = rt                 # 属性継承して値を最後の時刻の出力で更新　
        #self.c0[...] = ct                 # 属性継承して値を最後の時刻の出力で更新
        
        y = self.CPT.forward(y, width=CPT) # 一部出力のみの使用に対応
        # CPT指定した場合はドロップアウトしない
        y = self.DO.forward(y, dropout=dropout if CPT is None else 0.0)
        # 出力はドロップアウト対象で隠れ状態は非対象
        return y

    def __backward__(self, grad_y, flush=True): # grad_yは下流から受け取る勾配
        """
        順伝播の際にlayerに時刻毎のcellと変数が保存されているから、
        それを順伝播とは逆順に呼出して順伝播に準じた手順で逆伝播を行う
        xの勾配は器を用意して対応する時刻にはめ込んでいくいっぽう、
        w,v,bの勾配は、時系列に亘り算出した値を累積して、それを保存する
        最後に算出したrt/ctの勾配はr0/c0の勾配として保存する

        """
        # ドロップアウトした出力の勾配は逆伝播しないが隠れ状態からの勾配は逆伝播する
        grad_y = self.DO.backward(grad_y) 
        # 一部の時刻をキャプチャした場合の時系列長の調整はCaptureクラスが担う
        grad_y = self.CPT.backward(grad_y)     # 出力(下流)から内部へ遡上する勾配
        B, T, n = grad_y.shape                 # 時系列長Tが欲しい
        m, _, _ = self.config

        w, v, b = self.parameters()

        # -- rとcの勾配を初期化し、xの勾配の器を用意
        if flush:
            self.parameters.flush_gradient()

        grad_x = np.empty((B, T, m), dtype=Config.dtype)
        grad_rt = np.zeros_like(self.r0, dtype=Config.dtype)
        grad_ct = np.zeros_like(self.c0, dtype=Config.dtype)

        # -- 時系列を呼出して逆伝播 --
        for t in reversed(range(T)):
            cell = self.layer[t]
            grad_yt = grad_y[:, t, :] + grad_rt  # 出力からとリカレントを合算

            if self.mask is None:
                grad_xt, grad_rt, grad_ct, grad_wt, grad_vt, grad_bt = \
                    cell.backward(grad_yt, grad_ct, w, v, b) # grad_rt,grad_ct上書き 　

            else:
                mt = self.mask[:, t, None]  # shape: (B, 1)
                # maskが1の位置だけに対応する勾配を渡す
                grad_xt, grad_rtt, grad_ctt, grad_wt, grad_vt, grad_bt = \
                    cell.backward(mt*grad_yt, mt*grad_ct, w, v, b)
                # maskが1なら新しい勾配を、0ならそのまま伝搬
                grad_rt = grad_rtt + (1 - mt) * grad_rt # grad_yt? # mt==0に対応する位置はgrad_rtt==0
                grad_ct = grad_ctt + (1 - mt) * grad_ct # mt==0に対応する位置はgrad_ctt==0

            grad_x[:, t, :] = grad_xt
            # 展開中は勾配をparametersに直接蓄積する(flushしてはいけない)
            self.parameters.set_gradient(grad_wt, grad_vt, grad_bt, flush=False)

        self.grad_r0 = grad_rt # 最後に算出するgrad_rtはr0の勾配
        self.grad_c0 = grad_ct # c0の勾配は使わないが念のため

        if not self.r0_given:
            return grad_x
        if not self.c0_given:
            self.r0_given = False
            return grad_x, self.grad_r0
        self.c0_given = False
        return grad_x, self.grad_r0, self.grad_c0

    def step_and_stack(self, x):
        """ １時刻ずつデータを処理して状態を蓄積 """
        m, n, stateful = self.config  # B, m = x.shape; B, n = t.shape
        if x.ndim==2:
            B, _ = x.shape
        else:
            B = 1
            x = x.reshape(B, m)

        w, v, b = self.parameters()

        # -- r0とc0を初期化 --
        if self.r0 is None:
            self.r0 = np.zeros((B, n), dtype=Config.dtype)
        if self.c0 is None:
            self.c0 = np.zeros((B, n), dtype=Config.dtype)
        
        # -- リカレントにr0をセットし、cellを起こして順伝播 --
        r = self.r0
        c = self.c0
        cell = self.cell() # 選択したユニットのインスタンス生成
        y, c = cell.forward(x, r, c, w, v, b)
        self.layer.append(cell)
        self.r0 = y
        self.c0 = c

        return y

    def init_parameter(self):    
        raise Exception('Invalid configuration')

#### 時系列データをまとめて処理するN層(Truncated BPTT方式) ##################
"""
 m:Vector_size(入力数)、n:hidden_size(ニューロン数)
 cellのインスタンス化はforwardで時系列展開に際して行うため、
 cell内でnormalizationの有無に応じた処理分けが出来ない．
 やむなくnormalization無しと有りに分けたcellを用意して呼び分ける

"""
class RNN(RnnBaseLayer):
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.cell = RNN_Cell

class LSTM(RnnBaseLayer):
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.cell = LSTM_Cell

class GRU(RnnBaseLayer):
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.cell = GRU_Cell

               
#### RNN各種機能ユニット ###################################################
#### w,v,bは時系列共通、layer側で保持
        
class RNN_Cell:
    def __init__(self, normalize=False):
        self.dual_dot_linear = F.DualDotLinear()
        if normalize:
            self.norm = F.Normalize(axis=-1)
        else:
            self.norm = None
        self.normalize = normalize
        self.tanh = F.Tanh()
        
    def forward(self, x, r, c, w, v, b):   # cは使わない
        u = self.dual_dot_linear.forward(x, r, w, v, b)
        if self.normalize:
            u = self.norm.forward(u)       # 正規化
        y = self.tanh.forward(u)                   
        self.state = x, r, y               # rは前時刻のy
        return y, c                        # 出力,cはそのまま

    def backward(self, grad_y, grad_c, w, v, b): # c,g関連は使わない
        x, r, y = self.state
        delta = self.tanh.backward(grad_y)
        if self.normalize:
            delta = self.norm.backward(delta)  
        grad_x, grad_r, grad_w, grad_v, grad_b = self.dual_dot_linear.backward(delta)
        return grad_x, grad_r, grad_c, grad_w, grad_v, grad_b 

class LSTM_Cell:
    def __init__(self, normalize=False):
        self.dual_dot_linear = F.DualDotLinear()
        if normalize:
            self.norm_g = F.Normalize(axis=-1)
            self.norm_c = F.Normalize(axis=-1)
        else:
            self.norm_g = None
            self.norm_c = None
        self.normalize = normalize
        
    def forward(self, x, r, cp, w, v, b):   # 入力、前時刻状態
        B, n = r.shape
        u = self.dual_dot_linear.forward(x, r, w, v, b)
        if self.normalize:
            u = self.norm_g.forward(u)      # 諸ゲートの正規化
        gz = 1 / (1 + np.exp(-u[:, :3*n]))  # sigmoid 諸ゲート　
        gm = np.tanh(u[:, 3*n:])            # tanh  新しい記憶
        gf = gz[:, :n]                      # 忘却ゲート
        gi = gz[:, n:2*n]                   # 入力ゲート
        go = gz[:, 2*n:]                    # 出力ゲート
        cn = cp * gf + gm * gi              # 旧記憶＊忘却ゲート＋新記憶＊入力ゲート
        if self.normalize:
            cn = self.norm_c.forward(cn)    # 記憶の正規化
        y = np.tanh(cn) * go                # 記憶＊出力ゲート
        g = np.hstack((gz, gm))             # 内部状態
        self.state = x, r, cp, y, cn, g 
        return y, cn                        # 出力

    def backward(self, grad_y, grad_cn, w, v, b): # 引数＝下流からの勾配
        x, r, cp, y, cn, g = self.state
        B, n = r.shape
        gz = g[:,:3*n]                      # 忘却ゲート、入力ゲート、出力ゲート             
        gf = g[:,:n]                        # 忘却ゲート
        gi = g[:,n:2*n]                     # 入力ゲート
        go = g[:,2*n:3*n]                   # 出力ゲート
        gm = g[:,3*n:]                      # 新しい記憶
        tanh_c = np.tanh(cn)
        gcn = grad_cn + (grad_y * go) * (1 - tanh_c ** 2)
        if self.normalize:
            gcn = self.norm_c.backward(gcn) # 記憶の正規化の逆伝播
        
        dgm = gcn * gi                      # 新しい記憶の勾配

        # 諸ゲートの勾配： 忘却 dgf  入力 dgi  出力 dgo　 　　　　　　　　　　　　　
        dgz = np.hstack((gcn * cp, gcn * gm, grad_y * tanh_c))

        # 諸ゲート sigmoidの微分 と 新しい記憶  tanhの微分
        delta = np.hstack((dgz * gz * (1 - gz), dgm * (1 - gm ** 2)))
        if self.normalize:
            delta = self.norm_g.backward(delta) # 諸ゲートの正規化の逆伝播
            
        grad_cp = gcn * gf                    
        grad_x, grad_r, grad_w, grad_v, grad_b \
                            = self.dual_dot_linear.backward(delta)
        return grad_x, grad_r, grad_cp, grad_w, grad_v, grad_b 

class GRU_Cell:
    def __init__(self, normalize=False):
        self.dual_dot_linear_g = F.DualDotLinear()
        self.dual_dot_linear_m = F.DualDotLinear()
        if normalize:
            self.norm_g = F.Normalize(axis=-1)
            self.norm_u = F.Normalize(axis=-1)
        self.normalize = normalize
        
    def forward(self, x, r, c, w, v, b): # cは使わない
        B, n = r.shape
        wg, wm = w[:, :2*n], w[:, 2*n:]
        vg, vm = v[:, :2*n], v[:, 2*n:]
        bg, bm = b[:2*n],    b[2*n:]
   
        # 更新ゲートとリセットゲート
        #gu = np.dot(x, wg) + np.dot(r, vg) + bg
        gu = self.dual_dot_linear_g.forward(x, r, wg, vg, bg)
        if self.normalize:
            gu = self.norm_g.forward(gu)   # 諸ゲート正規化
        
        g = 1 / (1 + np.exp(-gu))          # sigmoid
        gz, gr = g[:, :n], g[:, n:]        # 更新ゲートとリセットゲート
        # 新しい記憶
        #u = np.dot(x, wm) + np.dot(gr * r, vm) + bm
        u = self.dual_dot_linear_m.forward(x, gr*r, wm, vm, bm)
        if self.normalize:
            u = self.norm_u.forward(u)     # 記憶正規化
        
        y = (1 - gz) * r + gz * np.tanh(u)
        self.state = x, r, y, g 
        return y, c                        # 出力,cはそのまま
    
    def backward(self, grad_y, grad_c, w, v, b): # c関連は使わない 
        x, r, y, g = self.state
        B, n = r.shape
        wg, wm = w[:, :2*n], w[:, 2*n:]
        vg, vm = v[:, :2*n], v[:, 2*n:]
        gz, gr = g[:, :n],   g[:, n:]
        
        # y算出の逆伝播
        tanh_u  = (y - (1 - gz) * r) / gz
        grad_r  = grad_y * (1 - gz) 
        grad_gz = grad_y * (tanh_u - r) ## 修正
        
        # 新しい記憶　
        delta_m = grad_y * gz * (1 - tanh_u ** 2) # 修正
        if self.normalize:
            delta_m = self.norm_u.backward(delta_m) 

        grad_x, grad_rm, grad_wm, grad_vm, grad_bm \
                        = self.dual_dot_linear_m.backward(delta_m)
        # gr * r の逆伝播 
        grad_r += gr * grad_rm
        grad_gr = grad_rm * r 

        # 更新ゲートとリセットゲート
        delta_g = np.hstack((grad_gz, grad_gr)) * g * (1 - g) # sigmoidの微分
        if self.normalize:
            delta_g = self.norm_g.backward(delta_g)
        
        grad_xm, grad_rm, grad_wg, grad_vg, grad_bg \
                        = self.dual_dot_linear_g.backward(delta_g)
        grad_r += grad_rm
        grad_x += grad_xm
        grad_w  = np.hstack((grad_wg, grad_wm))
        grad_v  = np.hstack((grad_vg, grad_vm))
        grad_b  = np.hstack((grad_bg, grad_bm))
        
        return grad_x, grad_r, grad_c, grad_w, grad_v, grad_b    


#### Embedding層用のparameter管理クラス #######################
# w:その行に対応する語のベクトルを各行が示す(行数m=語彙数、列数n=語ベクトル長)
#   全体で単語の分散表現
class ParametersForEmbedding:
    def __init__(self, layer, **kwargs):
        self.layer      = layer
        self.method     = kwargs.pop('method', 'uniform')
        self.width      = kwargs.pop('width',       None)
        self.debug_mode = kwargs.pop('debug_mode', False) # 重みを一律に初期化
        optimize        = kwargs.pop('optimize',   'SGD') 
        self.w, self.grad_w = None, None
        self.optimizer_w = cf.eval_in_module(optimize, Optimizers, **kwargs)

        # 環境に応じた関数の選択
        try:
            self.add_at = np.add.at
            print('embeddingには add.at を使う')
        except:
            try:
                self.add_at = np.scatter_add
                print('embeddingには scatter_add を使う')
            except:
                try:
                    self.add_at = np._cupyx.scatter_add
                    print('embeddingには cupyxのscatter_add を使う')
                except:
                    def f(x, y, z): # xのyの位置にzを加算する
                        for i, idx in enumerate(y):
                            x[idx] += z[i]
                    self.add_at = f        
                    print('embeddingはforループで関数を定義して使う')

    def __call__(self):
        if self.w is None:
            self.init_parameter()
        return self.w    

    def init_parameter(self):
        # 通常のニューロンとは違い、出力の次元数に応じた一様乱数で初期化(u(-√1/D,√1/D))
        m, n = self.layer.get_parameter_size()
        if self.method == 'uniform':
            width = np.sqrt(1/n) if self.width is None else self.width
            self.w = init_weight((m, n),
                                 distribution='uniform',
                                 width=width,
                                 debug_mode=self.debug_mode)
        elif self.method == 'Orthogonal':
            #width = np.sqrt(1/n) if self.width is None else self.width
            self.w = init_weight((m, n), method='Orthogonal', debug_mode=self.debug_mode)
        else:
            raise Exception('Invalid method for ' + self.__class__.__name__+' specified.')

    def update(self, eta=0.001, **kwargs):
        self.optimizer_w.update(self.w, self.grad_w, eta=eta, **kwargs)

    def accommodate(self):
        m, n = self.layer.get_parameter_size()
        if m <= self.w.shape[0]:
            return
        print(self.__class__.__name__, 'expand the size of w to accommodate new vocabulary.')
        xpcn = m - self.w.shape[0] # 拡張する行数
        center = np.mean(self.w, axis=0)
        new_rows = center + np.random.normal(0, 0.01, size=(xpcn, n), dtype=Config.dtype)
        print('new_rows =', new_rows.shape)
        self.w = np.concatenate([self.w, new_rows], axis=0)
        
    def set_gradient(self, x, gy, flush=True): # 未
        if flush:
            self.grad_w = np.zeros_like(self.w, dtype=Config.dtype)
        elif self.grad_w.shape == self.w.shape:
            pass
        self.add_at(self.grad_w, x, gy)
           
            
#### 時系列データをまとめて処理する Embedding層 #######################
# m:vocab_size(語彙数)、n:wordvec_size(語ベクトル長)
class Embedding(Function):
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        if len(configuration) == 2:
            m, n = configuration
        if len(configuration) == 1:
            m = 10000; n, = configuration
        if len(configuration) == 0:
            m = 10000; n = 100
        self.config = m, n    
        print('Initialize', self.__class__.__name__, self.config)
        self.parameters  = ParametersForEmbedding(self, **kwargs)
        self.mask = None            

    def get_parameter_size(self):
        return self.config

    def update(self, eta=0.001, **kwargs):
        self.parameters.update(eta=eta, **kwargs)

    def accommodate(self):
        self.parameters.accommodate()
        
    def __forward__(self, x, *, mask=None, dropout=0.0):
        """
        入力 x は w のどの行を抽出するかを示し
        yはxの指すwの行、xの形状(B,T)に対し、yの形状は(B, T, n)
        即ち長さnのベクトルがバッチ数×展開時間だけ並ぶ
        
        """
        w = self.parameters()
        y = w[x]
       
        if mask is None:
            self.mask = None
            return y
        if x.shape==mask.shape:
            self.mask = mask[..., None].astype(y.dtype)
            return y * self.mask 
        raise ValueError(f"Mask shape {mask.shape} does not match input shape {x.shape}" \
                           + self.__class__.__name__)

    def __backward__(self, gy):
        x, = self.inputs
        if self.mask is not None:
            gy *= self.mask
        self.parameters.set_gradient(x, gy)
            
        
#### 位置符号化 ####################################################　   
class PositionalEmbedding: 
    """ 入力の値に対する埋め込みと、その位置インデクスに対する埋め込みを合せて出力する """
    def __init__(self, vocab_size=10000, block_size=500, dimension=64, noise=0, **kwargs):
        self.token_embedding = Embedding(vocab_size, dimension, **kwargs)
        self.position_embedding = Embedding(block_size, dimension, **kwargs)
        self.block_size = block_size
        self.broadcast_to = None # forwardで形状が決まってから設定
        self.noise = noise
        
    def forward(self, x):
        """ 順伝播：positionはblock_sizeの範囲の0から始まる値でxと同形状 """
        token_vector = self.token_embedding(x)
        position = np.arange(x.shape[-1]).reshape(1, -1) % self.block_size
        position_vector = self.position_embedding(np.broadcast_to(position, x.shape))
        y = token_vector + position_vector # 出力をインプレース更新で作るとbacktrace失敗0250316AI
        if self.noise > 0:
            y += np.random.randn(*y.shape) * self.noise
        self.y = y    
        return y 

    def backward(self, gy=None):
        """ 逆伝播：入力へ勾配は伝搬しないから形状や剰余などの対応不要 """
        if gy is None:
            gy = np.ones_like(self.y)
        self.position_embedding.backward(gy) 
        self.token_embedding.backward(gy)
        
    def update(self, **kwargs):
        self.token_embedding.update(**kwargs)
        self.position_embedding.update(**kwargs)

class PositionalEncoding:
    def __init__(self, sequence_length=10000, dimension=2):
        self.dimension = dimension
        self.sequence_length = sequence_length
        self.division \
            = np.exp(np.arange(0, dimension, 2) * -(np.log(sequence_length) / dimension))
        # 上記はdivisionを決める際にsequence_lengthを見ているが元は下記
        #   = np.exp(np.arange(0, dimension, 2) * -(np.log(10000) / dimension))

        print('シーケンス長', sequence_length, '次元数', dimension, '分割', self.division)

    def __call__(self, positions):
        positions_shape = positions.shape
        positions = positions.reshape(-1, 1)
        pe = np.zeros((positions.shape[0],) + (self.dimension,)) # 末尾の次元を入替
        pe[:, 0::2] = np.sin(positions * self.division) # 偶数次元に対するサイン関数の適用
        pe[:, 1::2] = np.cos(positions * self.division) # 奇数次元に対するコサイン関数の適用
        pe = pe.reshape(positions_shape + (self.dimension,)) # 元の次元に末尾を加えた形状
        return pe

class PositionalEmbedding2: # 逆伝播が書けない
    """ 入力の値に対する埋め込みと、その位置インデクスに対する埋め込みを合せて出力する """
    def __init__(self, vocab_size=10000, block_size=500, dimension=64, **kwargs):
        self.token_embedding    = Embedding(vocab_size, dimension, **kwargs)
        self.position_embedding = Embedding(block_size, dimension, **kwargs)
        self.block_size = block_size
        
    def forward(self, x):
        token_vactor = self.token_embedding(x)
        position = np.arange(x.shape[-1]) % self.block_size # 位置インデクスのブロック長の範囲の値
        position_vector = self.position_embedding(position)
        y = token_vactor + position_vector
        return y

    def update(self, **kwargs):
        self.token_embedding.update(**kwargs)
        self.position_embedding.update(**kwargs)

class PositionalEmbedding_bkup: # broadcast_toはおかしいが、取りあえず残しておく20250315AI
    """ 入力の値に対する埋め込みと、その位置インデクスに対する埋め込みを合せて出力する """
    def __init__(self, vocab_size=10000, block_size=500, dimension=64, **kwargs):
        self.token_embedding = Embedding(vocab_size, dimension, **kwargs)
        self.position_embedding = Embedding(block_size, dimension, **kwargs)
        self.block_size = block_size
        self.broadcast_to = None # forwardで形状が決まってから設定
        
    def forward(self, x):
        tok_emb = self.token_embedding(x)
        position = np.arange(x.shape[-1]) % self.block_size # 位置インデクスのブロック長の範囲の値
        pos_emb = self.position_embedding(position)
        #print('### debug', self.__class__.__name__, x.shape, tok_emb.shape, pos_emb.shape)
        self.broadcast_to = F.BroadcastTo(tok_emb.shape)     # バッチ軸を拡張
        pos_emb = self.broadcast_to(pos_emb)
        #print('###', pos_emb.shape)
        y = tok_emb + pos_emb
        self.y = y
        return y

    def backward(self, gy=None):
        if gy is None:
            gy = np.ones_like(self.y)
        #gz = np.sum(gy, axis=0)
        gz = self.broadcast_to.backward(gy)
        self.position_embedding.backward(gz) # 入力へ勾配は伝搬しないから剰余などの対応不要
        self.token_embedding.backward(gy)    # こちらも入力へ勾配は伝搬しない

    def update(self, **kwargs):
        self.token_embedding.update(**kwargs)
        self.position_embedding.update(**kwargs)

#### Attention機構 #################################################
# v:入力 value、k:入力 key、q:入力 query、y:出力、a:attention_weight
# query に一致する key を探して、その key に対応する value を出力する
# mask:self attentionで時系列を扱う場合に自身より先の時刻を無視するなど
# scale:コンテクストベクトルが大きい場合にaが大きくなりすぎるのを抑制
# 
class AttentionUnit_bkup(Function):
    """ 汎用AttentionUnit(MultiHead対応) """
    def __init__(self, head=1, **kwargs): 
        super().__init__()
        print('Initialize', self.__class__.__name__, 'head =', head, kwargs)
        self.head = head
        causality   = kwargs.pop('causality',  False)     # 時系列の前後関係 
        self.scale  = kwargs.pop('scale',       True)
        temperature = kwargs.pop('temperature',  1.0)  
        regularizer = kwargs.pop('regularizer', None)
       
        if type(regularizer) == str:
            self.regularizer = cf.eval_in_module(regularizer, Regularizers)
        else:
            self.regularizer = copy.deepcopy(regularizer) # インスタンス分離のために必須

        self.causality = causality
        self.softmax = Activators.Softmax(temperature=temperature)
        self.DO = Dropout()
        self.iter = 0
        self.loss = 0
        self.result1 = None
        self.result2 = None
        self.result3 = None
        self.tril = None # causalityの制御のマスク
        self.mask = None # 無効トークンのマスク
       
    def __forward__(self, v, k, q, *, mask=None, dropout=0.0):
        B,Tv,C = v.shape
        B,Tk,C = k.shape
        B,Tq,C = q.shape
        h = self.head
        H = C // h
        v = v.reshape(B,Tv,h,H).transpose(0,2,1,3) # (B,Tv,C)->(B,h,Tv,H)
        k = k.reshape(B,Tk,h,H).transpose(0,2,1,3) # (B,Tk,C)->(B,h,Tk,H)
        q = q.reshape(B,Tq,h,H).transpose(0,2,1,3) # (B,Tq,C)->(B,h,Tq,H)
        a = np.matmul(q, k.transpose(0,1,3,2))     # (B,h,Tq,H)@(B,h,H,Tk)->(B,h,Tq,Tk)
        if self.scale:
            a *= np.array(H ** -0.5, dtype=a.dtype)

        if self.causality: # 時間の前後関係の保証
            if self.tril is not None and self.tril.shape==(Tq,Tk):
                pass
            elif Tq==Tk:
                self.tril = np.tril(np.ones((Tq,Tk), dtype=bool))
            else:
                raise Exception(f"causality cannot be applied" + self.__class__.__name__)
            a[:,:,self.tril==False] = - Config.inf

        if mask is None:   # 無効トークンの処理
            self.mask = None
        elif mask.shape == (B, Tk):
            self.mask = mask.astype(bool)
            a.transpose(0,3,1,2)[self.mask==False,:,:] = - Config.inf # (B,Tk,h,Tq)

        else:
            raise ValueError(f"Mask shape {mask.shape} must be ({B}, {Tk})" \
                           + self.__class__.__name__)

        a = self.softmax(a)
        
        if self.regularizer is not None: # aのエントロピーやKLDの算出
            self.loss = self.regularizer.forward(a)
            self.result1 = self.regularizer.result1
            self.result2 = self.regularizer.result2
            self.result3 = self.regularizer.result3
        self.iter += 1 

        a = self.DO.forward(a, dropout=dropout)
        y = np.matmul(a, v)            # (B,h,Tq,Tk)@(B,h,Tv,H)->(B,h,Tq,H), Tv=Tk
        y = y.transpose(0,2,1,3).reshape(B,Tq,C)     # (B,h,Tq,H)->(B,Tq,C)
        self.k = k                                   # key
        self.q = q                                   # query
        self.v = v                                   # value
        self.a = a                                   # attention_weight
        self.y = y
        return y                                     # (B,Tq,C)

    def __backward__(self, gy):
        k = self.k
        q = self.q
        v = self.v
        a = self.a
        B, h, Tk, H = k.shape
        B, h, Tq, H = q.shape
        B, h, Tv, H = v.shape
        C = h * H
        gy = gy.reshape(B,Tq,h,H).transpose(0,2,1,3) # (B,Tq,C)->(B,h,Tq,H)
        ga = np.matmul(gy, v.transpose(0,1,3,2))     # (B,h,Tq,H)@(B,h,H,Tv)->(B,h,Tq,Tv) Tv=Tk
        gv = np.matmul(a.transpose(0,1,3,2), gy)     # (B,h,Tk,Tq)@(B,h,Tq,H)->(B,h,Tk,H) tk=Tv

        ga = self.DO.backward(ga)

        if self.regularizer is not None:          
            ga2 = self.regularizer.backward()
            ga += ga2                                # 勾配加算率はregularizer側に設定
        
        ga = self.softmax.backward(ga)
        
        if self.mask is not None:
            ga.transpose(0,3,1,2)[self.mask==False,:,:] = 0
        if self.causality:    
            ga[:,:,self.tril==False] = 0
        if self.scale:
            ga *= np.array(H ** -0.5, dtype=ga.dtype)

        gq = np.matmul(ga, k)                      # (B,h,Tq,Tk)@(B,h,Tk,H)->(B,h,Tq,H)
        #gk = np.matmul(q.transpose(0,1,3,2), ga)  # <-これはNG 20250525AI
        gk = np.matmul(ga.transpose(0,1,3,2), q)   # (B,h,Tk,Tq)@(B,h,Tq,H)->(B,h,Tk,H)
        gv = gv.transpose(0,2,1,3).reshape(B,Tv,C) # (B,h,Tv,H)->(B,Tv,C)
        gk = gk.transpose(0,2,1,3).reshape(B,Tk,C) # (B,h,Tk,H)->(B,Tk,C)
        gq = gq.transpose(0,2,1,3).reshape(B,Tq,C) # (B,h,Tq,H)->(B,Tq,C)
        return gv, gk, gq

class AttentionUnit(Function):
    """ 汎用AttentionUnit(MultiHead対応) """
    def __init__(self, head=1, **kwargs): 
        super().__init__()
        print('Initialize', self.__class__.__name__, 'head =', head, kwargs)
        self.head = head
        causality   = kwargs.pop('causality',  False)     # 時系列の前後関係 
        self.scale  = kwargs.pop('scale',       True)
        temperature = kwargs.pop('temperature',  1.0)  
        regularizer = kwargs.pop('regularizer', None)
       
        if type(regularizer) == str:
            self.regularizer = cf.eval_in_module(regularizer, Regularizers)
        else:
            self.regularizer = copy.deepcopy(regularizer) # インスタンス分離のために必須

        self.causality = causality
        self.softmax = Activators.Softmax(temperature=temperature)
        self.DO = Dropout()
        self.iter = 0
        self.loss = 0
        self.result1 = None
        self.result2 = None
        self.result3 = None
        self.tril = None # causalityの制御のマスク
        self.mask = None # 無効トークンのマスク
       
    def __forward__(self, v, k, q, *, mask=None, dropout=0.0):
        B,Tv,C = v.shape
        B,Tk,C = k.shape
        B,Tq,C = q.shape
        h = self.head
        H = C // h
        v = v.reshape(B,Tv,h,H).transpose(0,2,1,3) # (B,Tv,C)->(B,h,Tv,H)
        k = k.reshape(B,Tk,h,H).transpose(0,2,1,3) # (B,Tk,C)->(B,h,Tk,H)
        q = q.reshape(B,Tq,h,H).transpose(0,2,1,3) # (B,Tq,C)->(B,h,Tq,H)
        a = np.matmul(q, k.transpose(0,1,3,2))     # (B,h,Tq,H)@(B,h,H,Tk)->(B,h,Tq,Tk)
        if self.scale:
            a *= np.array(H ** -0.5, dtype=a.dtype)

        if self.causality: # 時間の前後関係の保証
            if self.tril is not None and self.tril.shape[-2:]==(Tq,Tk):
                pass
            elif Tq==Tk:
                self.tril = np.tril(np.ones((1,1,Tq,Tk), dtype=Config.dtype))#[None, None, :, :]
            else:
                raise Exception(f"causality cannot be applied" + self.__class__.__name__)
            a += (self.tril - 1.0) * Config.inf
       
        if mask is None:   # 無効トークンの処理 
            self.mask = None
        elif mask.shape==(B, Tk):
            self.mask = mask.astype(a.dtype)[:, None, None, :]       # (B,1,1,Tk)
            a += (self.mask - 1.0) * Config.inf  # 無効個所は-inf
        else:
            raise ValueError(f"Mask shape {mask.shape} must be ({B}, {Tk})" \
                           + self.__class__.__name__)

        a = self.softmax(a)
        
        if self.regularizer is not None: # aのエントロピーやKLDの算出
            self.loss = self.regularizer.forward(a)
            self.result1 = self.regularizer.result1
            self.result2 = self.regularizer.result2
            self.result3 = self.regularizer.result3
        self.iter += 1 

        a = self.DO.forward(a, dropout=dropout)
        y = np.matmul(a, v)            # (B,h,Tq,Tk)@(B,h,Tv,H)->(B,h,Tq,H), Tv=Tk
        y = y.transpose(0,2,1,3).reshape(B,Tq,C)     # (B,h,Tq,H)->(B,Tq,C)
        self.k = k                                   # key
        self.q = q                                   # query
        self.v = v                                   # value
        self.a = a                                   # attention_weight
        self.y = y
        return y                                     # (B,Tq,C)

    def __backward__(self, gy):
        k = self.k
        q = self.q
        v = self.v
        a = self.a
        B, h, Tk, H = k.shape
        B, h, Tq, H = q.shape
        B, h, Tv, H = v.shape
        C = h * H
        gy = gy.reshape(B,Tq,h,H).transpose(0,2,1,3) # (B,Tq,C)->(B,h,Tq,H)
        ga = np.matmul(gy, v.transpose(0,1,3,2))     # (B,h,Tq,H)@(B,h,H,Tv)->(B,h,Tq,Tv) Tv=Tk
        gv = np.matmul(a.transpose(0,1,3,2), gy)     # (B,h,Tk,Tq)@(B,h,Tq,H)->(B,h,Tk,H) tk=Tv

        ga = self.DO.backward(ga)

        if self.regularizer is not None:          
            ga2 = self.regularizer.backward()
            ga += ga2                                # 勾配加算率はregularizer側に設定
        
        ga = self.softmax.backward(ga)
        if self.mask is not None:
            ga *= self.mask
        if self.causality:    
            ga *= self.tril
        if self.scale:
            ga *= np.array(H ** -0.5, dtype=ga.dtype)

        gq = np.matmul(ga, k)                      # (B,h,Tq,Tk)@(B,h,Tk,H)->(B,h,Tq,H)
        #gk = np.matmul(q.transpose(0,1,3,2), ga)  # <-これはNG 20250525AI
        gk = np.matmul(ga.transpose(0,1,3,2), q)   # (B,h,Tk,Tq)@(B,h,Tq,H)->(B,h,Tk,H)
        gv = gv.transpose(0,2,1,3).reshape(B,Tv,C) # (B,h,Tv,H)->(B,Tv,C)
        gk = gk.transpose(0,2,1,3).reshape(B,Tk,C) # (B,h,Tk,H)->(B,Tk,C)
        gq = gq.transpose(0,2,1,3).reshape(B,Tq,C) # (B,h,Tq,H)->(B,Tq,C)
        return gv, gk, gq

#### 時系列データをまとめて処理する Attention層 ############################
# q:入力 query、x:入力 keyとvalue、y:出力、w:attention_weight
# query に一致する key を探して、その key に対応する value を出力する
# Seq2seq(sequence to sequence、時系列データ変換器)での使い方として;
#  key にエンコーダ出力を入れ、
#  デコーダの隠れ層から受け取った query に対応する
#  エンコーダ出力から選んだコンテクストベクトルを得る
# 
class SimpleAttentionLayer(Function):
    """
    入力はx:key&valueとq:query　　　　
    このLayerはAttentionUnitをそのまま使う　

    """
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        self.unit = AttentionUnit()
        print('Initialize', self.__class__.__name__)
        
    def __forward__(self, x, q, *, dropout=0.0):
        return self.unit.forward(x, x, q, dropout=dropout) # valueとkeyは同じものを与える

    def __backward__(self, grad_y):       # grad_yは下流から受け取る勾配
        grad_v, grad_k, grad_q = self.unit.backward(grad_y)
        grad_x = grad_v + grad_k          # valueとkeyに同じものを与えたことに対応
        return grad_x, grad_q

class SelfAttention(Function):
    """ multiple heads of self_attention in parallel """
    def __init__(self, emb_dim=None, head_dim=None, n_head=1,
                 #scale=True, temperature=1.0, entropy_decay=True,
                 **kwargs):
        super().__init__()
        if emb_dim is not None and head_dim is None:
            head_dim = emb_dim // n_head
        self.config = emb_dim, head_dim, n_head
        print('Initialize', self.__class__.__name__, self.config, kwargs)
        optimize = kwargs.pop('optimize',   'Adam') 
        # linear_iとlinear_oのconfigはfix_configurationで設定
        self.linear_i = LinearLayer(matmul=True, bias=False,
                                    #scale=True,
                                    optimize=optimize,
                                    #spctrnorm=1,
                                    **kwargs)

        self.attention = AttentionUnit(head=n_head, causality='tri', **kwargs)
                         #scale=scale, temperature=temperature, entropy_decay=entropy_decay)
        self.linear_o = LinearLayer(matmul=True, bias=True,
                                    #scale=True, 
                                    optimize=optimize,
                                    #spctrnorm=1,
                                    **kwargs)
        self.DO = Dropout()
        self.step = 0
        
    def fix_configuration(self, shape):
        emb_dim, head_dim, n_head = self.config
        if emb_dim is None:
            emb_dim = shape[-1]
        elif emb_dim != shape[-1]: # 予め与えられたemb_dimがデータと合わない
            raise Exception('Data shape mismatch with configuration.', self.__class__.__name__)
        if head_dim is None:
            head_dim = emb_dim // n_head
        self.config = emb_dim, head_dim, n_head    
        self.linear_i.config = emb_dim, emb_dim*3 
        self.linear_o.config = emb_dim, emb_dim
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def __forward__(self, x, *, mask=None, dropout=0.0):
        if None in (*self.config, *self.linear_i.config, *self.linear_o.config):
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        z = self.linear_i.forward(x)
        key, query, value = np.split(z, 3, axis=-1)
        y = self.attention.forward(value, key, query, mask=mask, dropout=dropout)
        y = self.linear_o.forward(y)
        y = self.DO.forward(y, dropout=dropout)
        self.y = y
        return y

    def __backward__(self, gy):
        gx = self.DO.backward(gy)    
        gx = self.linear_o.backward(gx)
        gv, gk, gq = self.attention.backward(gx)
        gz = np.concatenate([gk, gq, gv], axis=-1)
        gx = self.linear_i.backward(gz)
        return gx

    def update(self, eta=0.001, **kwargs):
        self.linear_i.update(eta=eta, **kwargs)
        self.linear_o.update(eta=eta, **kwargs)
        
    def entropy(self):
        return self.attention.entropy

class MultiHeadSelfAttention(SelfAttention):
    """ multiple heads of self_attention in parallel """
    def __init__(self, emb_dim=None, head_dim=None, n_head=1, **kwargs):
        print(self.__class__.__name__, emb_dim, head_dim, n_head, kwargs)
        super().__init__(emb_dim, head_dim, n_head, **kwargs)
        
class SingleHeadSelfAttention(SelfAttention):
    """ single head of self_attention """
    def __init__(self, emb_dim=None, head_dim=None, **kwargs):
        print(self.__class__.__name__, emb_dim, head_dim, kwargs)
        super().__init__(emb_dim, head_dim, 1, **kwargs)
        
class MultiHeadSelfAttention2(Function):
    """ 先にhead分割し、各SingleAttentionに配る """
    def __init__(self, emb_dim=None, head_dim=None, n_head=1,
                 scale=True, temperature=1.0, optimize='Adam', **kwargs):
        super().__init__()
        if emb_dim is not None and head_dim is None:
            head_dim = emb_dim // n_head
        else:
            head_dim, head_dim = None, None
        self.config = emb_dim, head_dim, n_head
        print('Initialize', self.__class__.__name__, self.config)
        # linear_iとlinear_oのconfigはfix_configurationで設定
        self.linear_v = LinearLayer(matmul=True, bias=False, optimize=optimize, **kwargs)
        self.linear_k = LinearLayer(matmul=True, bias=False, optimize=optimize, **kwargs)
        self.linear_q = LinearLayer(matmul=True, bias=False, optimize=optimize, **kwargs)
        self.attention = [AttentionUnit(causality='tri', scale=scale, temperature=temperature)
                                                        for _ in range(n_head)]
        self.linear_o = LinearLayer(matmul=True, bias=True, optimize=optimize, **kwargs)
        self.DO = Dropout()
        
    def fix_configuration(self, shape):
        emb_dim, head_dim, n_head = self.config
        if emb_dim is None:
            emb_dim = shape[-1]
        elif emb_dim != shape[-1]: # 予め与えられたemb_dimがデータと合わない
            raise Exception('Data shape mismatch with configuration.',
                                                self.__class__.__name__)
        if head_dim is None:
            head_dim = emb_dim // n_head
        assert emb_dim % n_head==0, "embedding dimension must be divisible by n_head"
        assert emb_dim // n_head == head_dim, "head_dim must be emb_dim // n_head"
        self.config = emb_dim, head_dim, n_head
        self.linear_v.config = emb_dim, emb_dim 
        self.linear_k.config = emb_dim, emb_dim 
        self.linear_q.config = emb_dim, emb_dim 
        self.linear_o.config = emb_dim, emb_dim
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def __forward__(self, x, *, dropout=0.0):
        if None in (*self.config,
                    *self.linear_v.config,
                    *self.linear_k.config,
                    *self.linear_q.config,
                    *self.linear_o.config):
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        emb_dim, head_dim, n_head = self.config    
        value = self.linear_v.forward(x)
        key   = self.linear_k.forward(x)
        query = self.linear_q.forward(x)
        # head分割
        value = np.split(value, n_head, axis=-1)
        key   = np.split(key,   n_head, axis=-1)
        query = np.split(query, n_head, axis=-1)
        # 各ヘッド独立　　　　
        y = [a.forward(v, k, q, dropout=dropout) for a, v, k, q \
                       in zip(self.attention, value, key, query)] 
        # 各ヘッド出力を併合して出力層へ
        y = np.concatenate(y, axis=-1)
        y = self.linear_o.forward(y)
        y = self.DO.forward(y, dropout=dropout)
        self.y = y
        return y
        
    def __backward__(self, gy):
        emb_dim, head_dim, n_head = self.config
        gz = self.DO.backward(gy)
        gz = self.linear_o.backward(gz)
        gz = np.split(gz, n_head, axis=-1)
        
        gvkq = [a.backward(gzi) for a, gzi in zip(self.attention, gz)]

        gv = np.concatenate([g[0] for g in gvkq], axis=-1)
        gk = np.concatenate([g[1] for g in gvkq], axis=-1)
        gq = np.concatenate([g[2] for g in gvkq], axis=-1)
       
        gv = self.linear_v.backward(gv)
        gk = self.linear_k.backward(gk)
        gq = self.linear_q.backward(gq)
        gx = gv + gk + gq
        return gx

    def update(self, eta=0.001, **kwargs):
        self.linear_v.update(eta=eta, **kwargs)
        self.linear_k.update(eta=eta, **kwargs)
        self.linear_q.update(eta=eta, **kwargs)
        self.linear_o.update(eta=eta, **kwargs)

    def entropy(self):
        return [sa.entropy for sa in self.attention]

    def entropy_std_and_range(self):
        e_head = np.array([sa.attention.entropy for sa in self.attention])
        #print(e_head.shape) 
        e_std = np.std(e_head)
        e_rng = np.max(e_head) - np.min(e_head)
        #print(f'[Entropy] std={e_std:.4f}, range={e_rng:.4f}')
        return e_std, e_rng
        
class ParametersForContextualSelfAttention:
    """
    時系列入力xに対してコンテキストベクトルyを返す
    即ち、入力の時系列に並ぶものの重要なことを抽出して文脈とする
    この操作では、keyとvalueは必要だがqueryは何でも良い．　　　
    なぜならば、出力y(コンテクスト)は入力xに応じて決まれば良いのだから．
    そこでqueryは入力xによらないパラメタとして用意する．
         
    """
    def __init__(self, layer, **kwargs):
        print(self.__class__.__name__)
        self.layer   = layer 
        optimize     = kwargs.pop('optimize',   'SGD') 
        self.width   = kwargs.pop('width',       None)
        self.q_shape = kwargs.pop('q_shape', (1,1,-1))
        self.w = None; self.b = None; self.q = None
        self.optimizer_w = cf.eval_in_module(optimize, Optimizers, **kwargs)
        self.optimizer_b = cf.eval_in_module(optimize, Optimizers, bias=True, **kwargs)
        self.optimizer_q = cf.eval_in_module(optimize, Optimizers, **kwargs)
        self.debug_mode = kwargs.pop('debug_mode', False)
       
    def __call__(self):
        if self.w is None:
            self.init_parameter()
        return self.w, self.b, self.q

    def init_parameter(self):#, m, n):
        m, n = self.layer.get_parameter_size() 
        if m is None or n is None:
            raise Exception('Configuration is not fixed.', self.__class__.__name__)
        self.w = init_weight((m, n),
                             width=self.width,
                             debug_mode=self.debug_mode)
        self.b = np.zeros(n, dtype=Config.dtype)
        self.q = init_weight((1,n),
                             width=np.sqrt(1/m), # 通常とは異なるため指定が必要
                             debug_mode=self.debug_mode)
        self.q = self.q.reshape(*self.q_shape)
                             
        #(width * np.random.randn(1,1,n)).astype(Config.dtype) 

    def update(self, eta=0.001, **kwargs):
        self.optimizer_w.update(self.w, self.grad_w, eta, **kwargs) 
        self.optimizer_b.update(self.b, self.grad_b, eta, **kwargs) 
        self.optimizer_q.update(self.q, self.grad_q, eta, **kwargs)

    def set_gradient(self, *grads): 
        self.grad_w = grads[0]
        self.grad_b = grads[1]
        self.grad_q = grads[2]
        
        
class ContextualSelfAttention(Function):
    """
    時系列入力xに対してコンテキストベクトルyを返す
    即ち、入力の時系列に並ぶものの重要なことを抽出して文脈とする
    この操作では、keyとvalueは必要だがqueryは何でも良い．　　　
    なぜならば、出力y(コンテクスト)は入力xに応じて決まれば良いのだから．
    そこでqueryは入力xによらないパラメタとして用意する．
         
    """
    def __init__(self, *configuration,
                 affine_v=False, affine_k=False, q_shape=(1,1,-1), **kwargs):
        super().__init__()
        if len(configuration) == 2:
            m, n = configuration
        if len(configuration) == 1:
            m = None; n, = configuration
        self.config = m, n

        # linear_iとlinear_oのconfigはfix_configurationで設定
        self.linear_v = LinearLayer(matmul=True, bias=False,**kwargs) \
            if affine_v else None
        self.linear_k = LinearLayer(matmul=True, bias=False,**kwargs) \
            if affine_k else None
        # qもfix_configurationで設定
        self.q, self.grad_q = None, None
        self.q_shape = q_shape
        optimize = kwargs.pop('optimize', 'SGD')
        self.optimizer_q = cf.eval_in_module(optimize, Optimizers, **kwargs)
        # Attentionとdebug_mode
        self.attention = AttentionUnit(scale=True)
        self.debug_mode = kwargs.pop('debug_mode',  False)
       
    def fix_configuration(self, shape):
        self.config = shape[-1], self.config[1]
        print('config =', self.config)
        m, n = self.config

        if self.linear_v is not None:
            self.linear_v.config = m, n 
        if self.linear_k is not None:
            self.linear_k.config = m, n 

        self.q = init_weight((1, n),
                             width=np.sqrt(1/m), # 通常とは異なるため指定が必要
                             debug_mode=self.debug_mode)
        self.q = self.q.reshape(*self.q_shape)

    def update(self, eta=0.001, **kwargs):
        if self.linear_v is not None:
            self.linear_v.update(eta=eta, **kwargs)
        if self.linear_k is not None:
            self.linear_k.update(eta=eta, **kwargs)
        self.optimizer_q.update(self.q, self.grad_q, eta, **kwargs)

    def get_parameter_size(self):
        m, n = self.config
        return m, n

    def __forward__(self, x):
        if None in self.config or self.q is None:
            print(self.__class__.__name__, '.input.shape', x.shape)
            self.fix_configuration(x.shape)
        m, n = self.config
        B, _, _ = x.shape
        
        v = self.linear_v.forward(x) if self.linear_v is not None else x
        k = self.linear_k.forward(x) if self.linear_k is not None else x
        q = np.broadcast_to(self.q, (B, 1, n))
       
        y = self.attention.forward(v, k, q)
        y = y.reshape(B, n)
        return y

    def __backward__(self, gy):
        x, = self.inputs
        m, n = self.config
        B, _, _ = x.shape

        gy = gy.reshape(B, 1, n)

        gv, gk, gq = self.attention.backward(gy)

        self.grad_q = np.sum(gq, axis=0, keepdims=True) # (B,1,H)->(1,1,H) forwardでのBCに対応

        gxk = self.linear_k.backward(gk) if self.linear_k is not None else gk
        gxv = self.linear_v.backward(gv) if self.linear_v is not None else gv
        gx = gxv + gxk

        return gx
    
class ContextualSelfAttention_bkup(Function):
    """
    時系列入力xに対してコンテキストベクトルyを返す
    即ち、入力の時系列に並ぶものの重要なことを抽出して文脈とする
    この操作では、keyとvalueは必要だがqueryは何でも良い．　　　
    なぜならば、出力y(コンテクスト)は入力xに応じて決まれば良いのだから．
    そこでqueryは入力xによらないパラメタとして用意する．
         
    """
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        if len(configuration) == 2:
            m, n = configuration
        if len(configuration) == 1:
            m = None; n, = configuration
        self.config = m, n
        self.attention = AttentionUnit(scale=True)
        self.dot_linear = F.MatMulLinear()
        self.parameters = ParametersForContextualSelfAttention(self, **kwargs)
       
    def fix_configuration(self, shape):
        self.config = shape[-1], self.config[1]
        print('config =', self.config)

    def update(self, eta=0.001, **kwargs):
        self.parameters.update(eta=eta, **kwargs)

    def get_parameter_size(self):
        m, n = self.config
        return m, n

    def __forward__(self, x):
        if None in self.config:
            print(self.__class__.__name__, '.input.shape', x.shape)
            self.fix_configuration(x.shape)

        w, b, q = self.parameters()    
        m, n = self.config # m, n = H ベクトルサイズ
        B, T, _ = x.shape
        self.x = x
        # 入力xからkeyを生成 keyの形状 (B, T, H)
        self.k = self.dot_linear.forward(x, w, b) # (B,T,H)
        y = self.attention.forward(x, self.k, np.broadcast_to(q, (B, 1, n)))
        y = y.reshape(B, n)
        return y

    def __backward__(self, gy):
        x = self.x
        m, n = self.config
        B, _, _ = x.shape

        gy = gy.reshape(B, 1, n)
        gx, gk, gq = self.attention.backward(gy) 
        grad_q = np.sum(gq, axis=0, keepdims=True) # (B,1,H)->(1,1,H) forwardでのBCに対応

        # keyの逆伝播
        gx2, grad_w, grad_b = self.dot_linear.backward(gk)
        gx += gx2

        self.parameters.set_gradient(grad_w, grad_b, grad_q)
        return gx
    

class ContextualSelfAttentionZ1(ContextualSelfAttention):
    """
    ContextualSelfAttentionと同一の処理
    順伝播、逆伝播を展開している
         
    """
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.softmax = Activators.Softmax()

    def __forward__(self, x):
        if None in self.config:
            print(self.__class__.__name__, '.input.shape', x.shape)
            self.fix_configuration(x.shape)

        w, b, q = self.parameters()     
        m, n = self.config # m, n = H ベクトルサイズ
        B, T, _ = x.shape
        self.x = x
        # 入力xからkeyを生成 keyの形状 (B, T, H)
        #print('### x =', x.shape, 'w =', self.w.shape)   
        self.k = np.matmul(x, w) + b      # (B,T,H)
        #print('### k =', self.k.shape, 'q =', self.q.shape)   

        # s:scoreとa:attention_weightを算出 sとaの形状:(B,T)
        s = np.matmul(self.q, self.k.transpose(0,2,1)) # (1,1,H)*(B,T,H)->(B,1,T)
        #print('### s =', s.shape)
        s *= m ** -0.5                              # スケーリング
        self.a = self.softmax.forward(s)
        #print('### a =', self.a.shape)
        # 入力(B, T, H)をバッチ軸(B)はそのままにして、時系列軸(T)方向に重み付け
        c = np.matmul(self.a, x) # (B,1,T)*(B,T,H)->(B,1,H)
        # コンテキストベクトル y.shape:(B, H) 
        y = c.reshape(B, n)                         # (B,1,H) -> (B,H)
        #print('### y =', y.shape)
        #input()
        return y

    def __backward__(self, gy):
        x = self.x
        a = self.a
        m, n = self.config
        B, T, _ = x.shape
        
        w, b, q = self.parameters()     
        # 重み付け和の逆伝播
        gc = gy.reshape(B, 1, n) # (B,H)->(B,1,H)
 
        ga = np.matmul(gc, x.transpose(0,2,1))
        gx = np.matmul(a.transpose(0,2,1), gc)
        #print('### ga =', ga.shape, 'gx =', gx.shape)
        # attention_weightの逆伝播
        gs = self.softmax.backward(ga)
        gs *= m ** -0.5

        #print('### gs =', gs.shape, 'q =', self.q.shape)
        gk = np.matmul(gs.transpose(0,2,1), q)     # (B,1,T)*(1,1,H)->(B,T,H)
        #print('### gk =', gk.shape)
        #print('### k =', self.k.shape, 'gs =', gs.shape) 
        gq = np.matmul(gs, self.k)                      # (B,1,T)*(B,T,H)->(B,1,H)
        #print('### gq =', gq.shape)
        grad_q = np.sum(gq, axis=0, keepdims=True) # (B,1,H) -> (1,1,H)
        #print('### grad_q =', self.grad_q.shape)
        # keyの逆伝播
        #self.grad_w = np.matmul(x.reshape(-1, m).T, gk.reshape(-1, n))
        grad_w = np.matmul(x.transpose(0,2,1), gk)  # (B,H,T) * (B,T,H) -> (B,H,H)
        grad_w = np.sum(grad_w, axis=0)        # (B,H,H) -> (H,H)
        grad_b = np.sum(gk, axis=(0,1))        # (B,T,H) -> (H,)
        #print('### grad_w', self.grad_w.shape)
        gx += np.matmul(gk, self.w.T)
        #input()
        self.parameters.set_gradient(grad_w, grad_b, grad_q)
        return gx
  
class ContextualSelfAttentionZ2(ContextualSelfAttention):
    """
    ContextualSelfAttentionと同一の処理
    クエリ―の形状が違う
         
    """
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.softmax = Activators.Softmax()

    def init_parameter(self):#, m, n):
        m, n = self.get_parameter_size() 
        if m is None or n is None:
            raise Exception('Configuration is not fixed.', self.__class__.__name__)
        if self.width is not None:
            width = self.width
        else:    
            width = np.sqrt(1/m)  # Xavierの初期値

        self.w = (width * np.random.randn(m, n)).astype(Config.dtype) 
        self.b = np.zeros(n, dtype=Config.dtype)
        self.q = (width * np.random.randn(n)).astype(Config.dtype) 
        if self.debug_mode:
            self.w[...] = 1; self.q[...] = 1 # for debug
        print(self.__class__.__name__, 'init_parameters', m, n)

    def __forward__(self, x):
        if None in self.config:
            print(self.__class__.__name__, '.input.shape', x.shape)
            self.fix_configuration(x.shape)
        if self.w is None or self.b is None or self.q is None:
            self.init_parameter()
        m, n = self.config # m, n = H ベクトルサイズ
        B, T, _ = x.shape
        self.x = x
        # 入力xからkeyを生成 keyの形状 (B, T, H)
        #print('### x =', x.shape, 'w =', self.w.shape)   
        self.k = np.matmul(x, self.w) + self.b      # (B,T,H)
        #print('### k =', self.k.shape, 'q =', self.q.shape)   
        # s:scoreとa:attention_weightを算出 sとaの形状:(B,T)
        q = self.q.reshape(1,n,1)                   # (H,) -> (1,H,1) 
        s = np.matmul(self.k, q)                    # (B,T,H) * (1,H,1) -> (B,T,1)
        s = s.reshape(B, T)
        #print('### s =', s.shape)
        s *= n ** -0.5                              # スケーリング
        self.a = self.softmax.forward(s)
        #print('### a =', self.a.shape)
        # 入力(B, T, H)をバッチ軸(B)はそのままにして、時系列軸(T)方向に重み付け
        a = self.a.reshape(B, T, 1) #self.a[..., np.newaxis] (B,T) -> (B,T,1)
        c = x * a                                   # (B,T,H)*(B,T,1) -> (B,T,H)
        # コンテキストベクトル y.shape:(B, H) 
        y = np.sum(c, axis=1)                       # (B,T,H) -> (B,H)
        #print('### y =', y.shape)
        return y

    def __backward__(self, gy):
        x = self.x
        a = self.a
        m, n = self.config
        B, T, _ = x.shape
        
        # 重み付け和の逆伝播
        gc = gy.reshape(B, 1, m).repeat(T, axis=1)
        ga = np.sum(gc * x, axis=-1)#, keepdims=True)
        gx = gc * self.a[..., np.newaxis]
        #print('### ga =', ga.shape, 'gx =', gx.shape)
        # attention_weightの逆伝播
        gs = self.softmax.backward(ga)
        gs *= n ** -0.5
        gs = gs[..., np.newaxis]
        #print('### gs =', gs.shape, 'q =', self.q.shape)
        qT = self.q.reshape(1,1,n)
        gk = np.matmul(gs, qT)     # (B,T,1) * (1,1,H) -> (B,T,H)
        #print('### gk =', gk.shape)
        #print('### k =', self.k.shape, 'gs =', gs.shape) 
        gq = np.matmul(self.k.transpose(0,2,1), gs) # (B,T,H) * (B,T,1) -> (B,H,1)
        #print('### gq =', gq.shape)
        gq = np.sum(gq, axis=0)                     # (B,H,1) -> (H,1)
        self.grad_q = gq.reshape(n) 
        #print('### grad_q =', self.grad_q.shape)
        # keyの逆伝播
        grad_w = np.matmul(x.transpose(0,2,1), gk)  # (B,H,T) * (B,T,H) -> (B,H,H)
        self.grad_w = np.sum(grad_w, axis=0)        # (B,H,H) -> (H,H)
        self.grad_b = np.sum(gk, axis=(0,1))        # (B,T,H) -> (H,)
        #print('### grad_w', self.grad_w.shape)
        gx += np.matmul(gk, self.w.T)
        #input()
        return gx


class ContextualSelfAttentionZ4(ContextualSelfAttention):
    """
    ContextualSelfAttentionと同一の処理
    クエリ―の形状が違うが、
    attention weight生成の際にsoftmaxの軸が末尾となっており、
    本来の機能とならない問題あり     
         
    """
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.softmax = Activators.Softmax()

    def init_parameter(self):#, m, n):
        m, n = self.get_parameter_size() 
        if m is None or n is None:
            raise Exception('Configuration is not fixed.', self.__class__.__name__)
        if self.width is not None:
            width = self.width
        else:    
            width = np.sqrt(1/m)  # Xavierの初期値

        self.w = (width * np.random.randn(m, n)).astype(Config.dtype) 
        self.b = np.zeros(n, dtype=Config.dtype)
        self.q = (width * np.random.randn(n, 1)).astype(Config.dtype) 
        if self.debug_mode:
            self.w[...] = 1; self.q[...] = 1 # for debug
        print(self.__class__.__name__, 'init_parameters', m, n)

    def __forward__(self, x):
        if None in self.config:
            print(self.__class__.__name__, '.input.shape', x.shape)
            self.fix_configuration(x.shape)
        if self.w is None or self.b is None or self.q is None:
            self.init_parameter()
        m, n = self.config # m, n = H ベクトルサイズ
        B, T, _ = x.shape
        self.x = x
        # 入力xからkeyを生成 keyの形状 (B, T, H)
        #print('### x =', x.shape, 'w =', self.w.shape)   
        self.k = np.matmul(x, self.w) + self.b      # (B,T,H)
        #print('### k =', self.k.shape, 'q =', self.q.shape)   
        # s:scoreとa:attention_weightを算出 sとaの形状:(B,T)
        s = np.matmul(self.k, self.q)               # (B,T,H) * (H,1) -> (B,T,1)
        #print('### s =', s.shape)
        #s *= T ** -0.5                              # スケーリング
        s *= m ** -0.5                              # スケーリング
        self.a = self.softmax.forward(s)
        #print('### a =', self.a.shape)
        # 入力(B, T, H)をバッチ軸(B)はそのままにして、時系列軸(T)方向に重み付け
        c = x * self.a #[..., np.newaxis]
        # コンテキストベクトル y.shape:(B, H) 
        y = np.sum(c, axis=1)                       # (B,T,H) -> (B,H)
        #print('### y =', y.shape)
        return y

    def __backward__(self, gy):
        x = self.x
        a = self.a
        m, n = self.config
        B, T, _ = x.shape
        
        # 重み付け和の逆伝播
        gc = gy.reshape(B, 1, m).repeat(T, axis=1)
        ga = np.sum(gc * x, axis=-1, keepdims=True)
        gx = gc * self.a #[..., np.newaxis]
        #print('### ga =', ga.shape, 'gx =', gx.shape)
        # attention_weightの逆伝播
        gs = self.softmax.backward(ga)
        #gs *= T ** -0.5
        gs *= m ** -0.5
        #print('### gs =', gs.shape, 'q =', self.q.shape)
        gk = np.matmul(gs, self.q.T)                # (B,T) * (1,H) -> (B,T,H)
        #print('### gk =', gk.shape)
        #print('### k =', self.k.shape, 'gs =', gs.shape) 
        gq = np.matmul(self.k.transpose(0,2,1), gs) # (B,H,T) * (B,T,1) -> (B,H,1)
        #print('### gq =', gq.shape)
        self.grad_q = np.sum(gq, axis=0)            # (B,H,1) -> (H,1)
        #print('### grad_q =', self.grad_q.shape)
        # keyの逆伝播
        #self.grad_w = np.matmul(x.reshape(-1, m).T, gk.reshape(-1, n))
        grad_w = np.matmul(x.transpose(0,2,1), gk)  # (B,H,T) * (B,T,H) -> (B,H,H)
        self.grad_w = np.sum(grad_w, axis=0)        # (B,H,H) -> (H,H)
        self.grad_b = np.sum(gk, axis=(0,1))        # (B,T,H) -> (H,)
        #print('### grad_w', self.grad_w.shape)
        gx += np.matmul(gk, self.w.T)
        #input()
        return gx



#### ドロップアウト ###############################################　
class Dropout(Function): # inverted_dropout 
    def __init__(self, inplace=False):
        super().__init__()
        self.dropout_mx = None          # はじめて伝播する際に必要
        self.inplace = inplace          # inplace演算とするかどうか
        
    def __forward__(self, x, *, dropout=0.0): # x→y,ドロップアウト率(非学習時は0)
        y = x if self.inplace else x.copy() # inplaceではyはxと同一
        scale = 1 / (1 - dropout + 1e-7)
        if dropout > 0.0: # ドロップアウトする場合に残る割合で拡大
            rand = np.random.rand(*x.shape) # yと同じ形状の乱数の行列 
            # 予めスケールを合わせておく
            self.dropout_mx = np.where(rand > dropout, scale, 0).astype(Config.dtype)
            y *= self.dropout_mx  # ニューロンをランダムに無効化(0固定
        else:
            self.dropout_mx = 1         # ドロップアウトしたりしなかったりに対応   
        return y                        # inplaceの場合にはx更新で返り値不要

    def __backward__(self, gy):     # 順伝播時に無効化したニューロンは逆伝播しない
        gx = gy if self.inplace else gy.copy() # inplaceではgxはgyと同一
        gx *= self.dropout_mx           # 順伝播時の情報を使う
        return gx                       # inplaceの場合にはgy更新で返り値不要 
    
   
#### ドロップアウト ###############################################　
class Dropout2(Function): # direct_dropout
    def __init__(self, inplace=False):
        super().__init__()
        self.dropout_mx = None          # はじめて伝播する際に必要
        self.dropout_ratio = None
        self.inplace = inplace          # inplace演算とするかどうか
        
    def __forward__(self, x, *, dropout=0.0): # x→y,ドロップアウト率(非学習時は0)
        y = x if self.inplace else x.copy() # inplaceではyはxと同一
        if dropout > 0.0: 
            self.dropout_ratio = dropout   # ドロップアウト時に覚える
            self.dropout_mx = np.random.rand(*x.shape) > dropout # 0/1の行列
            self.dropout_mx = self.dropout_mx.astype(Config.dtype)
            y *= self.dropout_mx  # ニューロンをランダムに無効化(0固定
        else:           # ドロップアウトしない場合にスケールを合わせる
            dropout = 0 if self.dropout_ratio is None else self.dropout_ratio
            self.dropout_mx = 1.0       # ドロップアウトしたりしなかったりに対応　
            y *= (1 - dropout)
        return y                        # inplaceの場合にはx更新で返り値不要    
            
    def __backward__(self, gy):         # 順伝播時に無効化したニューロンは逆伝播しない
        gx = gy if self.inplace else gy.copy() # inplaceではgxはgyと同一
        gx *= self.dropout_mx           # 順伝播時の情報を使う
        return gx
#### ドロップアウト ###############################################　
class Dropout_bkup(Function): # inverted_dropout 
    def __init__(self):
        super().__init__()
        self.dropout_mx = None          # はじめて伝播する際に必要
        
    def __forward__(self, x, *, dropout=0.0): # x→y,ドロップアウト率(非学習時は0)
        scale = 1 / (1 - dropout)
        if dropout > 0.0: # ドロップアウトする場合に残る割合で拡大
            rand = np.random.rand(*x.shape) # yと同じ形状の乱数の行列 
            # 予めスケールを合わせておく
            self.dropout_mx = np.where(rand > dropout, scale, 0).astype(Config.dtype)
            return x * self.dropout_mx  # ニューロンをランダムに無効化(0固定
        
        else:
            self.dropout_mx = 1         # ドロップアウトしたりしなかったりに対応   
            return x

    def __backward__(self, grad_y):     # 順伝播時に無効化したニューロンは逆伝播しない
        return grad_y * self.dropout_mx # 順伝播時の情報を使う
    
   
#### ドロップアウト ###############################################　
class Dropout2_bkup(Function): # direct_dropout
    def __init__(self):
        super().__init__()
        self.dropout_mx = None          # はじめて伝播する際に必要
        self.dropout_ratio = None
        
    def __forward__(self, x, *, dropout=0.0): # x→y,ドロップアウト率(非学習時は0)
        if dropout > 0.0: 
            self.dropout_ratio = dropout   # ドロップアウト時に覚える
            self.dropout_mx = np.random.rand(*x.shape) > dropout # 0/1の行列
            self.dropout_mx = self.dropout_mx.astype(Config.dtype)
            return x * self.dropout_mx  # ニューロンをランダムに無効化(0固定
        else:           # ドロップアウトしない場合にスケールを合わせる
            dropout = 0 if self.dropout_ratio is None else self.dropout_ratio
            self.dropout_mx = 1.0       # ドロップアウトしたりしなかったりに対応　
            return (1 - dropout) * x
            
    def __backward__(self, grad_y):     # 順伝播時に無効化したニューロンは逆伝播しない
        return grad_y * self.dropout_mx # 順伝播時の情報を使う
   
  
        
#### キャプチャ #####################################################　
#  RNN などで出力の一部、例えば、最後の時刻のみを使うような場合に対応する
#  forwardとbackwardで時系列長が異なる場合、たとえば最後の出力のみを使うような場合には
#  backward の際 grad_y に foward で後続層で使用された時刻の出力範囲のみ与えられるので
#  その部分に grad_y をはめ込んで、grad_y の他の部分は 0 として受け渡す
#  注：backward では、時系列の全域にわたり、勾配を逆伝播する必要があるから、
#      forward で出力を使用しなかった範囲については backward で順次算出される
#      リカレントなパスからの勾配のみを伝播する
class Capture(Function): 
    def __forward__(self, x, *, width=None):
        self.config = None, None, width
        if width is None:
            return x
        #super().__init__()
        #print('Capture', x.shape, 'width', width, end='|')
        if   x.ndim==3:
            B, Tf, m = x.shape
        elif x.ndim==2:
            B, m     = x.shape
            Tf = 1
            x = x.reshape(B, 1, m)
        else:
            print('順伝播で入力の次元数が１以下ないしは４以上のため対応できません')
        self.config = Tf, m, width
        if width is not None:
            y = x[:, -width:, :]
        if width==1:
            y = y.reshape(B, m)
        return y

    def __backward__(self, grad_y):
        Tf, m, width = self.config
        x, = self.inputs
        if width is None:
            return grad_y
        if   grad_y.ndim==3:
            B, Tb, n = grad_y.shape
        elif grad_y.ndim==2:   
            B, n     = grad_y.shape
            Tb = 1
            grad_y = grad_y.reshape(B, 1, n) 
        else:
            print('逆伝播で入力の次元数が１以下ないしは４以上のため対応できません')
        grad_x = np.zeros((B, Tf, n), dtype='f4')
        grad_x[:, -width:, :] = grad_y
        if x.ndim==2:
            grad_x = grad_x.reshape(B, n)
        return grad_x
    
def set_axis_and_shape(shape, axis, exclude=False):
    """ shapeとaxisからexcludeに従い、新たな形状とそこから外れる軸を得る """
    # shapeから全軸を抽出し、axisの指定を正規化
    ndim = len(shape)
    all_axis = list(range(ndim))        # すべての軸
    if axis is None:                    # Noneに対応
        axis = all_axis
    if type(axis) not in (tuple, list): # 整数に対応
        axis = axis,
    axis = tuple(ndim + ax if ax<0 else ax for ax in axis) # 負値に対応

    # axisに含まれる軸と含まれない軸
    include_axis = tuple(ax for ax in all_axis if ax in axis)
    exclude_axis = tuple(ax for ax in all_axis if ax not in axis)
    # axisの指定によって抽出される形状と排除される形状
    include_shape = tuple(shape[ax] if ax in axis else 1 for ax in all_axis)
    exclude_shape = tuple(1 if ax in axis else shape[ax] for ax in all_axis) 

    #print('axisに含まれる軸と含まれない軸', include_axis, exclude_axis)
    #print('axisの指定によって抽出される形状と排除される形状', include_shape, exclude_shape)

    # axisとexcludeの指定による新たな形状とそこから外れる軸  
    if exclude:
        return exclude_shape, include_axis
    return include_shape, exclude_axis
    

#### 正規化のクラス #######################################################
class Normalization(Function):
    """ 平均0標準偏差1にする標準化(正規化の一種) """
    def __init__(self, axis=None, ppl=False, eps=1e-12,
                 mask_enable=False, inplace=False, **kwargs):
        super().__init__()
        #print('Initialize', self.__class__.__name__, axis, ppl)#, kwargs)
        self.axis = axis
        self.ppl = ppl
        self.eps = eps
        self.mu = None
        self.sigma = None
        self.mask = None
        self.mask_enable = mask_enable # mask small variant data
        self.inplace = inplace         # インプレース処理
        #optimize = kwargs.pop('optimize', 'SGD')
        if ppl: # 非訓練時に移動平均を使用(バッチノーマライゼーション)
            self.mu_ppl = None
            self.sigma_ppl = None
            self.OFm = cf.eval_in_module('SGD', Optimizers) # 最適化関数は固定 
            self.OFs = cf.eval_in_module('SGD', Optimizers) # 最適化関数は固定
        if mask_enable and inplace:
            msg = self.__class__.__name__ + ':Both mask_enable and inplace will cause wrong result.'
            warnings.warn(msg)
   
    def init_parameters(self, shape):       
        """ 軸の指定に従いmuとsigmaの形状を決めて初期化 """
        mu_sigma_shape, _ = set_axis_and_shape(shape, self.axis, True)
        #print('muとsigmaの形状', mu_sigma_shape)
        if self.ppl:
            self.mu_ppl = np.zeros(mu_sigma_shape, dtype=Config.dtype) # 全体平均(移動平均 moving average)
            self.sigma_ppl = np.ones(mu_sigma_shape, dtype=Config.dtype)  # 全体分散

    def update(self, *args, **kwargs):
        if self.ppl:
            self.OFm.update(self.mu_ppl,    self.mu_ppl    - self.mu,    eta=0.1)
            self.OFs.update(self.sigma_ppl, self.sigma_ppl - self.sigma, eta=0.1)

    def __forward__(self, x, *, train=False):
        if self.ppl and self.mu_ppl is None:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.init_parameters(x.shape)
        #y = x if self.inplace else x.copy() # inplaceではyはxと同一
        mu = np.mean(x, axis=self.axis, keepdims=True)
        sigma = np.std(x, axis=self.axis, keepdims=True)
        self.sigma = sigma
        self.mu = mu
        if self.ppl and not train:
            mu = self.mu_ppl 
            sigma = self.sigma_ppl
        if self.inplace:
            y = x
            y -= mu
        else:
            y = x - mu          
        y /= (sigma + self.eps)
        if not self.mask_enable:
            return y
        self.mask = sigma < self.eps # sigmaが極小値の場合には正規化しない
        if (self.mask==False).all():
            return y
        y *= (1 - self.mask)
        y += x * self.mask           # inplaceでは誤動作 
        return y 
   
    def __backward__(self, gy):
        gx = gy if self.inplace else gy.copy() # inplaceではyはxと同一
        x, = self.inputs
        y = self.get_outputs()       # inplaceではxと同一
        sigma = self.sigma + self.eps
        n = x.size//sigma.size       # 畳んだ大きさ
        gsigma = np.sum(-gy * y, axis=self.axis, keepdims=True) / sigma # gyが書き変わる前に
        gx /= sigma
        #print('\n# backward_1 gx\n', gx)
        gmu = - np.sum(gx, axis=self.axis, keepdims=True) / n
        gx += gmu
        #print('# backward_2 gx\n', gx, '\ny\n', y, '\ngsigma\n', gsigma)
        gx += gsigma * y / n
        #print('# backward_3 gx\n', gx)
        if not self.mask_enable:
            return gx
        if (self.mask==False).all():
            return gx
        gx *= (1 - self.mask)
        gx += gy * self.mask
        return gx 

    def __forward__bkup(self, x, *, train=False):
        """ 参照用の処理 """
        if self.ppl and self.mu_ppl is None:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.init_parameters(x.shape)
        mu = np.mean(x, axis=self.axis, keepdims=True)
        sigma = np.std(x, axis=self.axis, keepdims=True)
        self.sigma = sigma
        self.mu = mu
        if self.ppl and not train:
            mu = self.mu_ppl 
            sigma = self.sigma_ppl
        z = x - mu
        y = z / (sigma + self.eps)
        self.mask = sigma < self.eps # sigmaが極小値の場合には正規化しない
        return y * (1 - self.mask) + x * self.mask
   
    def __backward__bkup(self, gy):
        """ 参照用の処理 """
        x = self.inputs[0]
        y = self.get_outputs()
        sigma = self.sigma + self.eps
        n = x.size//sigma.size   # 畳んだ大きさ
        # y = z / sigma の逆伝播 
        gz = gy / sigma
        gsigma = -gy * y / sigma   # -gy * z / sigma ** 2
        gsigma = np.sum(gsigma, axis=self.axis, keepdims=True)
        # z = x - mu の逆伝播
        gx0 = gz
        gmu = np.sum(-gz, axis=self.axis, keepdims=True)
        # mu = np.mean(x) の逆伝播
        gx1 = np.broadcast_to(gmu, x.shape) / n
        # sigma = np.std(x) の逆伝播
        gx2 = gsigma * y / n   # gvar * dvar_dx = (gsigma * 0.5 / sigma) * ((2/n) * (x - mu))
        gx = gx0 + gx1 + gx2
        return gx * (1 - self.mask) + gy * self.mask

#### スケーリングとバイアスを適用するクラス ###############################
class ScaleAndBias(Function):
    def __init__(self, axis=None, exclude=False, **kwargs):
        super().__init__()
        #print('Initialize', self.__class__.__name__, axis, exclude)#, kwargs)
        self.axis = axis
        self.remain_axis = None
        self.exclude = exclude   # 処理がaxisの指定に沿うのか否か
        optimize = kwargs.pop('optimize',     'AdaGrad') # 勾配降下法
        self.OFg = cf.eval_in_module(optimize, Optimizers) # 最適化関数
        self.OFb = cf.eval_in_module(optimize, Optimizers) # 最適化関数
        self.gamma = None
        self.beta  = None

    def init_parameters(self, shape):
        """ 軸の指定に従いparameterの形状を決めて初期化、指定外の軸も設定 """
        parameter_shape, remain_axis = set_axis_and_shape(shape, self.axis, self.exclude)
        #print('parameterの形状', parameter_shape)
        self.gamma = np.ones(parameter_shape, dtype=Config.dtype)  # 広がり                    
        self.beta  = np.zeros(parameter_shape, dtype=Config.dtype) # オフセット                    
        self.remain_axis = remain_axis # 逆伝播に必要

    def update(self, eta=0.001, **kwargs): # 他のパラメタの更新と呼応して更新
        self.OFg.update(self.gamma, self.ggamma, eta, **kwargs) 
        self.OFb.update(self.beta,  self.gbeta,  eta, **kwargs)  
               
    def __forward__(self, x):
        if self.gamma is None:
            self.init_parameters(x.shape)
        y = x * self.gamma    # xを温存しないと逆伝播出来ない
        y += self.beta
        return y

    def __backward__(self, gy):
        x, = self.inputs
        self.gbeta  = np.sum(gy, axis=self.remain_axis, keepdims=True)
        self.ggamma = np.sum(x * gy, axis=self.remain_axis, keepdims=True)
        gx = gy * self.gamma
        return gx 

    def __forward__bkup(self, x):
        if self.gamma is None:
            self.init_parameters(x.shape)
        return self.gamma * x + self.beta

#### スカラ値によるスケーリングを適用するクラス ###############################
class ScalarScale(Function):
    def __init__(self, **kwargs):
        super().__init__()
        optimize = kwargs.pop('optimize',     'SGD')       # 勾配降下法
        self.OFg = cf.eval_in_module(optimize, Optimizers) # 最適化関数
        self.gamma = None

    def init_parameters(self):
        self.gamma = np.array(1.0, dtype=Config.dtype)                     

    def update(self, eta=0.001, **kwargs): # 他のパラメタの更新と呼応して更新
        self.OFg.update(self.gamma, self.ggamma, eta, **kwargs) 
               
    def __forward__(self, x):
        if self.gamma is None:
            self.init_parameters()
        y = x * self.gamma    # xを温存しないと逆伝播出来ない
        return y

    def __backward__(self, gy):
        x, = self.inputs
        self.ggamma = np.sum(x * gy)
        gx = gy * self.gamma
        return gx 

#### 正規化の汎用ベース #### 
class GeneralNormalizationBase(Function):
    def __init__(self, axis=None, ppl=False, scale_and_bias=False, exclude=False,
                       inplace=False, **kwargs):
        super().__init__()
        #print('Initialize', self.__class__.__name__)
        self.ppl = ppl
        self.normalization = Normalization(axis=axis, ppl=ppl, inplace=inplace, **kwargs)
        self.sb = scale_and_bias     
        if self.sb:
            self.scale_and_bias = ScaleAndBias(axis=axis, exclude=exclude, **kwargs)

    def init_parameters(self, shape):
        if self.ppl:
            self.normalization.init_parameters(shape)
        if self.sb:
            self.scale_and_bias.init_parameters(shape)

    def update(self, eta=0.001, **kwargs):
        self.normalization.update(eta, **kwargs)
        if self.sb:
            self.scale_and_bias.update(eta, **kwargs)
               
    def __forward__(self, x, *, train=False):
        y = self.normalization.forward(x, train=train)
        if self.sb:
            y = self.scale_and_bias.forward(y)
        return y        

    def __backward__(self, gy):
        if self.sb:
            gx = self.scale_and_bias.backward(gy)
        else:
            gx = gy # gyをインプレース更新
        gx = self.normalization.backward(gx)   
        return gx 

class BatchNormalization(GeneralNormalizationBase):
    def __init__(self, scale_and_bias=True, inplace=False, **kwargs):
        # 正規化の軸はバッチ軸0だが、scale_and_biasの軸は特長軸すなわち0以外の軸
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = 0
        kwargs['exclude']        = True
        kwargs['ppl']            = True
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class batch_normalization(BatchNormalization):
    pass


class LayerNormalization(GeneralNormalizationBase):
    def __init__(self, axis=-1, scale_and_bias=True, inplace=False, **kwargs):
        # 正規化もscale_and_biasも同じく特長軸をaxisで指定
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = axis
        kwargs['exclude']        = False
        kwargs['ppl']            = False
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)


#### 層正規化 #####################################################
class LayerNormalization_bkup(Function):
    def __init__(self, axis=None, ppl=False, scale_and_bias=False, **kwargs):
        super().__init__()
        print('Initialize', self.__class__.__name__)
        self.ppl = ppl
        self.normalization = Normalization(axis, ppl, **kwargs)
        self.sb = scale_and_bias     
        if self.sb:
            self.scale_and_bias = ScaleAndBias(axis, **kwargs)

    def init_parameters(self, shape):
        if self.ppl:
            self.normalization.init_parameters(shape)
        if self.sb:
            self.scale_and_bias.init_parameters(shape)

    def update(self, eta=0.001, **kwargs):
        self.normalization.update(eta, **kwargs)
        if self.sb:
            self.scale_and_bias.update(eta, **kwargs)
               
    def __forward__(self, x, *, train=False):
        y = self.normalization.forward(x, train=train)
        if self.sb:
            y = self.scale_and_bias.forward(y)
        return y        

    def __backward__(self, gy):
        if self.sb:
            gx = self.scale_and_bias.backward(gy)
        else:
            gx = gy
        gx = self.normalization.backward(gy)   
        return gx 

#### バッチノーマライゼーションの関数 ###############################
class batch_normalization2(Function):
    def __init__(self, *n):
        super().__init__()
        print('Initialize', self.__class__.__name__)   
        self.OFg = cf.eval_in_module('AdaGrad', Optimizers) # 最適化関数
        self.OFb = cf.eval_in_module('AdaGrad', Optimizers) # 最適化関数
        self.OFm = cf.eval_in_module('SGD',     Optimizers) # 最適化関数 
        self.OFv = cf.eval_in_module('SGD',     Optimizers) # 最適化関数
        self.gamma   = None
        self.beta    = None
        self.mu_ppl  = None
        self.var_ppl = None
        self.cnt      = 0                 # 学習回数

    def init_parameters(self, n):         # nは入力の大きさ(バッチサイズではない)
        self.gamma   = np.ones(n, dtype='f4')      # 広がり                    
        self.beta    = np.zeros(n, dtype='f4')     # オフセット                    
        self.mu_ppl  = np.zeros(n, dtype='f4')     # 全体平均(移動平均 moving average)
        self.var_ppl = np.ones(n, dtype='f4')      # 全体分散

    def update(self):
        self.OFg.update(self.gamma, self.ggamma)
        self.OFb.update(self.beta,  self.gbeta)
        self.OFm.update(self.mu_ppl,  self.mu_ppl  - self.mu,  eta=0.1)#, g_clip=100)
        self.OFv.update(self.var_ppl, self.var_ppl - self.var, eta=0.1)#, g_clip=100)
               
    def __forward__(self, x, *, train=False):
        if self.mu_ppl is None:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.init_parameters(x.shape[1:])
        '''
        画像の周辺の背景部分などではバッチ内で同一値となるのは当然起きる
        この時バッチ内で分散は極小値(理論的には0)となる
        バッチ内分散が極小の場合は標準偏差も極小となって、分母が極小のため正規化はできない
        そこで分散を値1.0に固定する(標準偏差も1.0になる)
        '''
        self.x  = x
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0, ddof=0)
        #mask = var < 1e-12      # varが極小値でmask=1
        #var += mask             # varを１に固定
        #self.mask = mask
        self.var  = var
        self.mu   = mu
        mu  = mu  if train else self.mu_ppl 
        var = var if train else self.var_ppl
        xhat = (x - mu) / (np.sqrt(var) + 1e-12)            
        return self.gamma * xhat + self.beta

    def __backward__(self, gy):
        #mask  = self.mask
        istd = 1 / np.sqrt(self.var) 
        iN   = 1 / len(gy)      # バッチサイズ
        xc   = self.x - self.mu
        xhat = xc * istd
        self.gbeta  = np.sum(gy, axis=0)
        self.ggamma = np.sum(xhat * gy, axis=0)
        gxhat  = gy * self.gamma
        gy_sum = np.sum(gxhat * xc, axis=0, keepdims=True)
        gz   = (gxhat - (xhat * gy_sum * istd * iN)) * istd
        gz_sum = np.sum(gz, axis=0, keepdims=True)
        gx   = gz - (gz_sum * iN)
        return gx 
        #return gx * (1 - mask) + gy * mask # BN非対象の場合にはgyを直にgxに伝播

#### L2ノーマライゼーションの関数 ###############################
class L2Normalize(Function):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis
        self.config = None
    
    def __forward__(self, x):
        x = np.array(x)
        l2n = np.sum(x**2, axis=self.axis, keepdims=True)**0.5
        y = x / l2n
        self.x = x
        self.l2n = l2n
        self.y = y
        return y
   
    def __backward__(self, gy=1):
        x = self.x
        y = self.y
        l2n = self.l2n
        gx0 = gy / l2n
        gl2n = - gy * x / l2n ** 2
        gl2n = np.sum(gl2n, self.axis, keepdims=True)
        gsqsm = gl2n * 0.5 / (l2n + 1e-12)
        gx1 = 2 * x * gsqsm
        gx = gx0 + gx1
        return gx


### 平坦化 #####################################################
class Flatten(Function):
    def __init__(self):
        super().__init__()
        print("Functionsに定義されたものを使ってください")
        self.config = None

    def __forward__(self, x, *args, **kwargs):
        self.x_shape = x.shape
        return x.reshape(self.x_shape[0], -1)

    def __backward__(self, gy=1):
        if isinstance(gy, np.ndarray):
            return gy.reshape(*self.x_shape)
        else:
            return np.ones(self.x_shape, dtype=Config.dtype)
        
### 平均 #######################################################
class Mean(Function):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        print("Functionsに定義されたものを使ってください")
        self.axis = axis
        self.keepdims=keepdims
        
    def __forward__(self, x):
        self.x_shape = x.shape
        self.x_size  = x.size
        y = np.mean(x, axis=self.axis, keepdims=self.keepdims)
        self.y_shape = y.shape
        return y

    def __backward__(self, gy=1):
        gy = gy if isinstance(gy, np.ndarray) else np.array(gy, dtype=Config.dtype) 
        gy = np.broadcast_to(gy, self.y_shape)  # 先ずはyの形状に合わせる
        if self.axis is not None:               # 畳まれた軸は1、他は元の形状
            gy_shape = self.x_shape[:self.axis] + (1,) + self.x_shape[self.axis+1:]
            n = self.x_shape[self.axis]         # 畳まれる軸内の要素数
        else:
            gy_shape = self.y_shape
            n = self.x_size
        gx = (1/n) * gy.reshape(gy_shape)
        gx = np.broadcast_to(gx, self.x_shape)
        return gx

    
#########################################
if __name__=='__main__':
    print('\n#### all cast ####')
    import inspect
    import sys
    current_module = sys.modules[__name__]
    classes = map(lambda x:x[0],inspect.getmembers(current_module,inspect.isclass))
    for c in classes:
        print(c)
    
