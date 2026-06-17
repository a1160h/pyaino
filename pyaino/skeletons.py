# skeletons
# 20260617 A.Inoue

from pyaino.Config import *
#set_derivative(True)
#from pyaino import stems_blocks_heads as sbh
#from pyaino import Neuron as nn
#from pyaino import Activators
#from pyaino import LossFunctions as lf
#from pyaino import Functions as F
#from pyaino import common_function as cf
#import warnings
#import copy


class PredictionSkeleton:
    def __init__(self, core, time_mlp=None, label_mlp=None):
        self.core = core
        self.time_mlp = time_mlp
        self.label_mlp = label_mlp
        self.time_mlp_used = False; self.label_mlp_used = False

    def forward(self, x, timesteps=None, labels=None, train=True):
        self.time_mlp_used  = False
        self.label_mlp_used = False
        time_ctx  = None
        label_ctx = None
        # time embedding -> mlp
        if (self.time_mlp is not None) and (timesteps is not None):
            t0 = timesteps
            t0 = self.normalize_t(t0, x.shape[0])    # バッチサイズだけ合わせる
            time_ctx = self.time_mlp(t0, train=train)
            self.time_mlp_used = True  
        # label embedding -> mlp
        if (self.label_mlp is not None) and (labels is not None):
            label_ctx = self.label_mlp(labels, train=train)
            self.label_mlp_used = True 
        # 注入ベクトルの確定(使ったことを受けて設定)
        if self.time_mlp_used and self.label_mlp_used: 
            v = time_ctx + label_ctx
        elif self.time_mlp_used:
            v = time_ctx
        elif self.label_mlp_used:
            v = label_ctx
        else:
            v = None
            
        y = self.core(x, v=v, train=train)
        return y

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, eta=0.001, **kwargs):
        if self.time_mlp is not None and self.time_mlp_used:
            self.time_mlp.update(eta=eta, **kwargs)     
        if self.label_mlp is not None and self.label_mlp_used:
            self.label_mlp.update(eta=eta, **kwargs)     
        self.core.update(eta=eta, **kwargs)

    def normalize_t(self, t, B=1):
        """
        x: (B,C,H,W)
        t: int (scalar) or array-like (B,)
        returns: t_vec (B,) int32
        """

        # スカラ int の場合
        if isinstance(t, (int,)):
            return np.full((B,), t, dtype=np.int32)

        # numpy/cupy scalar の場合（np.int32(5) 等）
        t_arr = np.asarray(t)
        if getattr(t_arr, "ndim", 0) == 0:
            return np.full((B,), int(t_arr), dtype=np.int32)

        # ベクトルの場合
        if t_arr.shape != (B,):
            raise ValueError(f"t must be scalar or shape (B,), got {t_arr.shape}, B={B}")

        return t_arr.astype(np.int32, copy=False)


class VAESkeleton:
    def __init__(self, encoder, sampling, decoder, loss_function,
                 alpha=1.0, **kwargs):
        self.encoder  = encoder
        self.sampling = sampling
        self.decoder  = decoder
        self.loss_function = loss_function
        self.alpha = alpha # 画像と潜在変数のスケール合わせ

    def forward(self, x, **kwargs):
        e = self.encoder(x)
        s = self.sampling(e)
        if isinstance(s, (tuple, list)):
            z, kll, mi = s
        else:
            z = s; kll = 0; mi = 0
        y = self.decoder(z)
        l = self.loss_function(y, x)
        return y, l*self.alpha, kll, mi

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, eta=0.0, **kwargs):
        self.encoder.update(eta=eta, **kwargs)
        self.decoder.update(eta=eta, **kwargs)
