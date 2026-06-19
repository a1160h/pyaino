# skeletons
# 20260619 A.Inoue

from pyaino.Config import *
from pyaino import Neuron as nn

class PredictionSkeleton:
    def __init__(self, core, time_embed=False, num_labels=None, embed_dim=128,
                 **kwargs):

        self.core = core

        mlp_option = {'optimize' : kwargs.pop('optimize', 'SGD'),
                      'w_decay'  : kwargs.pop('w_decay',   0.0 ),}

        if time_embed:
            self.time_mlp = nn.Sequential(
                nn.PositionalEncoding(dimension=embed_dim),
                nn.NeuronLayer(embed_dim, activate=None, **mlp_option),
            )
        else:
            self.time_mlp = None

        if num_labels is not None:
            self.label_mlp = nn.Sequential(
                nn.Embedding(num_labels, embed_dim, **mlp_option),
                nn.NeuronLayer(embed_dim, activate=None, **mlp_option),
            )
        else:
            self.label_mlp = None

        self.time_mlp_used = False; self.label_mlp_used = False

    def forward(self, x, timesteps=None, labels=None, **kwargs):
        self.time_mlp_used  = False
        self.label_mlp_used = False
        time_ctx  = None
        label_ctx = None
        # time embedding -> mlp
        if (self.time_mlp is not None) and (timesteps is not None):
            t0 = timesteps
            t0 = self.normalize_t(t0, x.shape[0])    # バッチサイズだけ合わせる
            time_ctx = self.time_mlp(t0, **kwargs)
            self.time_mlp_used = True  
        # label embedding -> mlp
        if (self.label_mlp is not None) and (labels is not None):
            label_ctx = self.label_mlp(labels, **kwargs)
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
            
        y = self.core(x, v=v, **kwargs)
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
    def __init__(self, encoder, decoder, loss_function,
                 alpha=1.0, **kwargs):
        self.encoder  = encoder
        
        self.sampling = nn.LatentSampling(
            rate = kwargs.pop('rate', 1.0), # 正規分布の乱数のスケール係数
            kld  = kwargs.pop('kld',  1.0), # KullbackLeiblerDivergenceのスケール係数
            mil  = kwargs.pop('mil',  0.0), # MutualInformationLossのスケール係数
            )

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
