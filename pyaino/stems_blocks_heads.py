# stems_blocks_heads
# 20260615 A.Inoue

from pyaino.Config import *
from pyaino import nucleus
from pyaino import Neuron as nn
from pyaino import Functions as F
from pyaino import Activators as A
from pyaino import LossFunctions as lf
from pyaino import common_function as cf

class TransformerBlock(nucleus.Function):
    """ Transformer block: communication followed by computation """

    def __init__(self, emb_dim=64, n_head=4, causality=None, proj=False, 
                 rms=False, activate='Mish', **kwargs):
        super().__init__()
        self.sa = nn.MultiHeadSelfAttention(
            emb_dim, emb_dim//n_head, n_head, causality=causality, **kwargs) # entropy制御はkwargsで指定
        self.ffwd = nn.Sequential(
            nn.NeuronLayer(emb_dim, emb_dim*n_head, matmul=True, activate=activate, **kwargs),
            nn.NeuronLayer(emb_dim*n_head, emb_dim, matmul=True, dropout=True, **kwargs),
            )
        Norm = nn.RMSNormalization if rms else nn.LayerNormalization
        self.ln1 = Norm(**kwargs)
        self.ln2 = Norm(**kwargs)

        # ベクトル加算用
        if proj:
            self.proj = nn.LinearLayer(emb_dim, **kwargs) # 出力幅未定
        else:
            self.proj = None
        self.proj_used = False
            
    def __forward__(self, x, v=None, mask=None, dropout=0.0):
        self.proj_used = False

        z = self.ln1.forward(x)
        z = self.sa.forward(z, mask=mask, dropout=dropout)

        if (self.proj is not None) and (v is not None): # timeやlabelのmlpからの注入口
            u = self.proj(v)                            # projで形状を合わせて
            z = x + z + u[:,None,:]                     # 加算注入
            self.proj_used = True
        else:
            z = x + z
         
        y = self.ln2.forward(z)
        y = self.ffwd.forward(y, dropout=dropout)
        y = z + y
        self.y = y
        return y

    def __backward__(self, gy):
        gz = self.ffwd.backward(gy)     
        gz = self.ln2.backward(gz)      
        gz += gy                        

        if self.proj_used:
            gu = np.sum(gz, axis=1)     # u[:,None,:]として足し込んでいるのでsumで戻す
            gv = self.proj.backward(gu)
        else:
            gv = None

        gx = self.sa.backward(gz)
        gx = self.ln1.backward(gx)
        gx += gz

        return gx, gv

    def update(self, **kwargs):
        self.sa.update(**kwargs)

        if self.proj is not None and self.proj_used:
            self.proj.update(**kwargs)

        self.ffwd.update(**kwargs)
        self.ln1.update(**kwargs)
        self.ln2.update(**kwargs)


# VAEの構成
class MyVAE:
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
        #print('###', e.shape, z.shape)    
        y = self.decoder(z)
        
        l = self.loss_function(y, x)
        return y, l*self.alpha, kll, mi

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, eta=0.0, **kwargs):
        self.encoder.update(eta=eta, **kwargs)
        self.decoder.update(eta=eta, **kwargs)

    def analize(self, U, Z):
        mu, log_var = np.split(U, 2, axis=1)
        sigma = np.exp(0.5 * log_var)
        print(f"           mean    std     min     max")
        print(f"mu     : {mu.mean():7.4f} {mu.std():7.4f} {mu.min():7.4f} {mu.max():7.4f}")
        print(f"logvar : {log_var.mean():7.4f} {log_var.std():7.4f} {log_var.min():7.4f} {log_var.max():7.4f}")
        print(f"sigma  : {sigma.mean():7.4f} {sigma.std():7.4f} {sigma.min():7.4f} {sigma.max():7.4f}")
        print(f"z      : {Z.mean():7.4f} {Z.std():7.4f} {Z.min():7.4f} {Z.max():7.4f}")

class ImageHead:
    def __init__(self, C=3, alpha=0.1, beta=0.1,
                 activate='Sigmoid', hl_act='Mish', **kwargs):
        self.C = C 
        self.alpha = alpha
        self.beta = beta
        # cnn_refine, mlp_refineの各層は遅延初期化
        self.cnn_refine = nn.Sequential(
            nn.Conv2dLayer(None, 3, 1, activate=hl_act, **kwargs),
            nn.Conv2dLayer(None, 3, 1, activate=hl_act, **kwargs),
        )
        self.conv = nn.Conv2dLayer(C, 1, 1, 0, activate=None, **kwargs)
        self.mlp_refine = nn.Sequential(
            nn.NeuronLayer(None, full_connection=True, activate=hl_act, **kwargs),
            nn.NeuronLayer(None, activate=None, **kwargs),
            F.Reshape(),
        )
        self.act = cf.eval_in_module(activate, A)
        self.init_flag = False

    def fix_configuration(self, shape):
        """ 遅延初期化のために無理やり.configを上書きする """
        # cf.load_parameters() が forward 前にも下層へ到達できるよう、
        # 層オブジェクト自体は __init__ で生成し、
        # shape だけを初回 forward で補完する。
        B, M, Ih, Iw = shape
        C = self.C
        # Conv2dLayer.config = C,Ih,Iw,M,Fh,Fw,Sh,Sw,pad,Oh,Ow
        work_config = list(self.cnn_refine.layers[0].config)
        work_config[3] = M
        self.cnn_refine.layers[0].config = tuple(work_config)
        work_config = list(self.cnn_refine.layers[1].config)
        work_config[3] = M
        self.cnn_refine.layers[1].config = tuple(work_config)
        # NeuronLayer.config = m, n
        self.mlp_refine.layers[0].config = None, M*4, 
        self.mlp_refine.layers[1].config = None, C*Ih*Iw
        self.mlp_refine.layers[2].shape = -1,C,Ih,Iw
        self.init_flag = True

    def forward(self, x):
        if not self.init_flag:
            self.fix_configuration(x.shape)
        if self.beta > 0:
            h = x + self.beta * self.cnn_refine(x)
        else:
            h = x
        y_base = self.conv(h)
        if self.alpha == 0:
            return self.act(y_base)
        y = self.mlp_refine(y_base)
        y = y_base + self.alpha * y
        return self.act(y)

    def update(self, eta=0.001):
        if self.beta != 0:
            self.cnn_refine.update(eta=eta)
        self.conv.update(eta=eta)
        if self.alpha != 0:
            self.mlp_refine.update(eta=eta)


