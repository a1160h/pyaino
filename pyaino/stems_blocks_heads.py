# stems_blocks_heads
# 20260513 A.Inoue

from pyaino.Config import *
from pyaino import Neuron as nn
from pyaino import Functions as F
from pyaino import Activators as A
from pyaino import LossFunctions as lf
from pyaino import common_function as cf


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


