# stems_blocks_heads
# 20260825 A.Inoue

from pyaino.Config import *
from pyaino import nucleus
from pyaino import Neuron as nn
from pyaino import Functions as F
from pyaino import Activators as A
from pyaino import LossFunctions as lf
from pyaino import common_function as cf

class FeedForward: 
    """ a simple linear layer followed bu a non-linearity """

    def __init__(self, emb_dim=64, expansion=4, activate='Mish', **kwargs):
        activator = cf.eval_in_module(activate, A)
        self.net = nn.Sequential(
              nn.LinearLayer(emb_dim, emb_dim*expansion, matmul=True, **kwargs),
              activator, # オリジナルはReLU
              nn.LinearLayer(emb_dim*expansion, emb_dim, matmul=True, **kwargs),
              nn.Dropout(),
              )

    def forward(self, x, dropout=0.0):
        y = self.net.forward(x, dropout=dropout)
        return y

    def backward(self, gy=None):
        if gy is None:
            gy = np.ones_like(self.y)
        gx = self.net.backward(gy)    
        return gx

    def update(self, **kwargs):
        self.net.update(**kwargs)

class TransformerBlock(nucleus.Function):
    """ Transformer block: communication followed by computation """

    def __init__(self, emb_dim=64, n_head=4, causality=None, proj=False, 
                 expansion=4, rms=False, activate='Mish', **kwargs):
        super().__init__()
        self.sa = nn.MultiHeadSelfAttention(
            emb_dim, emb_dim//n_head, n_head, causality=causality, **kwargs) # entropy制御はkwargsで指定
        self.ffwd = FeedForward(emb_dim, expansion, activate=activate, **kwargs)
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

        z += x # res接続
        if (self.proj is not None) and (v is not None): # timeやlabelのmlpからの注入口
            u = self.proj(v)                            # projで形状を合わせて
            z += u[:,None,:]                            # 加算注入
            self.proj_used = True 

        y = self.ln2.forward(z)
        y = self.ffwd.forward(y, dropout=dropout)
        y += z # res接続
        self.y = y
        return y

    def __forward__bkup(self, x, v=None, mask=None, dropout=0.0):
        """ インプレース更新をしない """
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

        if not self.proj_used:
            return gx
        return gx, gv

    def update(self, **kwargs):
        self.sa.update(**kwargs)

        if self.proj is not None and self.proj_used:
            self.proj.update(**kwargs)

        self.ffwd.update(**kwargs)
        self.ln1.update(**kwargs)
        self.ln2.update(**kwargs)


class ConvBlock(nucleus.Function):
    """
    加算注入されるベクトル入力を備え残差接続可能な基本構成のConvBlock
    - in_ch  -> out_ch (3x3)
    - out_ch -> out_ch (3x3)
    """
    def __init__(self, out_ch, stride=1, proj=False, 
                 residual=False, attention=False, pre_activation=False, 
                 **kwargs):
        super().__init__()
        print('__init__', self.__class__.__name__, out_ch, stride, kwargs)
        self.out_ch = out_ch
        # Conv 本体（非bottleneck 構造）
        activate = kwargs.pop('activate', 'ReLU')
        self.convs = [
            nn.Conv2dLayer(out_ch, 3, stride, 1,
                           activate=activate, pre_activation=pre_activation,
                           **kwargs),
            nn.Conv2dLayer(out_ch, 3, 1, 1,
                           activate=activate, pre_activation=pre_activation,
                           residual=residual, # 残差接続の注入点
                           **kwargs)
            ]
        n_head = max(out_ch//32, 1) 
        self.n_head = n_head
        
        # embedding からのベクトル加算用
        if proj:
            self.proj = nn.LinearLayer(out_ch, **kwargs) # 出力幅未定
        else:
            self.proj = None
        self.proj_used = False
        self.residual = residual      # 残差接続の有無 
        if residual:    
            self.shortcut = nn.Conv2dLayer(out_ch, 1, stride, 0, **kwargs)
        self.shortcut_used = False    
        if attention:
            self.attn = nn.SpatialSelfAttention(n_head=n_head, **kwargs)
        else:
            self.attn = None
    
    def __forward__(self, x, v=None, train=True):
        in_ch = x.shape[1]
        self.proj_used = False
        self.shortcut_used=False 
        y = self.convs[0](x, train=train)
        if (self.proj is not None) and (v is not None): # timeやlabelのmlpからの注入口
            z = self.proj(v, train=train)               # projで形状を合わせて
            y = y + z[:,:,None,None]                    # 加算注入
            self.proj_used = True
        if self.attn is not None:
            y = self.attn(y)                            # attentionを挿入
        if self.residual:
            if in_ch == self.out_ch:
                y = self.convs[1](y, x, train=train)    # resはx直結
            else:
                r = self.shortcut(x)
                y = self.convs[1](y, r, train=train)    # resはshortcut経由
                self.shortcut_used = True               # update有無のフラグ
        else:
            y = self.convs[1](y, train=train)
        return y
        
    def __backward__(self, gy):
        # convs[1] 側
        if self.residual:
            gy, gr = self.convs[1].backward(gy)

            if self.shortcut_used:
                gx = self.shortcut.backward(gr)
            else:
                gx = gr
        else:
            gy = self.convs[1].backward(gy)
            gx = 0

        # attention
        if self.attn is not None:
            gy = self.attn.backward(gy)

        # time / label projection injection
        if self.proj_used:
            gz = gy.sum(axis=(2, 3))
            gv = self.proj.backward(gz)
        else:
            gv = None

        # convs[0]
        gx = self.convs[0].backward(gy) + gx

        if self.proj_used:
            return gx, gv
        else:
            return gx
    

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, eta=0.001, **kwargs):
        if self.proj is not None and self.proj_used:
            self.proj.update(eta=eta, **kwargs)
        for conv in self.convs:
            conv.update(eta=eta, **kwargs)
        if self.residual and self.shortcut_used:
            self.shortcut.update(eta=eta, **kwargs)
            
        
class ConvBlockBottleneck(ConvBlock):
    """
    1x1 Conv を使ったボトルネック型 ConvBlock
    - in_ch  -> mid_ch (1x1)
    - mid_ch -> mid_ch (3x3)
    - SpatialSelfAttention
    - mid_ch -> out_ch (1x1)
    """
    def __init__(self, out_ch, stride=1, proj=False, bottleneck_ratio=0.5, min_mid_ch=16,
                 residual=False, attention=False, pre_activation=False,
                 **kwargs):
        nucleus.Function.__init__(self)
        print('__init__', self.__class__.__name__, out_ch,
              proj, bottleneck_ratio, min_mid_ch, kwargs)
        self.out_ch = out_ch
       
        # mid_ch を下限を設けて決定(in_chとout_chの両方を見る)
        mid_ch = max(int(out_ch * bottleneck_ratio), min_mid_ch)

        # Conv 本体（bottleneck 構造）
        activate = kwargs.pop('activate', 'ReLU')
        if pre_activation:
            activates = activate, activate, activate
        else: # 通常は3段目は直出力(残差接続の接続点)
            activates = activate, activate, None
        self.convs = [
            nn.Conv2dLayer(mid_ch, 1, 1, 0,# 1x1: in_ch  -> mid_ch
                           activate=activates[0], pre_activation=pre_activation,
                           **kwargs), 
            nn.Conv2dLayer(mid_ch, 3, stride, 1,# 3x3: mid_ch -> mid_ch（padding=1 前提）
                           activate=activates[1], pre_activation=pre_activation,
                           **kwargs), 
            nn.Conv2dLayer(out_ch, 1, 1, 0,# 1x1: mid_ch -> out_ch
                           activate=activates[2], pre_activation=pre_activation,
                           residual=residual, # 残差接続の注入点
                           **kwargs),
            ]
        n_head = max(mid_ch//32, 1)
        self.n_head = n_head
        # embedding からのベクトル加算用
        if proj:
            self.proj = nn.LinearLayer(mid_ch, **kwargs)
        else:
            self.proj = None
        self.proj_used = False      
        self.residual = residual
        if residual:
            self.shortcut = nn.Conv2dLayer(out_ch, 1, stride, 0, **kwargs)
        self.shortcut_used = False    
        if attention:
            self.attn = nn.SpatialSelfAttention(n_head=n_head, **kwargs)
        else:
            self.attn = None

    def __forward__(self, x, v=None, train=True):
        in_ch = x.shape[1]
        self.proj_used = False
        self.shortcut_used=False 
        y = self.convs[0](x, train=train)
        y = self.convs[1](y, train=train)
        if (self.proj is not None) and (v is not None): # timeやlabelのmlpからの注入口
            z = self.proj(v, train=train)               # projで形状を合わせて
            y = y + z[:,:,None,None]                    # 加算注入
            self.proj_used = True  
        if self.attn is not None:
            y = self.attn(y)                            # attentionを挿入
        if self.residual:
            if in_ch == self.out_ch:
                y = self.convs[2](y, x, train=train)    # resはx直結
            else:
                r = self.shortcut(x)
                y = self.convs[2](y, r, train=train)    # resはshortcut経由
                self.shortcut_used = True               # update有無のフラグ
        else:
            y = self.convs[2](y, train=train)
        return y


    def __backward__(self, gy):
        # convs[2] 側
        if self.residual:
            gy, gr = self.convs[2].backward(gy)

            if self.shortcut_used:
                gx = self.shortcut.backward(gr)
            else:
                gx = gr
        else:
            gy = self.convs[2].backward(gy)
            gx = 0

        # attention
        if self.attn is not None:
            gy = self.attn.backward(gy)

        # time / label projection injection
        if self.proj_used:
            gz = gy.sum(axis=(2, 3))
            gv = self.proj.backward(gz)
        else:
            gv = None

        # convs[1] -> convs[0]
        gy = self.convs[1].backward(gy)
        gx = self.convs[0].backward(gy) + gx

        if self.proj_used:
            return gx, gv
        else:
            return gx
    


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

class ClassificationHead:
    def __init__(self, classes=10, **kwargs):
        options_for_hidden = {'batchnorm'  : kwargs.pop('batchnorm',  False), 
                              'layernorm'  : kwargs.pop('layernorm',  False),
                              'normaffine' : kwargs.pop('normaffine', False),
                              'optimize'   : kwargs.get('optimize',   'SGD'),
                              'w_decay'    : kwargs.get('w_decay',      0.0),
                              'activate'   : kwargs.pop('activate',  'ReLU'),
                              }

        options_for_output = {'activate'   : kwargs.pop('ol_act', 'Softmax'),
                              'optimize'   : kwargs.pop('optimize',   'SGD'),
                              'w_decay'    : kwargs.pop('w_decay',      0.0),
                              }
        
        self.net = [nn.GlobalAveragePooling(   **options_for_hidden),
                    nn.NeuronLayer(classes,    **options_for_output),
                    ]
        self.loss_function = lf.CrossEntropyError()

    def forward(self, x, t=None, train=True, dropout=0.0):
        y = self.net[0](x, train=train, dropout=dropout) 
        y = self.net[1](y)  
        if t is None:
            return y
        l = self.loss_function(y, t)            
        return y, l  

    def update(self, **kwargs):
        self.net[1].update(**kwargs)

class LmHead:
    """ 言語モデル用のHead """
    # 隠れ状態を語彙サイズのベクトルに変換,unifyに従い最終LN～最終層～損失関数を構築
    def __init__(self, emb_dim, vocab_size, matmul=True, unify=True, rms=False, 
                       tile_size=200, ignore=-1, **kwargs):
        optimize = kwargs.get('optimize', 'AdamT')
        if rms:
            self.ln_f = nn.RMSNormalization(optimize=optimize)
        else:    
            self.ln_f = nn.LayerNormalization(optimize=optimize) #mask_enable=True) 
        if unify: # 損失関数までの一体処理
            self.linear_layer = nn.LinearLayerCrossEntropy(
                emb_dim, vocab_size, matmul=matmul, tile_size=tile_size,
                **kwargs)
        else:     # 機能別処理
            self.linear_layer = nn.LinearLayer(
                emb_dim, vocab_size, matmul=matmul, **kwargs)
            self.loss_function = lf.CrossEntropyErrorForLogits(
                ignore=ignore,
                reduction='mean', # 1token 当たりの平均 negative log likelihood
                )
        self.unify = unify

    def forward(self, x, targets=None):
        x = self.ln_f(x)
        if self.unify:
            y = self.linear_layer(x, targets) # logits.shape=(B,T,vocab_size)
            return y # (max_index, max_logit) if targets is None else (max_index, loss) 
        logits = self.linear_layer(x) # logits.shape=(B,T,vocab_size)
        if targets is None:
            return logits
        loss = self.loss_function(logits, targets)
        return logits, loss
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, gy=None):
        if self.unify:
            gx = self.linear_layer.backward()
        else:
            if gy is None: # gyが指定されない場合は損失関数から得る
                gy = self.loss_function.backward()
            gx = self.linear_layer.backward(gy)
        gx = self.ln_f.backward(gx)
        return gx     

    def update(self, **kwargs):
        self.ln_f.update(**kwargs)
        self.linear_layer.update(**kwargs)

    def accommodate(self):
        self.linear_layer.accommodate()



class TransformerBlock_bkup: # 使わなくなった20260616AI
    """ Transformer block: communication followed by computation """

    def __init__(self, emb_dim=64, n_head=4, block_size=500, rms=False,
                  causality=None, activate='Mish', **kwargs):
        # emb_dim: embedding dimension, n_head: the number of heads we'd like
        self.sa = nn.MultiHeadSelfAttention(
            emb_dim, emb_dim//n_head, n_head, causality=causality, **kwargs) # entropy制御はkwargsで指定
        #self.ffwd = FeedForward(emb_dim, n_head, **kwargs)
        #self.ln1 = nn.Normalization(axis=-1, mask_enable=True) # layer normalization
        #self.ln2 = nn.Normalization(axis=-1, mask_enable=True) # layer normalization
        self.ffwd = nn.Sequential(
            nn.NeuronLayer(emb_dim, emb_dim*n_head, matmul=True, activate=activate, **kwargs),
            nn.NeuronLayer(emb_dim*n_head, emb_dim, matmul=True, dropout=True, **kwargs),
            )
        if rms:
            self.ln1 = nn.RMSNormalization(**kwargs)
            self.ln2 = nn.RMSNormalization(**kwargs)
        else:    
            self.ln1 = nn.LayerNormalization(**kwargs) # 20250515AI
            self.ln2 = nn.LayerNormalization(**kwargs) # 20250515AI
            
    def forward(self, x, mask=None, dropout=0.0):
        z = self.ln1.forward(x)
        z = self.sa.forward(z, mask=mask, dropout=dropout)
        #z += x # 自動微分で問題 20250528AI
        z = z + x
        y = self.ln2.forward(z)
        y = self.ffwd.forward(y, dropout=dropout)
        #y += z # 自動微分で問題 20250528AI
        y = y + z
        self.y = y
        return y

    def backward(self, gy=None):
        if gy is None:
            gy = np.ones_like(self.y)
        gz = self.ffwd.backward(gy)
        gz = self.ln2.backward(gz)
        gz += gy
        gx = self.sa.backward(gz)
        gx = self.ln1.backward(gx)
        gx += gz
        return gx

    def update(self, **kwargs):
        self.sa.update(**kwargs)
        self.ffwd.update(**kwargs)
        self.ln1.update(**kwargs)
        self.ln2.update(**kwargs)

