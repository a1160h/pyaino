# UNet
# 20260523 A.Inoue

from pyaino.Config import *
#set_derivative(True)
from pyaino import Neuron as nn
from pyaino import Activators
from pyaino import LossFunctions as lf
from pyaino import Functions as F
from pyaino import common_function as cf
import warnings
import copy

class ConvBlock:
    """
    加算注入されるベクトル入力を備え残差接続可能な基本構成のConvBlock
    - in_ch  -> out_ch (3x3)
    - out_ch -> out_ch (3x3)
    """
    def __init__(self, out_ch, stride=1, proj=False, 
                 residual=False, attention=False,
                 activate=(None, None), pre_activation=False, 
                 **kwargs):
        print('__init__', self.__class__.__name__, out_ch, stride, kwargs)
        self.out_ch = out_ch
        # Conv 本体（非bottleneck 構造）
        self.convs = [
            nn.Conv2dLayer(out_ch, 3, stride, 1,
                           activate=activate[0], pre_activation=pre_activation,
                           **kwargs),
            nn.Conv2dLayer(out_ch, 3, 1, 1,
                           activate=activate[1], pre_activation=pre_activation,
                           residual=residual, # 残差接続の注入点
                           **kwargs)
            ]
        n_head = out_ch//32 
        self.n_head = n_head
        opt_for_opt = {'optimize' : kwargs.pop('optimize', 'SGD'),
                        'w_decay' : kwargs.pop('w_decay',    0.0),}
        # embedding からのベクトル加算用
        if proj:
            self.proj = nn.LinearLayer(out_ch, **opt_for_opt) # 出力幅未定
        else:
            self.proj = None
        self.proj_used = False
        self.residual = residual      # 残差接続の有無 
        #self.shortcut = None          # 残差接続の機構
        if residual:    
            self.res_conv = nn.Conv2dLayer(out_ch, 1, stride, 0, **opt_for_opt)
            self.shortcut = None # 遅延初期化
        if attention:
            self.attn = nn.SpatialSelfAttention(n_head=n_head, **kwargs)
        else:
            self.attn = None
    
    def forward(self, x, v=None, train=True):
        in_ch = x.shape[1]

        if self.shortcut is None:
            if in_ch == self.out_ch:
                self.shortcut = F.Assign()   # identity
            else:
                self.shortcut = self.res_conv
        
        self.proj_used = False
        y = self.convs[0](x, train=train)
        if (self.proj is not None) and (v is not None): # timeやlabelのmlpからの注入口
            z = self.proj(v, train=train)               # projで形状を合わせて
            y = y + z[:,:,None,None]                    # 加算注入
            self.proj_used = True
        if self.attn is not None:
            y = self.attn(y)                            # attentionを挿入
        if self.residual:
            r = self.shortcut(x)
            y = self.convs[1](y, r, train=train)
        else:
            y = self.convs[1](y, train=train)
        return y
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, eta=0.001, **kwargs):
        if self.proj is not None and self.proj_used:
            self.proj.update(eta=eta, **kwargs)
        for conv in self.convs:
            conv.update(eta=eta, **kwargs)
        if self.residual and hasattr(self.shortcut, 'update'):
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
                 residual=False, attention=False,
                 activate=(None, None, None), pre_activation=False,
                 **kwargs):
        print('__init__', self.__class__.__name__, out_ch,
              proj, bottleneck_ratio, min_mid_ch, kwargs)
        self.out_ch = out_ch
       
        # mid_ch を下限を設けて決定(in_chとout_chの両方を見る)
        mid_ch = max(int(out_ch * bottleneck_ratio), min_mid_ch)

        # Conv 本体（bottleneck 構造）
        self.convs = [
            nn.Conv2dLayer(mid_ch, 1, 1, 0,# 1x1: in_ch  -> mid_ch
                           activate=activate[0], pre_activation=pre_activation,
                           **kwargs), 
            nn.Conv2dLayer(mid_ch, 3, stride, 1,# 3x3: mid_ch -> mid_ch（padding=1 前提）
                           activate=activate[1], pre_activation=pre_activation,
                           **kwargs), 
            nn.Conv2dLayer(out_ch, 1, 1, 0,# 1x1: mid_ch -> out_ch
                           activate=activate[2], pre_activation=pre_activation,
                           residual=residual, # 残差接続の注入点
                           **kwargs),
            ]
        n_head = mid_ch//32
        self.n_head = n_head
        opt_for_opt = {'optimize' : kwargs.pop('optimize', 'SGD'),
                        'w_decay' : kwargs.pop('w_decay',    0.0),}
        # embedding からのベクトル加算用
        if proj:
            self.proj = nn.LinearLayer(mid_ch, **opt_for_opt)
        else:
            self.proj = None
        self.proj_used = False      
        self.residual = residual
        #self.shortcut = None          # 残差接続の機構
        if residual:
            self.res_conv = nn.Conv2dLayer(out_ch, 1, stride, 0, **opt_for_opt)
            self.shortcut = None # 遅延初期化
        if attention:
            self.attn = nn.SpatialSelfAttention(n_head=n_head, **kwargs)
        else:
            self.attn = None

    def forward(self, x, v=None, train=True):
        in_ch = x.shape[1]

        if self.shortcut is None:
            if in_ch == self.out_ch:
                self.shortcut = F.Assign()   # identity
            else:
                self.shortcut = self.res_conv
                
        self.proj_used = False
        y = self.convs[0](x, train=train)
        y = self.convs[1](y, train=train)
        if (self.proj is not None) and (v is not None): # timeやlabelのmlpからの注入口
            z = self.proj(v, train=train)               # projで形状を合わせて
            y = y + z[:,:,None,None]                    # 加算注入
            self.proj_used = True  
        if self.attn is not None:
            y = self.attn(y)                            # attentionを挿入
        if self.residual:
            r = self.shortcut(x)
            y = self.convs[2](y, r, train=train)
        else:
            y = self.convs[2](y, train=train)
        return y

def palindrom(n):
    """ 2のべき乗が真ん中を最高にして対称に上り下りして並ぶ """
    # 中央の値（n 以下の最大の 2 の冪）
    center = 1
    while center * 2 <= n:
        center *= 2
    # 左側（1,2,4,...,center）
    left = []
    x = 1
    while x < center:
        left.append(x)
        x *= 2
    # 右側（left の逆順）
    right = left[::-1]
    return left + [center] + right


class UNet:
    """ 完全畳み込み構造のUNetで(H,W)は2のべき乗でなくても対応 """
    def __init__(self, depth=3, in_ch=None, base_ch=32,
        bottleneck=True, bottleneck_ratio=0.5, n_bottom=1, attention=False,
        **kwargs):
        # ---- U-Net core skeleton ----
        self.depth = depth
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.bottleneck = bottleneck
        self.bottleneck_ratio = bottleneck_ratio
        self.n_bottom = n_bottom
        self.attention = attention

        # ---- embedding / conditioning ----
        self.time_embed  = kwargs.pop('time_embed', False)
        self.num_labels  = kwargs.pop('num_labels',  None)
        self.embed_dim   = kwargs.pop('embed_dim',    128)

        # ---- architecture variations ----
        self.stem_kernel = kwargs.pop('stem_kernel',    1)
        self.skip_ratio  = kwargs.pop('skip_ratio',  None)
        self.residual    = kwargs.pop('residual',   False)
        self.bias_last   = kwargs.pop('bias_last',  False)

        # ---- activation / normalization ----
        self.activate    = kwargs.pop('activate',  'ReLU')
        self.pre_activation = kwargs.pop('pre_activation', False)
        self.optimize    = kwargs.pop('optimize', 'AdamT')
        self.w_decay     = kwargs.pop('w_decay',     0.01)

        norm_options =   {'batchnorm' : kwargs.pop('batchnorm',    None),
                          'layernorm' : kwargs.pop('layernorm',    None),
                          'normaffine': kwargs.pop('normaffine',  False),}
        
        common_options = {**norm_options,
                          'optimize': self.optimize, 'w_decay': self.w_decay} 

        if kwargs: # Safety check
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")


        # timeおよびlabelのembeddingとその次元変換用のmlp
        if self.time_embed or (self.num_labels is not None):
            options_for_mlpn = {**common_options, 'activate': self.activate,}
            options_for_mlpo = {'optimize': self.optimize, 'w_decay': self.w_decay}
            proj = True  # Convのprojを有効に
        else:
            proj = False # Convのprojを無効に

        if self.time_embed:
            self.time_mlp = nn.Sequential(
                nn.PositionalEncoding(dimension=self.embed_dim),
                nn.NeuronLayer(base_ch*4, **options_for_mlpn),
                nn.LinearLayer(self.embed_dim, **options_for_mlpo))
        else:
            self.time_mlp = None

        if self.num_labels is not None:
            self.label_mlp = nn.Sequential(
                nn.Embedding(self.num_labels, self.embed_dim, optimize=self.optimize),
                nn.NeuronLayer(base_ch*4, **options_for_mlpn),
                nn.LinearLayer(self.embed_dim, **options_for_mlpo))
        else:
            self.label_mlp = None

        self.time_mlp_used = None; self.label_mlp_used = None # forward()でフラグとして使用

        # Stem 入力直は、ActやNormではなく、Convから
        stem_pad = 1 if self.stem_kernel > 1 else 0 # 仮処置20260413AI
        self.stem = nn.Conv2dLayer(base_ch, self.stem_kernel, 1, stem_pad, **common_options)

        # 本体＝畳込みブロックとチャネル構成の決定
        Conv = ConvBlockBottleneck if bottleneck else ConvBlock
        c_down = [base_ch * (2 ** i) for i in range(self.depth)]
        c_bot  = base_ch * (2 ** (self.depth - 1)) 
        c_up   = [base_ch * (2 ** i) for i in reversed(range(self.depth))]

        # 畳込みブロック
        options_for_blocks = {**common_options, 
                              'proj'          : proj,  # projの有効無効を指定
                              'residual'      : self.residual,
                              'pre_activation': self.pre_activation,}
        if bottleneck:
            options_for_blocks['bottleneck_ratio'] = bottleneck_ratio

        if bottleneck and self.pre_activation: 
            options_for_blocks['activate'] = self.activate, self.activate, self.activate
        elif bottleneck:
            options_for_blocks['activate'] = self.activate, self.activate, None
        else:
            options_for_blocks['activate'] = self.activate, self.activate

        options_for_attnblk = options_for_blocks.copy()
        options_for_attnblk['attention'] = self.attention 
            
        # Down path
        self.down , self.pool = [], []
        for i in range(self.depth):
            self.down.append(Conv(c_down[i], **options_for_blocks))
            self.pool.append(nn.Pooling2dLayer(2))
                
        # Bottleneck
        self.bot = []
        attn_pos = (n_bottom - 1) // 2 # この位置だけattentionを入れる
        for i in range(self.n_bottom):
            options = options_for_attnblk #if i==attn_pos else options_for_blocks
            self.bot.append(Conv(c_bot, **options))

        # Up path（Upsample + concat + Conv）
        self.upsample, self.concat, self.up = [], [], []
        for i in range(self.depth):
            self.upsample.append(nn.Interpolate2d(scale_factor=2,
                                          mode='bilinear', align='center'))
            self.concat.append(F.Concatenate())
            self.up.append(Conv(c_up[i], **options_for_blocks))

        # 出力Head  チャネル数合わせのみで、ActもNormも無し
        options_for_ol = {**common_options,
                          'bias'           : self.bias_last,}
        # 出力 1x1 Conv（チャネルだけin_chに戻す。forwardまで確定しない場合もある）
        self.out = nn.Conv2dLayer(in_ch, 1, 0, **options_for_ol)
            
        # down->upのスキップ接続　
        options_for_skip_proj = {'optimize' : self.optimize, 'w_decay' : self.w_decay,}

        # Skip projection (弱スキップ:カーネルサイズ1でskip_ratioに従いチャネル圧縮)
        if self.skip_ratio is not None and self.skip_ratio > 0:
            self.skip_proj = []
            for i in range(depth):
                proj_ch = max(1, round(c_down[i] * skip_ratio))
                self.skip_proj.append(nn.Conv2dLayer(
                                      proj_ch, 1, 0, **options_for_skip_proj)) 

    def fix_out_ch(self, shape):
        """ 出力のConv2dの出力チャネル数の確定(入力の形状を見て合わせる) """
        if self.out.config[3] is not None:
            return
        else:
            self.in_ch = shape[1]  # shape=(B,C,Ih,Iw)
            #print(self.out.config, self.in_ch)
            list_out_config = list(self.out.config)  
            list_out_config[3] = self.in_ch
            self.out.config = tuple(list_out_config)
            #print(self.out.config)

    def center_crop(self, x, crop_h, crop_w):
        h, w = x.shape[-2:]
        # 開始位置の計算
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        # スライス
        return x[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]

    def forward(self, x, timesteps=None, labels=None, train=True):
        self.fix_out_ch(x.shape)
        self.time_mlp_used = False; self.label_mlp_used = False 
        # time/label embedding -> mlp
        if (self.time_mlp is not None) and (timesteps is not None):
            t0 = timesteps
            t0 = self.normalize_t(t0, x.shape[0])    # バッチサイズだけ合わせる
            if (t0 < 0).any() or (t0 >= 1000).any(): # 仮20260107AI
                print('###debug t0', t0)
            time_ctx = self.time_mlp(t0, train=train)
            self.time_mlp_used = True  

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
            
        shapes = []
        zs = []    

        # Stem
        x = self.stem(x)

        # Down
        for i in range(self.depth):
            shapes.append(x.shape) # Down前の元の形状を記録
            z = self.down[i](x, v, train=train)       # H, W
            x  = self.pool[i](z)                      # H/2, W/2
            zs.append(z)           # 中間結果を記録 

        # Bottleneck
        for i in range(self.n_bottom):
            x  = self.bot[i](x, v, train=train)          # H/8, W/8

        # Up
        for i in range(self.depth):
            shape = shapes.pop()   # 元の形状=Up変換後の形状　 
            x = self.upsample[i](x)                   # H/4, W/4
            if x.shape[-2:] != shape[-2:]:            # 形状が違う場合
                # x = x[:, :, 0:shape[-2], 0:shape[-1]]        # 左上基準でもOK
                x = self.center_crop(x, shape[-2], shape[-1]) # center_crop

            z = zs.pop()           # Downパスの中間結果を逆順FILOで取出す
            if self.skip_ratio is None:
                x = self.concat[i](x, z, axis=1)      # C4 + C3
            elif self.skip_ratio > 0:
                z = self.skip_proj[self.depth - 1 - i](z) # skip_projは逆順参照
                x = self.concat[i](x, z, axis=1)
            elif self.skip_ratio == 0:
                pass                                  # skip connection を使わない
            else:
                raise ValueError("skip_ratio must be None or >= 0")
            
            x = self.up[i](x, v, train=train)         # -> C3

        assert not zs 
        # Output
        y  = self.out(x, train=train)                 # (B, in_ch, H, W)
        return y

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, eta=0.001, **kwargs):
        if self.time_mlp is not None and self.time_mlp_used:
            self.time_mlp.update(eta=eta, **kwargs)     
        if self.label_mlp is not None and self.label_mlp_used:
            self.label_mlp.update(eta=eta, **kwargs)     
        self.stem.update(eta=eta, **kwargs)
        for i in range(self.depth):
            self.down[i].update(eta=eta, **kwargs)
        for i in range(self.n_bottom):    
            self.bot[i].update(eta=eta, **kwargs)
        if self.skip_ratio is not None and self.skip_ratio > 0:
            for i in range(self.depth):
                self.skip_proj[i].update(eta=eta, **kwargs)
        for i in range(self.depth):
            self.up[i].update(eta=eta, **kwargs)
        self.out.update(eta=eta, **kwargs)

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


class UNet2:
    """ 完全畳み込み構造のUNetで(H,W)は2のべき乗でなくても対応 """
    # down/bot/up 個別にbottleneck率を指定できるように拡張
    def __init__(self, depth=3, in_ch=None, base_ch=32,
        bottleneck=True, bottleneck_ratio=0.5,
        **kwargs):
        # ---- U-Net core skeleton ----
        self.depth = depth
        self.in_ch = in_ch
        self.base_ch = base_ch

        # ---- bottleneck (bool or tuple) ----
        if isinstance(bottleneck, bool):
            self.bottleneck_down = bottleneck
            self.bottleneck_bot  = bottleneck
            self.bottleneck_up   = bottleneck
        elif isinstance(bottleneck, (tuple, list)):
            if len(bottleneck) != 3:
                raise ValueError("bottleneck must be (down, bot, up)")
            self.bottleneck_down, self.bottleneck_bot, self.bottleneck_up = bottleneck
        else:
            raise TypeError("bottleneck must be bool or tuple")

        # ---- bottleneck_ratio (float or tuple) ----
        if isinstance(bottleneck_ratio, (int, float)):
            self.bottleneck_ratio_down = bottleneck_ratio
            self.bottleneck_ratio_bot  = bottleneck_ratio
            self.bottleneck_ratio_up   = bottleneck_ratio
        elif isinstance(bottleneck_ratio, (tuple, list)):
            if len(bottleneck_ratio) != 3:
                raise ValueError("bottleneck_ratio must be (down, bot, up)")
            (self.bottleneck_ratio_down,
             self.bottleneck_ratio_bot,
             self.bottleneck_ratio_up) = bottleneck_ratio
        else:
            raise TypeError("bottleneck_ratio must be float or tuple")

        # ---- embedding / conditioning ----
        self.time_embed  = kwargs.pop('time_embed', False)
        self.num_labels  = kwargs.pop('num_labels',  None)
        self.embed_dim   = kwargs.pop('embed_dim',    128)

        # ---- architecture variations ----
        self.stem_kernel = kwargs.pop('stem_kernel', 1)
        self.skip_ratio  = kwargs.pop('skip_ratio',  None)
        self.residual    = kwargs.pop('residual',   False)
        self.bias_last   = kwargs.pop('bias_last',  False)

        # ---- activation / normalization ----
        self.activate    = kwargs.pop('activate',  'ReLU')
        self.pre_activation = kwargs.pop('pre_activation', False)
        self.optimize    = kwargs.pop('optimize', 'AdamT')
        self.w_decay     = kwargs.pop('w_decay',     0.01)

        norm_options = {
            'batchnorm' : kwargs.pop('batchnorm',    None),
            'layernorm' : kwargs.pop('layernorm',    None),
            'normaffine': kwargs.pop('normaffine',  False),
        }

        common_options = {
            **norm_options,
            'optimize': self.optimize,
            'w_decay' : self.w_decay,
        }

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        # timeおよびlabelのembeddingとその次元変換用のmlp
        if self.time_embed or (self.num_labels is not None):
            options_for_mlpn = {**common_options, 'activate': self.activate}
            options_for_mlpo = {'optimize': self.optimize, 'w_decay': self.w_decay}
            proj = True   # Convのprojを有効に
        else:
            proj = False  # Convのprojを無効に

        if self.time_embed:
            self.time_mlp = nn.Sequential(
                nn.PositionalEncoding(dimension=self.embed_dim),
                nn.NeuronLayer(base_ch * 4, **options_for_mlpn),
                nn.LinearLayer(self.embed_dim, **options_for_mlpo)
            )
        else:
            self.time_mlp = None

        if self.num_labels is not None:
            self.label_mlp = nn.Sequential(
                nn.Embedding(self.num_labels, self.embed_dim, optimize=self.optimize),
                nn.NeuronLayer(base_ch * 4, **options_for_mlpn),
                nn.LinearLayer(self.embed_dim, **options_for_mlpo)
            )
        else:
            self.label_mlp = None

        self.time_mlp_used = None
        self.label_mlp_used = None  # forward()でフラグとして使用

        # Stem 入力直は、ActやNormではなく、Convから
        stem_pad = 1 if self.stem_kernel > 1 else 0
        self.stem = nn.Conv2dLayer(
            base_ch, self.stem_kernel, 1, stem_pad, **common_options
        )

        # チャネル構成
        c_down = [base_ch * (2 ** i) for i in range(depth)]
        c_bot  =  base_ch * (2 ** depth)
        c_up   = [base_ch * (2 ** i) for i in reversed(range(depth))]

        # pathごとに ConvBlock / ConvBlockBottleneck を切り替える
        ConvDown = ConvBlockBottleneck if self.bottleneck_down else ConvBlock
        ConvBot  = ConvBlockBottleneck if self.bottleneck_bot  else ConvBlock
        ConvUp   = ConvBlockBottleneck if self.bottleneck_up   else ConvBlock

        def make_block_options(use_bottleneck, ratio):
            opts = {
                **common_options,
                'proj'          : proj,
                'residual'      : self.residual,
                'pre_activation': self.pre_activation,
            }
            if use_bottleneck:
                opts['bottleneck_ratio'] = ratio

            if use_bottleneck and self.pre_activation:
                opts['activate'] = (self.activate, self.activate, self.activate)
            elif use_bottleneck:
                opts['activate'] = (self.activate, self.activate, None)
            else:
                opts['activate'] = (self.activate, self.activate)
            return opts

        options_down = make_block_options(self.bottleneck_down, self.bottleneck_ratio_down)
        options_bot  = make_block_options(self.bottleneck_bot,  self.bottleneck_ratio_bot)
        options_up   = make_block_options(self.bottleneck_up,   self.bottleneck_ratio_up)

        # ---- Down ----
        self.down, self.pool = [], []
        for i in range(depth):
            self.down.append(ConvDown(c_down[i], **options_down))
            self.pool.append(nn.Pooling2dLayer(2))

        # ---- Bottleneck ----
        self.bot = ConvBot(c_bot, **options_bot)

        # ---- Up (Upsample + conacat + Conv) ---- 
        self.upsample, self.concat, self.up = [], [], []
        for i in range(depth):
            self.upsample.append(
                nn.Interpolate2d(scale_factor=2, mode='bilinear', align='center')
            )
            self.concat.append(F.Concatenate())
            self.up.append(ConvUp(c_up[i], **options_up))

        # ---- Output(Conv1x1 in_chに合わせる) ----
        options_for_ol = {**common_options, 'bias': self.bias_last}
        self.out = nn.Conv2dLayer(in_ch, 1, 0, **options_for_ol)

        # ---- down->upのスキップ接続 ----
        options_for_skip_proj = {'optimize': self.optimize, 'w_decay' : self.w_decay}

        # Skip projection (弱スキップ:カーネルサイズ1でskip_ratioに従いチャネル圧縮)
        if self.skip_ratio is not None and self.skip_ratio > 0:
            self.skip_proj = []
            for i in range(depth):
                proj_ch = max(1, round(c_down[i] * self.skip_ratio))
                self.skip_proj.append(
                    nn.Conv2dLayer(proj_ch, 1, 0, **options_for_skip_proj)
                )

    def fix_out_ch(self, shape):
        """ 出力のConv2dの出力チャネル数の確定(入力の形状を見て合わせる) """
        if self.out.config[3] is not None:
            return
        else:
            self.in_ch = shape[1]  # shape=(B,C,Ih,Iw)
            list_out_config = list(self.out.config)
            list_out_config[3] = self.in_ch
            self.out.config = tuple(list_out_config)

    def center_crop(self, x, crop_h, crop_w):
        h, w = x.shape[-2:]
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        return x[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]

    def forward(self, x, timesteps=None, labels=None, train=True):
        self.fix_out_ch(x.shape)
        self.time_mlp_used = False
        self.label_mlp_used = False

        # time/label embedding -> mlp
        if (self.time_mlp is not None) and (timesteps is not None):
            t0 = timesteps
            t0 = self.normalize_t(t0, x.shape[0])    # バッチサイズだけ合わせる
            if (t0 < 0).any() or (t0 >= 1000).any():
                print('###debug t0', t0)
            time_ctx = self.time_mlp(t0, train=train)
            self.time_mlp_used = True

        if (self.label_mlp is not None) and (labels is not None):
            label_ctx = self.label_mlp(labels, train=train)
            self.label_mlp_used = True

        # 注入ベクトルの確定
        if self.time_mlp_used and self.label_mlp_used:
            v = time_ctx + label_ctx
        elif self.time_mlp_used:
            v = time_ctx
        elif self.label_mlp_used:
            v = label_ctx
        else:
            v = None

        shapes = []
        zs = []

        # Stem
        x = self.stem(x)

        # Down
        for i in range(self.depth):
            shapes.append(x.shape)            # Down前の元の形状を記録
            z = self.down[i](x, v, train=train)
            x = self.pool[i](z)
            zs.append(z)

        # Bottleneck
        x = self.bot(x, v, train=train)

        # Up
        for i in range(self.depth):
            shape = shapes.pop()
            x = self.upsample[i](x)
            if x.shape[-2:] != shape[-2:]:
                x = self.center_crop(x, shape[-2], shape[-1])

            z = zs.pop()
            if self.skip_ratio is None:
                x = self.concat[i](x, z, axis=1)
            elif self.skip_ratio > 0:
                z = self.skip_proj[self.depth - 1 - i](z)
                x = self.concat[i](x, z, axis=1)
            elif self.skip_ratio == 0:
                pass
            else:
                raise ValueError("skip_ratio must be None or >= 0")

            x = self.up[i](x, v, train=train)

        assert not zs

        # Output
        y = self.out(x, train=train)
        return y

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, eta=0.001, **kwargs):
        if self.time_mlp is not None and self.time_mlp_used:
            self.time_mlp.update(eta=eta, **kwargs)
        if self.label_mlp is not None and self.label_mlp_used:
            self.label_mlp.update(eta=eta, **kwargs)

        self.stem.update(eta=eta, **kwargs)

        for i in range(self.depth):
            self.down[i].update(eta=eta, **kwargs)

        self.bot.update(eta=eta, **kwargs)

        if self.skip_ratio is not None and self.skip_ratio > 0:
            for i in range(self.depth):
                self.skip_proj[i].update(eta=eta, **kwargs)

        for i in range(self.depth):
            self.up[i].update(eta=eta, **kwargs)

        self.out.update(eta=eta, **kwargs)

    def normalize_t(self, t, B=1):
        """
        x: (B,C,H,W)
        t: int (scalar) or array-like (B,)
        returns: t_vec (B,) int32
        """
        if isinstance(t, (int,)):
            return np.full((B,), t, dtype=np.int32)

        t_arr = np.asarray(t)
        if getattr(t_arr, "ndim", 0) == 0:
            return np.full((B,), int(t_arr), dtype=np.int32)

        if t_arr.shape != (B,):
            raise ValueError(f"t must be scalar or shape (B,), got {t_arr.shape}, B={B}")

        return t_arr.astype(np.int32, copy=False)



class CNN_MultiStageStack:
    """ UNetを起源とする汎用的なCNN """

    def __init__(self, scale_direction='down', depth=3, in_ch=None, 
                 outtype=None, outdim=None, 
                 base_ch=32, bottleneck=True, bottleneck_ratio=0.5,
                 stem_kernel=1,
                 residual=False, activate='ReLU', pre_activation=False,
                 batchnorm=None, layernorm=None, normaffine=False, 
                 optimize='AdamT', w_decay=0.01, bias_last=False, ol_act=None,
                 ):
                 
        warnings.warn(self.__class__.__name__
                      +"Use this module with 'set_derivative(True)'.")
        
        self.in_ch = in_ch
        self.depth = depth
        self.optimize = optimize
        self.base_ch = base_ch
        self.bottleneck = bottleneck
        self.bottleneck_ratio = bottleneck_ratio

        common_options = {'batchnorm'  : batchnorm,
                          'layernorm'  : layernorm,
                          'normaffine' : normaffine,
                          'optimize'   : optimize,
                          'w_decay'    : w_decay}

        # Stem 入力直は、ActやNormではなく、Convから
        self.stem = nn.Conv2dLayer(base_ch, stem_kernel, 0, **common_options)

        # 本体＝畳込みブロックとチャネル構成の決定
        Conv = ConvBlockBottleneck if bottleneck else ConvBlock

        if scale_direction == 'down':
            ch = [base_ch * (2 ** i) for i in range(depth)]
        elif scale_direction == 'up':
            ch = [base_ch * (2 ** i) for i in reversed(range(depth))]

        # 畳込みブロック　
        options_for_blocks = {**common_options, 
                              'residual'      : residual,
                              'pre_activation': pre_activation,}
        if bottleneck:
            options_for_blocks['bottleneck_ratio'] = bottleneck_ratio

        if bottleneck and pre_activation: 
            options_for_blocks['activate'] = activate, activate, activate
        elif bottleneck:
            options_for_blocks['activate'] = activate, activate, None
        else:
            options_for_blocks['activate'] = activate, activate

        # Down path (Conv + Pool)
        if scale_direction == 'down':
            self.down, self.pool = [], []
            for i in range(depth):
                self.down.append(Conv(ch[i], **options_for_blocks))
                # 最終層はPoolingを避けてIdentity
                if i == depth - 1: 
                    self.pool.append(F.Assign())
                else:    
                    self.pool.append(nn.Pooling2dLayer(2, dropout=True))
                    
        # Up path（Upsample + Conv）
        elif scale_direction == 'up':
            self.upsample, self.up = [], []
            for i in range(depth):
                self.upsample.append(nn.Interpolate2d(scale_factor=2, mode='bilinear'))
                self.up.append(Conv(ch[i], **options_for_blocks))

        else: 
            raise ValueError(f"scale_direction must be 'up' or 'down'," \
                             + f"got '{scale_direction}'")
        self.scale_direction = scale_direction
        
        # 出力Head
        options_for_ol = {**common_options,
                          'activate': ol_act,
                          'bias' : bias_last,}
        # 出力 1x1 Conv（チャネルだけin_chに戻す。forwardまで確定しない場合もある）
        if outtype is None:
            self.out = None
        elif outtype in ("C", "c", "Conv", "Conv"):     
            self.out = nn.Conv2dLayer(outdim, 1, 0, **options_for_ol)
        elif outtype in("F","f","Full","full","N","n","nn","neuron"):    
            self.out = nn.NeuronLayer(outdim, full_connection=True, **options_for_ol)

    def fix_out_ch(self, shape):
        """ 出力のConv2dの出力チャネル数の確定(入力の形状を見て合わせる) """
        if self.out.config[3] is not None:
            return
        else:
            self.in_ch = shape[1]  # shape=(B,C,Ih,Iw)
            #print(self.out.config, self.in_ch)
            list_out_config = list(self.out.config)  
            list_out_config[3] = self.in_ch
            self.out.config = tuple(list_out_config)
            #print(self.out.config)

    def forward(self, x, train=True, dropout=0.0):
        #print('###debug0', train, dropout)
        # Stem
        x = self.stem(x)
        # Down
        if self.scale_direction == 'down':
            for i in range(self.depth):
                z = self.down[i](x, train=train)       # H, W
                #print(self.pool[i].__class__)
                if hasattr(self.pool[i], 'DO'): # 仮対処20260405AI
                    x = self.pool[i](z, dropout=dropout) # H/2, W/2
                    #print('###debug DO', dropout)
                else:
                    x = self.pool[i](z)

        # Up
        elif self.scale_direction == 'up':
            for i in range(self.depth):
                x = self.upsample[i](x)                # H/4, W/4
                x = self.up[i](x, train=train)         # -> C3

        # Output
        if self.out is None:
            return x
        y = self.out(x, train=train)                  # (B, in_ch, H, W)
        return y

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, eta=0.001, **kwargs):
        self.stem.update(eta=eta, **kwargs)
        if self.scale_direction == 'down':
            for i in range(self.depth):
                self.down[i].update(eta=eta, **kwargs)
        elif self.scale_direction == 'up':
            for i in range(self.depth):
                self.up[i].update(eta=eta, **kwargs)
        if self.out is not None:        
            self.out.update(eta=eta, **kwargs)


class ResStage:
    def __init__(self, depth=3, base_ch=16, stride=2,
                 residual=True, pre_activation=False, **kwargs):
        
        activate  = kwargs.pop('activate',  'ReLU')
        optimize  = kwargs.pop('optimize', 'AdamT')
        w_decay   = kwargs.pop('w_decay',     0.01)
        batchnorm = kwargs.pop('batchnorm',   True)

        options = {'residual':residual,
                   'activate':(activate, activate),
                   'optimize':optimize,
                   'w_decay' :w_decay,
                   'batchnorm':batchnorm}
        
        strides = [i%2+1 for i in range(depth)]
        chanels = [int(base_ch * 2**(0.5*i)) for i in range(depth)]
        print(strides)
        print(chanels)
        if stride not in (1, 2):
            raise ValueError('Invalid stride specified.')

        self.blocks = nn.Sequential(
            *[ConvBlock(chanels[i], strides[i], **options)
              for i in range(depth)]
            )
        
    def forward(self, x, train=True, dropout=0.0):
        return self.blocks.forward(x, train=train)

    def update(self, **kwargs):
        self.blocks.update(**kwargs)


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
         

if __name__=='__main__':
    set_derivative(True)

    for i in range(10):
        h, w = np.random.randint(1, 100, 2)
        c = np.random.randint(1, 5)
        b = np.random.randint(1,10)
        x = np.random.rand(int(b), int(c), int(h), int(w))

        model = UNet(layernorm=True,pre_activation=True, residual=True)#bottleneck=False)#in_ch=int(c))
        print('\n', f'##### test No.{i} x.shape = {x.shape} #####')
        y = model(x)
        y.backtrace()
        gx = model.down[0].convs[0].inputs[0].grad
        print(f'##### y.shape = {y.shape} gx.shape = {gx.shape} #####')

    """#
    for i in range(10):
        h, w = np.random.randint(1, 100, 2)
        c = np.random.randint(1, 5)
        b = np.random.randint(1,10)
        x = np.random.rand(int(b), int(c), int(h), int(w))
        outdim = int(np.random.randint(1,100))

        model = CNN_MultiStageStack(outdim=outdim, bottleneck=False, residual=True,)#in_ch=int(c))
        print('\n', f'##### test No.{i} x.shape = {x.shape} #####')
        y = model(x)
        y.backtrace()
        gx = model.down[0].convs[0].inputs[0].grad
        print(f'##### y.shape = {y.shape} gx.shape = {gx.shape} #####')

    #"""#
