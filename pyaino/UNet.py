# UNet
# 20260207 A.Inoue

from pyaino.Config import *
#set_derivative(True)
from pyaino import Neuron as neuron
from pyaino import LossFunctions as lf
from pyaino import Functions as F
from pyaino import common_function as cf
import warnings

def center_crop(x, crop_h, crop_w):
    h, w = x.shape[-2:]
    
    # 開始位置の計算
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    # スライス
    return x[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]

class ConvBlock:
    def __init__(self, out_ch, proj=False,
                 **kwargs):
        print('__init__', self.__class__.__name__, out_ch, kwargs)
        self.out_ch = out_ch
        # Conv 本体（bottleneck 構造）
        self.convs = [neuron.Conv2dLayer(out_ch, 3, 1, **kwargs),
                      neuron.Conv2dLayer(out_ch, 3, 1, **kwargs)]
        # embedding からのベクトル加算用
        if proj:
            opt_proj = {'optimize':  kwargs.get('optimize', 'SGD'),
                        'w_decay':   kwargs.get('w_decay')}
            self.proj = neuron.LinearLayer(out_ch, **opt_proj) # 出力幅未定
        else:
            self.proj = None
        self.proj_used = False      

    def forward(self, x, v=None, train=True):
        self.proj_used = False
        y = self.convs[0](x, train=train)
        if (self.proj is not None) and (v is not None): # timeやlabelのmlpからの注入口
            z = self.proj(v, train=train)               # projで形状を合わせて
            y = y + z[:,:,None,None]                    # 加算注入
            self.proj_used = True  
        y = self.convs[1](y, train=train)   
        return y
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, eta=0.001, **kwargs):
        if self.proj is not None and self.proj_used:
            self.proj.update(eta=eta, **kwargs)
        for conv in self.convs:
            conv.update(eta=eta, **kwargs)    
        
class ConvBlockBottleneck(ConvBlock):
    """
    1x1 Conv を使ったボトルネック型 ConvBlock
    - in_ch  -> mid_ch (1x1)
    - mid_ch -> mid_ch (3x3)
    - mid_ch -> out_ch (1x1)
    """
    def __init__(self, out_ch, proj=False, 
                 bottleneck_ratio=0.5, min_mid_ch=16,
                 **kwargs):
        print('__init__', self.__class__.__name__, out_ch,
              proj, bottleneck_ratio, min_mid_ch, kwargs)
        self.out_ch = out_ch
       
        # mid_ch を下限を設けて決定(in_chとout_chの両方を見る)
        mid_ch = max(int(out_ch * bottleneck_ratio), min_mid_ch)

        # Conv 本体（bottleneck 構造）
        self.convs = [
            neuron.Conv2dLayer(mid_ch, 1, 0, **kwargs), # 1x1: in_ch  -> mid_ch
            neuron.Conv2dLayer(mid_ch, 3, 1, **kwargs), # 3x3: mid_ch -> mid_ch（padding=1 前提）
            neuron.Conv2dLayer(out_ch, 1, 0, **kwargs), # 1x1: mid_ch -> out_ch
        ]
        # embedding からのベクトル加算用
        if proj:
            opt_proj = {'optimize':  kwargs.get('optimize', 'SGD'),
                        'w_decay':   kwargs.get('w_decay')}
            self.proj = neuron.LinearLayer(mid_ch, **opt_proj)
        else:
            self.proj = None
        self.proj_used = False      

    def forward(self, x, v=None, train=True):
        self.proj_used = False
        y = self.convs[0](x, train=train)
        y = self.convs[1](y, train=train)
        if (self.proj is not None) and (v is not None): # timeやlabelのmlpからの注入口
            z = self.proj(v, train=train)               # projで形状を合わせて
            y = y + z[:,:,None,None]                    # 加算注入
            self.proj_used = True  
        y = self.convs[2](y, train=train)   
        return y

class UNet:
    """ 完全畳み込み構造のUNetで(H,W)は2のべき乗でなくても対応 """

    def __init__(self, depth=3, in_ch=None,
                 time_embed=False, num_labels=None, embed_dim=128,  
                 base_ch=32, bottleneck=True, bottleneck_ratio=0.5, 
                 batchnorm=None, layernorm=None, activate='ReLU', 
                 optimize='AdamT', w_decay=0.01, bias_last=False):
        warnings.warn(self.__class__.__name__
                      +"Use this module with 'set_derivative(True)'.")
        
        self.in_ch = in_ch
        self.time_embed = time_embed
        self.num_labels = num_labels
        self.embed_dim = embed_dim 
        self.depth = depth
        self.optimize = optimize
        self.base_ch = base_ch
        self.bottleneck = bottleneck
        self.bottleneck_ratio = bottleneck_ratio

        # timeおよびlabelのembeddingとその次元変換用のmlp
        if time_embed or (num_labels is not None):
            options_for_mlpn = {'batchnorm' : batchnorm,
                                'layernorm' : layernorm,
                                'activate' : activate,
                                'optimize' : optimize,
                                'w_decay'  : w_decay}
            options_for_mlpo = {'optimize' : optimize,
                                'w_decay'  : w_decay}
            proj = True  # Convのprojを有効に
        else:
            proj = False # Convのprojを無効に

        if time_embed:
            self.time_mlp = neuron.Sequential(
                neuron.PositionalEncoding(dimension=embed_dim),
                neuron.NeuronLayer(base_ch*4, **options_for_mlpn),
                neuron.LinearLayer(embed_dim, **options_for_mlpo))
        else:
            self.time_mlp = None

        if num_labels is not None:
            self.label_mlp = neuron.Sequential(
                neuron.Embedding(num_labels, embed_dim, optimize=optimize),
                neuron.NeuronLayer(base_ch*4, **options_for_mlpn),
                neuron.LinearLayer(embed_dim, **options_for_mlpo))
        else:
            self.label_mlp = None

        self.time_mlp_used = None
        self.label_mlp_used = None

        # 本体＝畳込みブロック
        if bottleneck:
            Conv = ConvBlockBottleneck
        else:
            Conv = ConvBlock

        options_for_blocks = {'proj'     : proj,      # projの有効無効を指定
                              'batchnorm': batchnorm,
                              'layernorm': layernorm,
                              'activate' : activate,
                              'optimize' : optimize,
                              'w_decay'  : w_decay}
        if bottleneck:
            options_for_blocks['bottleneck_ratio'] = bottleneck_ratio

        options_for_ol     = {'bias'     : bias_last,
                              'optimize' : optimize,
                              'batchnorm': batchnorm,
                              'layernorm': layernorm,
                              'w_decay'  : w_decay}

        print(options_for_blocks, options_for_ol)

        # チャネル構成の決定
        c_down = [base_ch * (2 ** i) for i in range(depth)]
        c_bot  =  base_ch * (2 ** depth)
        c_up   = [base_ch * (2 ** i) for i in reversed(range(depth))]

        # Down path
        self.down , self.pool = [], []
        for i in range(depth):
            self.down.append(Conv(c_down[i], **options_for_blocks))
            self.pool.append(neuron.Pooling2dLayer(2))
                
        # Bottleneck
        self.bot = Conv(c_bot, **options_for_blocks)

        # Up path（Upsample + concat + Conv）
        self.upsample, self.concat, self.up = [], [], []
        for i in range(depth):
            self.upsample.append(neuron.Interpolate2d(scale_factor=2, mode='bilinear'))
            self.concat.append(F.Concatenate()) 
            self.up.append(Conv(c_up[i], **options_for_blocks))

        # 出力 1x1 Conv（チャネルだけin_chに戻す。forwardまで確定しない場合もある）
        self.out = neuron.Conv2dLayer(in_ch, 1, 0, **options_for_ol)

    def fix_out_ch(self, shape):
        """ 出力のConv2dの出力チャネル数の確定(入力の形状を見て合わせる) """
        if self.out.config[3] is not None:
            return
        else:
            self.in_ch = shape[1]  # shape=(B,C,Ih,Iw)
            #print(self.out.config, self.in_ch)
            org_out_config = self.out.config
            list_out_config = list(self.out.config)  
            list_out_config[3] = self.in_ch
            self.out.config = tuple(list_out_config)
            #print(self.out.config)

    def forward(self, x, timesteps=None, labels=None, train=True):
        self.fix_out_ch(x.shape)
        self.time_mlp_used = False
        self.label_mlp_used = False
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

        # Down
        for i in range(self.depth):
            shapes.append(x.shape) # Down前の元の形状を記録
            z = self.down[i](x, v, train=train)      # H, W
            x  = self.pool[i](z)                     # H/2, W/2
            zs.append(z)           # 中間結果を記録 

        # Bottleneck
        x  = self.bot(x, v, train=train)             # H/8, W/8

        # Up
        for i in range(self.depth):
            shape = shapes.pop()   # 元の形状=Up変換後の形状　 
            z = zs.pop()
            x = self.upsample[i](x)                   # H/4, W/4
            if x.shape[-2:] != shape[-2:]:            # 形状が違う場合 
                x = x[:, :, 0:shape[-2], 0:shape[-1]] # 元の形状にトリミング
                x2 = center_crop(x, shape[-2], shape[-1]) # 検証用仮実装(上の行でOKなはずだが確認のため)
                assert (x==x2).all()   
            x = self.concat[i](x, z, axis=1)          # C4 + C3
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

        for i in range(self.depth):
            self.down[i].update(eta=eta, **kwargs)
        self.bot.update(eta=eta, **kwargs)
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



if __name__=='__main__':
    set_derivative(True)

    for i in range(10):
        h, w = np.random.randint(1, 100, 2)
        c = np.random.randint(1, 5)
        b = np.random.randint(1,10)
        x = np.random.rand(int(b), int(c), int(h), int(w))

        model = UNet()#in_ch=int(c))
        print('\n', f'##### test No.{i} x.shape = {x.shape} #####')
        y = model(x)
        y.backtrace()
        gx = model.down[0].convs[0].inputs[0].grad
        print(f'##### y.shape = {y.shape} gx.shape = {gx.shape} #####')

