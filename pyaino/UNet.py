# UNet
# 20260123 A.Inoue

from pyaino.Config import *
#set_derivative(True)
from pyaino import Neuron as neuron
from pyaino import LossFunctions as lf
from pyaino import Functions as F
from pyaino import common_function as cf
import warnings

class ConvBlock:
    def __init__(self, in_ch, out_ch, proj=False,
                 **kwargs):
        print('__init__', self.__class__.__name__, in_ch, out_ch, kwargs)
        self.in_ch = in_ch
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
    def __init__(self, in_ch, out_ch, proj=False, 
                 bottleneck_ratio=0.5, min_mid_ch=16,
                 **kwargs):
        print('__init__', self.__class__.__name__, in_ch, out_ch,
              proj, bottleneck_ratio, min_mid_ch, kwargs)
        self.in_ch = in_ch
        self.out_ch = out_ch
       
        # mid_ch を下限を設けて決定
        mid_ch = max(int(in_ch * bottleneck_ratio), min_mid_ch)

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

    def __init__(self, depth=3, in_ch=1,
                 time_embed=False, num_labels=None, embed_dim=128,  
                 base_ch=32, bottleneck=True, bottleneck_ratio=0.5, 
                 batchnorm=None, layernorm=None, activate='ReLU', 
                 optimize='AdamT', w_decay=0.01):
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

        options_for_ol     = {'optimize' : optimize,
                              'batchnorm': batchnorm,
                              'layernorm': layernorm,
                              'w_decay'  : w_decay}

        print(options_for_blocks, options_for_ol)

        # Down path
        self.down , self.pool = [], []
        C0 = in_ch; C1 = base_ch
        for _ in range(depth):
            self.down.append(Conv(C0, C1, **options_for_blocks))
            self.pool.append(neuron.Pooling2dLayer(2))
            C0 = C1; C1 *=2
                
        # Bottleneck
        self.bot = Conv(C0, C1, **options_for_blocks)

        # Up path（Upsample + concat + Conv）
        self.upsample, self.concat, self.up = [], [], []
        for _ in range(depth):
            self.upsample.append(neuron.Interpolate2d(scale_factor=2, mode='bilinear'))
            self.concat.append(F.Concatenate()) 
            self.up.append(Conv(C1 + C0, C0, **options_for_blocks))
            C0 //= 2; C1 //= 2

        # 出力 1x1 Conv（チャネルだけ in_ch に戻す）
        assert C1 == base_ch, f'{C1} {base_ch}'
        self.out = neuron.Conv2dLayer(in_ch, 1, 0, **options_for_ol)

    def forward(self, x, timesteps=None, labels=None, train=True):
        assert x.shape[1] == self.in_ch, \
                             f"input C={x.shape[1]} but model in_ch={self.in_ch}"
        self.time_mlp_used = False
        self.label_mlp_used = False
        v = None
        # time/label embedding -> mlp
        if self.time_mlp is None or timesteps is None:
            time_ctx = 0
        else:
            t0 = timesteps
            t0 = self.normalize_t(t0, x.shape[0])    # バッチサイズだけ合わせる
            if (t0 < 0).any() or (t0 >= 1000).any(): # 仮20260107AI
                print('###debug t0', t0)
            time_ctx = self.time_mlp(t0, train=train)
            self.time_mlp_used = True

        if self.label_mlp is None or labels is None:
            label_ctx = 0
        else:
            label_ctx = self.label_mlp(labels, train=train)
            self.label_mlp_used = True

        v = time_ctx + label_ctx
        
        if isinstance(v, (int, float)) and v == 0:   # 仮処置20260123AI
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

    #### 2. CIFAR-10 データロード ####
    set_derivative(True)
    from pyaino import CIFER10
    import time

    epoch = 5
    batch_size = 160
    interval = 1

    input_train, correct_train, target_train, input_test, correct_test, target_test \
        = CIFER10.get_data(normalize=True, image=True)
    label_list = CIFER10.label_list()

    # -- モデルの生成 -- 
    model = UNet(in_ch=3,
                 depth=1,
                 time_embed=True,
                 num_labels=10,
                 bottleneck=False, #True,
                 )
    loss_func = lf.MeanSquaredError()   # 再構成誤差

    input('wait')
    cf.get_obj_info(model)

    # -- 学習データの選択用 --
    errors, accuracy = [], []
    index_rand = np.arange(len(input_train))
    start = time.time()
    # -- 学習と経過の記録 --
    for i in range(epoch):
        np.random.shuffle(index_rand)             # epoch毎index_randをランダマイズ  
        # -- 学習 --
        for j in range(0, len(input_train), batch_size): # 0～n_trainまでbatch_sizeずつ更新
            mb_index = index_rand[j:j+batch_size] # index_randからbatch_sizeずつ取出
            x = input_train[mb_index]#, :]
            t = target_train[mb_index]

            # 順伝播と逆伝播→更新
            y = model.forward(x, labels=t, train=True)#, dropout=0.3)
            l = loss_func(y, x) 
            l.backtrace()
            
            model.update(eta=0.0003)#, g_clip=2.5)
            acc = cf.get_accuracy(y, x)  
            errors.append(float(l)) 
            accuracy.append(acc)
            
            print(f'{i:6d} {j:6d} {float(l):7.4f} {acc:6.3f}')


    # -- 誤差の記録をグラフ表示 --
    cf.graph_for_error(errors, accuracy)

    index_test = np.arange(100)
    x = input_test[index_test]
    t = target_test[index_test]
    y = model.forward(x, train=False)

    CIFER10.show_multi_samples(y, t, label_list)



