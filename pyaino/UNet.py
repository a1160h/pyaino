from pyaino.Config import *
#set_derivative(True)
from pyaino import Neuron as neuron
from pyaino import LossFunctions as lf
from pyaino import Functions as F
from pyaino import common_function as cf
import warnings

class ConvBlock:
    def __init__(self, in_ch, out_ch, time_embed_dim=None, **kwargs):
        print('__init__', self.__class__.__name__, kwargs)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_embed_dim = time_embed_dim
       
        # Conv 本体（bottleneck 構造）
        self.convs = neuron.Sequential(
            neuron.Conv2dLayer(out_ch, 3, 1, **kwargs),
            neuron.Conv2dLayer(out_ch, 3, 1, **kwargs),
        )

        # time embedding 用 MLP
        if time_embed_dim is None:
            self.mlp = None
        else:
            opt_mlpn = {'batchnorm': kwargs.get('batchnorm'),
                        'layernorm': kwargs.get('layernorm'),
                        'activate':  kwargs.get('activate'),
                        'optimize':  kwargs.get('optimize', 'SGD'),
                        'w_decay':   kwargs.get('w_decay')}
            opt_mlpo = {'optimize':  kwargs.get('optimize', 'SGD'),
                        'w_decay':   kwargs.get('w_decay')}
            self.mlp = neuron.Sequential(
                neuron.NeuronLayer(time_embed_dim, **opt_mlpn),
                neuron.LinearLayer(in_ch, **opt_mlpo),
            )
        self.mlp_used = False      

    def forward(self, x, v=None, train=True):
        self.mlp_used = False
        if self.mlp is None or v is None:
            return self.convs(x, train=train)
        B, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.reshape(B,C,1,1)
        self.mlp_used = True  
        return self.convs(x + v, train=train)
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, eta=0.001, **kwargs):
        #print('### mlp.used =', self.mlp_used)
        self.convs.update(eta=eta, **kwargs)
        if self.mlp_used:
            self.mlp.update(eta=eta, **kwargs)
        
class ConvBlockBottleneck(ConvBlock):
    """
    1x1 Conv を使ったボトルネック型 ConvBlock
    - in_ch  -> mid_ch (1x1)
    - mid_ch -> mid_ch (3x3)
    - mid_ch -> out_ch (1x1)
    """
    def __init__(self, in_ch, out_ch, time_embed_dim=None,
                 bottleneck_ratio=0.5, min_mid_ch=16,
                 **kwargs):
        print('__init__', self.__class__.__name__, in_ch, out_ch,
              time_embed_dim, bottleneck_ratio, min_mid_ch,
              kwargs)
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_embed_dim = time_embed_dim
       
        # mid_ch を下限を設けて決定
        mid_ch = max(int(in_ch * bottleneck_ratio), min_mid_ch)

        # Conv 本体（bottleneck 構造）
        self.convs = neuron.Sequential(
            neuron.Conv2dLayer(mid_ch, 1, 0, **kwargs),  # 1x1: in_ch  -> mid_ch
            neuron.Conv2dLayer(mid_ch, 3, 1, **kwargs),  # 3x3: mid_ch -> mid_ch （padding=1 前提）
            neuron.Conv2dLayer(out_ch, 1, 0, **kwargs),  # 1x1: mid_ch -> out_ch
        )

        # time embedding 用 MLP
        if time_embed_dim is None:
            self.mlp = None
        else:
            opt_mlpn = {'batchnorm': kwargs.get('batchnorm'),
                        'layernorm': kwargs.get('layernorm'),
                        'activate':  kwargs.get('activate'),
                        'optimize':  kwargs.get('optimize', 'SGD'),
                        'w_decay':   kwargs.get('w_decay')}
            opt_mlpo = {'optimize':  kwargs.get('optimize', 'SGD'),
                        'w_decay':   kwargs.get('w_decay')}
            self.mlp = neuron.Sequential(
                neuron.NeuronLayer(time_embed_dim, **opt_mlpn),
                neuron.LinearLayer(in_ch, **opt_mlpo),
            )
        self.mlp_used = False      


class UNet:
    """
    1x1 Conv ボトルネック付き U-Net
    - 完全畳み込み構造なので、(H, W) は 8 の倍数程度なら可変対応
    - 高解像度側の層でも 1x1 bottleneck で軽量化
    """
    def __init__(self, depth=3, in_ch=1, time_embed_dim=None,
                 base_ch=32, bottleneck=True, bottleneck_ratio=0.5, 
                 batchnorm=None, layernorm=None, activate='ReLU', 
                 optimize='AdamT', w_decay=0.01):
        warnings.warn(self.__class__.__name__
                      +"Use this module with 'set_derivative(True)'.")
        
        self.in_ch = in_ch
        self.time_embed_dim = time_embed_dim
        self.depth = depth
        self.optimize = optimize
        self.base_ch = base_ch
        self.bottleneck = bottleneck
        self.bottleneck_ratio = bottleneck_ratio

        # time embedding
        if time_embed_dim is None:
            self.pos_encoding = None
        else:
            self.pos_encoding = neuron.PositionalEncoding(dimension=time_embed_dim)

        if bottleneck:
            Conv = ConvBlockBottleneck
        else:
            Conv = ConvBlock

        options_for_blocks = {'time_embed_dim' : time_embed_dim,
                              'batchnorm' : batchnorm,
                              'layernorm' : layernorm,
                              'activate' : activate,
                              'optimize' : optimize,
                              'w_decay'  : w_decay}
        if bottleneck:
            options_for_blocks['bottleneck_ratio'] = bottleneck_ratio

        options_for_ol     = {'optimize' : optimize,
                              'batchnorm' : batchnorm,
                              'layernorm' : layernorm,
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

    def forward(self, x, timesteps=None, train=True):
        assert x.shape[1] == self.in_ch, f"input C={x.shape[1]} but model in_ch={self.in_ch}"

        # time embedding
        if self.pos_encoding is None or timesteps is None:
            v = None
        else:
            t0 = timesteps
            #t0 -= 1   # Diffuser(1..T) → PosEnc(0..T-1)
            t0 = self.normalize_t(t0, x.shape[0]) # バッチサイズだけ合わせる
            v = self.pos_encoding(t0)
            if (t0 < 0).any() or (t0 >= 1000).any(): # 仮20260107AI
                print('###debug t0', t0)

        zs = []    

        # Down
        for i in range(self.depth):
            z = self.down[i](x, v, train=train)      # H, W
            x  = self.pool[i](z)                     # H/2, W/2
            zs.append(z)

        # Bottleneck
        x  = self.bot(x, v, train=train)       # H/8, W/8

        # Up
        for i in range(self.depth):
            z = zs.pop()
            x = self.upsample[i](x)           # H/4, W/4
            x = self.concat[i](x, z, axis=1)          # C4 + C3
            x = self.up[i](x, v, train=train)    # -> C3

        assert not zs 
        # Output
        y  = self.out(x, train=train)       # (B, in_ch, H, W)
        return y

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, eta=0.001, **kwargs):
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
        returns: t_vec (B,) int64
        """

        # スカラ int の場合
        if isinstance(t, (int,)):
            return np.full((B,), t, dtype=np.int64)

        # numpy/cupy scalar の場合（np.int64(5) 等）
        t_arr = np.asarray(t)
        if getattr(t_arr, "ndim", 0) == 0:
            return np.full((B,), int(t_arr), dtype=np.int64)

        # ベクトルの場合
        if t_arr.shape != (B,):
            raise ValueError(f"t must be scalar or shape (B,), got {t_arr.shape}, B={B}")

        return t_arr.astype(np.int64, copy=False)



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
                 bottleneck=True,
                 )
    loss_func = lf.MeanSquaredError()   # 再構成誤差
    cf.get_obj_info(model)

    # -- 学習データの選択用 --
    errors, accuracy = [], []
    index_rand = np.arange(len(input_train))
    start = time.time()
    # -- 学習と経過の記録 --
    for i in range(epoch):
        np.random.shuffle(index_rand)             # epoch毎index_randをランダマイズ  
        # -- 学習 --
        for j in range(0, len(input_train), batch_size):   # 0～n_trainまでbatch_sizeずつ更新
            mb_index = index_rand[j:j+batch_size] # index_randからbatch_sizeずつ取出
            x = input_train[mb_index, :]

            # 順伝播と逆伝播→更新
            y = model.forward(x)#, train=True, dropout=0.3)
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



