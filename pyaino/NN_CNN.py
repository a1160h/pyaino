# NN_CNN
# 2025.10.16 A.Inoue　
from pyaino.Config import *
from pyaino import Neuron as neuron
from pyaino import LossFunctions #as lf
from pyaino import common_function as cf

class NN_CNN_Base:
    def __init__(self, *args, **kwargs):
        '''
        入出力の指定のみの判別はOutのみで構成できるが、
        NNはInをニューロン数のデフォルト値に使う
        
        '''
        self.title = self.__class__.__name__ 
        print(self.title, args, kwargs)
        # 入出力の判定
        if len(args) == 2:
            In, Out = args
        elif len(args) == 0:
            In = None; Out = None
        else:
            In = None; Out = args[-1]
        print(args, 'number of output =', Out)
        self.layers = []
    
        # 損失関数
        loss_function_name = kwargs.pop('loss', None)
        if loss_function_name is not None:
            self.loss_function = cf.eval_in_module(loss_function_name, LossFunctions)

        self.error_layer = None # デバグ用
        self.outputshape = {}   # デバグ用
        self.last_but_out = kwargs.pop('last_but_out', False) # 最後の層が出力層でなくdropout対象

        return In, Out

        
    def summary(self):
        print('～～ model summary of ' + str(self.__class__.__name__) + '  ～～～～～～～～～～～～～～～～～～～～～')
        for i, layer in enumerate(self.layers):
            post_nl = False
            print('layer', i, layer.__class__.__name__, end=' ')
            if hasattr(layer, 'config') and layer.config is not None:
                print('config =', layer.config)
            if hasattr(layer, 'method'):
                print(' method =', layer.method, end=' ')
                post_nl = True
            if hasattr(layer, 'activator'):
                print(' activate =', layer.activator.__class__.__name__, end=' ')
                post_nl = True
            if getattr(layer, 'DO', None) is not None:
                print('', layer.DO.__class__.__name__, 'applicable', end='')
                post_nl = True
            if getattr(layer, 'Norm', None) is not None and layer.Norm.__class__.__name__!='Identity':
                print('', layer.Norm.__class__.__name__, '= True')
                post_nl = False

            if hasattr(layer, 'optimizer_w'):
                if post_nl:
                    print()
                print(' optimize =', layer.optimizer_w.__class__.__name__, end=' ')
                post_nl = True
                item = layer.optimizer_w
                if item.scheduler is not None:
                    print(' scheduler =', item.scheduler.__class__.__name__, end=' ')
                if item.w_decay!=0:   
                    print(' weight_decay_lambda =', item.w_decay)
                    post_nl=False
                if getattr(item, 'spctrnorm', None) is not None:
                    print('', item.spctrnorm.__class__.__name__,
                          '=', item.spctrnorm.power_iterations, end='')
                    post_nl=True
                if getattr(item, 'wghtclpng', None) is not None:
                    print(' weight_clipping =', item.wghtclpng.__class__.__name__,
                          'clip =', item.wghtclpng.clip, end='')
                    post_nl = True
            if post_nl:
                print()
            print('------------------------------------------------------------------------')
        if hasattr(self, 'loss_function'):
            print('loss_function =', self.loss_function.__class__.__name__)
        print('～～ end of summary ～～～～～～～～～～～～～～～～～～～～～～～～～～～～\n')

    # -- 順伝播 --
    def forward(self, x, t=None, train=False, dropout=0.0, **kwargs): # kwargsは使わない
        y = x
        last_layer = len(self.layers) -1 + self.last_but_out            # 最後の層の番号
        for i, layer in enumerate(self.layers):
            self.error_layer = layer                                    # デバグ用
            dropout_rate = dropout if i<last_layer else 0.0             # 中間層だけが対象
            y = layer.forward(y, train=train, dropout=dropout_rate)
            self.outputshape[str(i)+layer.__class__.__name__] = y.shape # デバグ用
        if t is None:
            return y
        if not hasattr(self, 'loss_function') or y.size!=t.size:
            raise Exception("Can't get loss by forward.")
        if y.shape!=t.shape:
            y = y.reshape(t.shape)
        l = self.loss_function.forward(y, t)
        return y, l
        
    # -- 逆伝播 --
    def backward(self, gy=None, gl=1):
        if gy is not None:
            pass
        elif self.loss_function.t is not None:
            gy = self.loss_function.backward(gl)
        else:
            raise Exception("Can't get gradient for backward." \
                            , 'gy =', gy, 'gl =', gl)
        gx = gy.reshape(len(gy), -1)              # バッチサイズ分拡大 
        for layer in reversed(self.layers):
            self.error_layer = layer
            gx = layer.backward(gx)
        return gx
    
    def loss(self, y, t):
        if hasattr(self, 'loss_function'):
            l = self.loss_function.forward(y, t)
            return l
        else:
            raise Exception('No loss_function defined.')

    # -- 重みとバイアスの更新 --
    def update(self, **kwargs):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(**kwargs)

    def recover(self):
        for layer in self.layers:
            if hasattr(layer, 'recover'):
                layer.recover()

    def backup(self):
        for layer in self.layers:
            if hasattr(layer, 'backup'):
                layer.backup()

    # -- パラメタから辞書 --
    def export_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w'):
                params['layer'+str(i)+'_w'] = np.array(layer.w)
            if hasattr(layer, 'b'):
                params['layer'+str(i)+'_b'] = np.array(layer.b)
        return params

    # -- 辞書からパラメタ --
    def import_params(self, params):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w'):
                layer.w = np.array(params['layer'+str(i)+'_w']) 
            if hasattr(layer, 'b'):
                layer.b = np.array(params['layer'+str(i)+'_b'])

    # -- 勾配から辞書 --　　　
    def export_grads(self):
        grads = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w'):
                grads['layer'+str(i)+'_w'] = copy.deepcopy(layer.grad_w)
            if hasattr(layer, 'b'):
                grads['layer'+str(i)+'_b'] = copy.deepcopy(layer.grad_b)
        return grads


    # -- 学習結果の保存 --
    def save_parameters(self, file_name):
        title = self.__class__.__name__
        cf.save_parameters_cpb(file_name, title, self.export_params())

    # -- 学習結果の継承 --
    def load_parameters(self, file_name):
        title_f, params = cf.load_parameters_cpb(file_name)
        title = self.__class__.__name__
        if title == title_f:
            self.import_params(params)
            print('パラメータが継承されました')
        else:
            print('!!構成が一致しないためパラメータは継承されません!!')
        return params

    # -- 計測用 --
    def mesurement(self, x, t, n=None, mchx=False):
        return cf.mesurement(self.forward, self.loss_function, x, t, n, mchx)


# -- 数値微分による勾配算出 --
def gradient(x, t):
    params = model.export_params()           # 現状パラメタ全体を受け取る　　　
    grads = {}
    for key in params.keys(): 
        def loss_of_param(z):                # 数値微分を呼出すための関数を定義　
            params[key] = z                  # 注目パラメタを z で置換(z:仮の引数)
            model.import_params(params)      # z で置換したパラメタを有効化
            y = model.forward(x)             # 上記条件で順伝播
            return model.loss(y, t) 
        grads[key] = cf.numerical_gradient(loss_of_param, params[key])
    return grads

# -- 勾配チェック -- 
def gradient_check(grad1, grad2):
    return cf.gradient_check(grad1, grad2)

# -- サンプルの提示と順伝播の結果のテスト -- 
def test_sample(image_show, x, t, label_list=None, label_list2=None):
    cf.test_sample(image_show, model.forward, x, t, label_list, label_list2)


#### 個別のNN #####
class NN_0(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        In, Out = super().__init__(*args, **kwargs)
        self.layer = neuron.NeuronLayer(Out, **kwargs)
        self.layers.append(self.layer)
    
class NN_1(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        In, Out = super().__init__(*args, **kwargs)
        ml_nn = kwargs.pop('ml_nn', 3)
        self.middle_layer = neuron.NeuronLayer(ml_nn, **kwargs)
        self.output_layer = neuron.NeuronLayer(Out, **kwargs)
        self.layers.append(self.middle_layer)    
        self.layers.append(self.output_layer)

class NN_m(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
        中間層＋出力層
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 全結合中間層 
        ml_nn_cand = In + 2 if In is not None else 100
        ml_nn = kwargs.pop('ml_nn', ml_nn_cand)                     # ニューロン数
        opt_for_ml = {}
        opt_for_ml['activate']  = kwargs.pop('ml_act', 'Sigmoid')   # 活性化関数
        opt_for_ml['optimize']  = kwargs.pop('ml_opt', 'Adam')      # 最適化関数
        opt_for_ml['batchnorm'] = kwargs.pop('bn',      False)      # 出力層では bn 不要 
        opt_for_ml['layernorm'] = kwargs.pop('ln',      False)      # 出力層では ln 不要 
        opt_for_ml['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'Identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['optimize']  = kwargs.pop('ol_opt', 'SGD')       # 最適化関数
        opt_for_ol['spctrnorm'] = kwargs.pop('sn',      None)
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_ml.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer  = neuron.NeuronLayer(ml_nn, **opt_for_ml)
        # layer 2
        self.output_layer  = neuron.NeuronLayer(Out,   **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)    
        self.layers.append(self.output_layer)

    def forward(self, x, t=None, train=False, dropout=0.0):
        x = x.reshape(len(x), -1)
        return super().forward(x, t, train=train, dropout=dropout)

class NN_mm(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
        中間層＋中間層＋出力層
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 全結合中間層 
        ml1_nn_cand = In + 2 if In is not None else 100
        ml1_nn = kwargs.pop('ml1_nn', ml1_nn_cand)                  # ニューロン数
        opt_for_ml1 = {}
        opt_for_ml1['activate']  = kwargs.pop('ml1_act', 'Sigmoid') # 活性化関数
        opt_for_ml1['optimize']  = kwargs.pop('ml1_opt', 'Adam')    # 最適化関数
        opt_for_ml1['dropout']   = kwargs.get('dropout', True)      # ドロップアウト
        opt_for_ml1['batchnorm'] = kwargs.get('bn',      False)     # 次層でも使う 
        opt_for_ml1['layernorm'] = kwargs.get('ln',      False)     # 次層でも使う
        opt_for_ml1['spctrnorm'] = kwargs.get('sn',      None)      # 次層でも使う 
        # layer2 全結合中間層 
        ml2_nn_cand = In + 2 if In is not None else 100
        ml2_nn = kwargs.pop('ml2_nn', ml2_nn_cand)                  # ニューロン数
        opt_for_ml2 = {}
        opt_for_ml2['activate']  = kwargs.pop('ml2_act', 'Sigmoid') # 活性化関数
        opt_for_ml2['optimize']  = kwargs.pop('ml2_opt', 'Adam')    # 最適化関数
        opt_for_ml2['dropout']   = kwargs.pop('dropout', True)      # ドロップアウト
        opt_for_ml2['batchnorm'] = kwargs.pop('bn',      False)     # 出力層では bn 不要 
        opt_for_ml2['layernorm'] = kwargs.pop('ln',      False)     # 出力層では bn 不要 
        opt_for_ml2['spctrnorm'] = kwargs.get('sn',      None)
        # layer3 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'Identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['optimize']  = kwargs.pop('ol_opt',   'SGD')     # 最適化関数
        opt_for_ol['spctrnorm'] = kwargs.pop('sn',       None)
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_ml1.update(kwargs)
        opt_for_ml2.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer1 = neuron.NeuronLayer(ml1_nn, **opt_for_ml1)
        # layer 2
        self.middle_layer2 = neuron.NeuronLayer(ml2_nn, **opt_for_ml2)
        # layer 3
        self.output_layer  = neuron.NeuronLayer(Out,    **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer1)    
        self.layers.append(self.middle_layer2)    
        self.layers.append(self.output_layer)

    def forward(self, x, t=None, train=False, dropout=0.0):
        x = x.reshape(len(x), -1)
        return super().forward(x, t, train=train, dropout=dropout)

class NN_m2(NN_mm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


#### 個別のCNN #####
  
class CNN_cpm(NN_CNN_Base): #ニューラルネットワーク　畳込み層＋プーリング層＋全結合層（中間層＋出力層）
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層＋プーリング層
        ＋全結合層(中間層＋出力層)
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層
        M      = kwargs.pop('M',     24)            # フィルタ数
        kernel_size = kwargs.pop('kernel_size', 3)  # フィルタ高
        stride = kwargs.pop('stride', 1)            # ストライド
        cl_pad = kwargs.pop('cl_pad', 0)            # パディング
        opt_for_cl = {}
        opt_for_cl['activate']  = kwargs.pop('cl_act', 'ReLU')    # 活性化関数 
        opt_for_cl['optimize']  = kwargs.pop('cl_opt', 'Adam')    # 最適化関数
        opt_for_cl['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 プーリング層
        opt_for_pl = {}
        pool   = kwargs.pop('pool',   2)            # プーリング
        pl_pad = kwargs.pop('pl_pad', 0)            # パディング
        method = kwargs.pop('method', 'max')        # プーリングメソッド 
        opt_for_pl['dropout']   = kwargs.get('dropout', True)     # ドロップアウト
        # layer3 全結合中間層 
        ml_nn  = kwargs.pop('ml_nn',  200)          # ニューロン数
        opt_for_ml = {}
        opt_for_ml['activate']  = kwargs.pop('ml_act', 'ReLU')    # 活性化関数
        opt_for_ml['optimize']  = kwargs.pop('ml_opt', 'Adam')    # 最適化関数
        opt_for_ml['dropout']   = kwargs.pop('dropout', True)     # ドロップアウト
        opt_for_ml['batchnorm'] = kwargs.pop('bn',      False)    # 出力層では bn 不要 
        opt_for_ml['layernorm'] = kwargs.pop('ln',      False)    # 出力層では ln 不要 
        opt_for_ml['spctrnorm'] = kwargs.get('sn',      None)
        opt_for_ml['full_connection'] = True         
        # layer4 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['optimize']  = kwargs.pop('ol_opt', 'SGD')     # 最適化関数
        opt_for_ol['spctrnorm'] = kwargs.pop('sn',      None)
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl.update(kwargs) 
        opt_for_ml.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        # layer 1
        self.conv_layer    = neuron.Conv2dLayer(M, kernel_size, stride, cl_pad, **opt_for_cl)
        # layer 2
        self.pooling_layer = neuron.Pooling2dLayer(pool, pl_pad, method, **opt_for_pl)
        # layer 3
        self.middle_layer  = neuron.NeuronLayer(ml_nn, **opt_for_ml)
        # layer 4
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer)
        self.layers.append(self.pooling_layer)
        self.layers.append(self.middle_layer)    
        self.layers.append(self.output_layer)

class CNN_ccpm(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×２＋プーリング層
        ＋全結合層(中間層＋出力層)
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層1
        M1      = kwargs.pop('M1',      24)          # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2) # フィルタ高
        stride1 = kwargs.pop('stride1',  1)          # ストライド
        cl1_pad = kwargs.pop('cl1_pad',  0)          # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')    # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')    # 最適化関数
        opt_for_cl1['dropout']   = kwargs.get('dropout', True)      # ドロップアウト
        opt_for_cl1['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl1['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 畳込み層2
        M2      = kwargs.pop('M2',      24)          # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2) # フィルタ高
        stride2 = kwargs.pop('stride2',  1)          # ストライド
        cl2_pad = kwargs.pop('cl2_pad',  0)          # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')    # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')    # 最適化関数
        opt_for_cl2['batchnorm'] = kwargs.get('bn',      False)     # bn は中間層でも使う
        opt_for_cl2['layernorm'] = kwargs.get('ln',      False)     # ln は中間層でも使う
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',      None)
        # layer3 プーリング層
        opt_for_pl = {}
        pool   = kwargs.pop('pool',      2)          # プーリング
        pl_pad = kwargs.pop('pl_pad',    0)          # パディング
        method = kwargs.pop('method', 'max')         # プーリングメソッド 
        opt_for_pl['dropout']    = kwargs.get('dropout', True)      # ドロップアウト
        # layer4 全結合中間層 
        ml_nn  = kwargs.pop('ml_nn',  200)           # ニューロン数
        opt_for_ml = {}
        opt_for_ml['activate']   = kwargs.pop('ml_act', 'ReLU')     # 活性化関数
        opt_for_ml['optimize']   = kwargs.pop('ml_opt', 'Adam')     # 最適化関数
        opt_for_ml['dropout']    = kwargs.pop('dropout', True)      # ドロップアウト
        opt_for_ml['batchnorm']  = kwargs.pop('bn',      False)     # bn は中間層でも使う
        opt_for_ml['layernorm']  = kwargs.pop('ln',      False)     # ln は中間層でも使う
        opt_for_ml['spctrnorm']  = kwargs.get('sn',      None)
        opt_for_ml['full_connection'] = True         
        # layer5 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']   = kwargs.pop('ol_act', ol_act_cand)# 活性化関数
        opt_for_ol['optimize']   = kwargs.pop('ol_opt', 'SGD')      # 最適化関数
        opt_for_ol['spctrnorm']  = kwargs.pop('sn',      None)
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_ml.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        # layer 1
        self.conv_layer1   = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        # layer 2
        self.conv_layer2   = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        # layer 3
        self.pooling_layer = neuron.Pooling2dLayer(pool, pl_pad, method, **opt_for_pl)
        # layer 4
        self.middle_layer  = neuron.NeuronLayer(ml_nn, **opt_for_ml)
        # layer 5
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.pooling_layer)
        self.layers.append(self.middle_layer)    
        self.layers.append(self.output_layer)


class CNN_cccpm(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×３＋プーリング層
        ＋全結合層(中間層＋出力層)
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層1
        M1      = kwargs.pop('M1',      16)          # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2) # フィルタ高
        stride1 = kwargs.pop('stride1',  1)          # ストライド
        cl1_pad = kwargs.pop('cl1_pad',  0)          # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')    # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')    # 最適化関数
        opt_for_cl1['dropout']   = kwargs.get('dropout', True)      # ドロップアウト
        opt_for_cl1['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl1['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 畳込み層2
        M2      = kwargs.pop('M2',      16)          # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2) # フィルタ高
        stride2 = kwargs.pop('stride2',  1)          # ストライド
        cl2_pad = kwargs.pop('cl2_pad',  0)          # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')    # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')    # 最適化関数
        opt_for_cl2['dropout']   = kwargs.get('dropout', True)      # ドロップアウト
        opt_for_cl2['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl2['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',      None)
        # layer3 畳込み層3
        M3      = kwargs.pop('M3',      16)          # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2) # フィルタ高
        stride3 = kwargs.pop('stride3',  1)          # ストライド
        cl3_pad = kwargs.pop('cl3_pad',  0)          # パディング
        opt_for_cl3 = {}
        opt_for_cl3['activate']  = kwargs.pop('cl3_act', 'ReLU')    # 活性化関数 
        opt_for_cl3['optimize']  = kwargs.pop('cl3_opt', 'Adam')    # 最適化関数
        opt_for_cl3['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl3['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl3['spctrnorm'] = kwargs.get('sn',      None)
        # layer4 プーリング層
        opt_for_pl = {}
        pool   = kwargs.pop('pool',      2)         # プーリング
        pl_pad = kwargs.pop('pl_pad',    0)         # パディング
        method = kwargs.pop('method', 'max')        # プーリングメソッド 
        opt_for_pl['dropout']    = kwargs.get('dropout', True)      # ドロップアウト
        # layer5 全結合中間層 
        ml_nn  = kwargs.pop('ml_nn',  200)          # ニューロン数
        opt_for_ml = {}
        opt_for_ml['activate']   = kwargs.pop('ml_act', 'ReLU')    # 活性化関数
        opt_for_ml['optimize']   = kwargs.pop('ml_opt', 'Adam')    # 最適化関数
        opt_for_ml['dropout']    = kwargs.pop('dropout', True)      # ドロップアウト
        opt_for_ml['batchnorm']  = kwargs.pop('bn',      False)    
        opt_for_ml['layernorm']  = kwargs.pop('ln',      False)    
        opt_for_ml['spctrnorm']  = kwargs.get('sn',      None)
        opt_for_ml['full_connection'] = True         
        # layer6 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']   = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['optimize']   = kwargs.pop('ol_opt', 'SGD')     # 最適化関数
        opt_for_ol['spctrnorm']  = kwargs.pop('sn',      None)
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_cl3.update(kwargs) 
        opt_for_ml.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        # layer 1
        self.conv_layer1   = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        # layer 2
        self.conv_layer2   = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        # layer 3
        self.conv_layer3   = neuron.Conv2dLayer(M3, kernel_size3, stride3, cl3_pad, **opt_for_cl3)
        # layer 4
        self.pooling_layer = neuron.Pooling2dLayer(pool, pl_pad, method, **opt_for_pl)
        # layer 5
        self.middle_layer  = neuron.NeuronLayer(ml_nn, **opt_for_ml)
        # layer 6
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.conv_layer3)
        self.layers.append(self.pooling_layer)
        self.layers.append(self.middle_layer)    
        self.layers.append(self.output_layer)


class CNN_ccpccpm(NN_CNN_Base): #ニューラルネットワーク　畳込み層×２＋プーリング層＋畳込み層×２＋プーリング層＋畳込み層×２＋プーリング層＋全結合層(中間層＋出力層)
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×２＋プーリング層
        ＋畳込み層×２＋プーリング層
        ＋全結合層(中間層＋出力層)
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層1
        M1      = kwargs.pop('M1',     24)            # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2)  # フィルタ高
        stride1 = kwargs.pop('stride1', 1)            # ストライド
        cl1_pad = kwargs.pop('cl1_pad', 0)            # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')    # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')    # 最適化関数
        opt_for_cl1['dropout']   = kwargs.get('dropout', True)      # ドロップアウト
        opt_for_cl1['batchnorm'] = kwargs.get('bn',      False)    
        opt_for_cl1['layernorm'] = kwargs.get('ln',      False)    
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 畳込み層2
        M2      = kwargs.pop('M2',     24)            # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2)  # フィルタ高
        stride2 = kwargs.pop('stride2', 1)            # ストライド
        cl2_pad = kwargs.pop('cl2_pad', 0)            # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')    # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')    # 最適化関数
        opt_for_cl2['batchnorm'] = kwargs.get('bn',      False)    
        opt_for_cl2['layernorm'] = kwargs.get('ln',      False)    
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',      None)
        # layer3 プーリング層1
        opt_for_pl1 = {}
        pool1   = kwargs.pop('pool1',   2)            # プーリング
        pl1_pad = kwargs.pop('pl1_pad', 0)            # パディング
        method1 = kwargs.get('method', 'max')         # プーリングメソッド
        opt_for_pl1['dropout']   = kwargs.get('dropout', True)      # ドロップアウト
        # layer4 畳込み層3
        M3      = kwargs.pop('M3',     32)            # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2)  # フィルタ高
        stride3 = kwargs.pop('stride3', 1)            # ストライド
        cl3_pad = kwargs.pop('cl3_pad', 0)            # パディング
        opt_for_cl3 = {}
        opt_for_cl3['activate']  = kwargs.pop('cl3_act', 'ReLU')   # 活性化関数 
        opt_for_cl3['optimize']  = kwargs.pop('cl3_opt', 'Adam')   # 最適化関数
        opt_for_cl3['dropout']   = kwargs.get('dropout', True)      # ドロップアウト
        opt_for_cl3['batchnorm'] = kwargs.get('bn',      False)    
        opt_for_cl3['layernorm'] = kwargs.get('ln',      False)    
        opt_for_cl3['spctrnorm'] = kwargs.get('sn',      None)
        # layer5 畳込み層4
        M4      = kwargs.pop('M4',     32)            # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 2)  # フィルタ高
        stride4 = kwargs.pop('stride4', 1)            # ストライド
        cl4_pad = kwargs.pop('cl4_pad', 0)            # パディング
        opt_for_cl4 = {}
        opt_for_cl4['activate']  = kwargs.pop('cl4_act', 'ReLU')   # 活性化関数 
        opt_for_cl4['optimize']  = kwargs.pop('cl4_opt', 'Adam')   # 最適化関数
        opt_for_cl4['batchnorm'] = kwargs.get('bn',      False)    
        opt_for_cl4['layernorm'] = kwargs.get('ln',      False)    
        opt_for_cl4['spctrnorm'] = kwargs.get('sn',      None)
        # layer6 プーリング層2
        opt_for_pl2 = {}
        pool2   = kwargs.pop('pool2',   2)            # プーリング
        pl2_pad = kwargs.pop('pl2_pad', 0)            # パディング
        method2 = kwargs.pop('method', 'max')         # プーリングメソッド
        opt_for_pl2['dropout']   = kwargs.get('dropout', True)      # ドロップアウト
        # layer7 全結合中間層 
        ml_nn  = kwargs.pop('ml_nn',   200)           # ニューロン数
        opt_for_ml = {}
        opt_for_ml['activate']   = kwargs.pop('ml_act', 'ReLU')    #活性化関数
        opt_for_ml['optimize']   = kwargs.pop('ml_opt', 'Adam')    #最適化関数
        opt_for_ml['dropout']    = kwargs.pop('dropout', True)      # ドロップアウト
        opt_for_ml['batchnorm']  = kwargs.pop('bn',      False)    
        opt_for_ml['layernorm']  = kwargs.pop('ln',      False)    
        opt_for_ml['spctrnorm']  = kwargs.get('sn',      None)
        opt_for_ml['full_connection'] = True         
        # layer8 全結合出力層
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']   = kwargs.pop('ol_act', ol_act_cand) #活性化関数
        opt_for_ol['optimize']   = kwargs.pop('ol_opt', 'SGD')     #最適化関数
        opt_for_ol['spctrnorm']  = kwargs.pop('sn',      None)

        # kwargs に残ったものを結合
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_cl3.update(kwargs) 
        opt_for_cl4.update(kwargs) 
        opt_for_ml.update(kwargs)  
        opt_for_ol.update(kwargs)  

        # -- 各層の初期化 -- 
        self.conv_layer1    = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        self.conv_layer2    = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        self.pooling_layer1 = neuron.Pooling2dLayer(pool1, pl1_pad, method1, **opt_for_pl1)
        self.conv_layer3    = neuron.Conv2dLayer(M3, kernel_size3, stride3, cl3_pad, **opt_for_cl3)
        self.conv_layer4    = neuron.Conv2dLayer(M4, kernel_size4, stride4, cl4_pad, **opt_for_cl4)
        self.pooling_layer2 = neuron.Pooling2dLayer(pool2, pl2_pad, method2, **opt_for_pl2)
        self.middle_layer   = neuron.NeuronLayer(ml_nn, **opt_for_ml)
        self.output_layer   = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.pooling_layer1)
        self.layers.append(self.conv_layer3)
        self.layers.append(self.conv_layer4)
        self.layers.append(self.pooling_layer2)
        self.layers.append(self.middle_layer)    
        self.layers.append(self.output_layer)


class CNN_ccpccpccpm(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×２＋プーリング層
        ＋畳込み層×２＋プーリング層
        ＋畳込み層×２＋プーリング層
        ＋全結合層(中間層＋出力層)
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層1
        M1      = kwargs.pop('M1',     16)            # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2)  # フィルタ高
        stride1 = kwargs.pop('stride1', 1)            # ストライド
        cl1_pad = kwargs.pop('cl1_pad', 0)            # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')    # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')    # 最適化関数
        opt_for_cl1['dropout']   = kwargs.get('dropout',   True)    # ドロップアウト
        opt_for_cl1['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl1['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',        None)
        # layer2 畳込み層2
        M2      = kwargs.pop('M2',     16)            # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2)  # フィルタ高
        stride2 = kwargs.pop('stride2', 1)            # ストライド
        cl2_pad = kwargs.pop('cl2_pad', 0)            # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')   # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')   # 最適化関数
        opt_for_cl2['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl2['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',        None)
        # layer3 プーリング層1
        opt_for_pl1 = {}
        pool1   = kwargs.pop('pool1',   2)            # プーリング
        pl1_pad = kwargs.pop('pl1_pad', 0)            # パディング
        method1 = kwargs.get('method', 'max')         # プーリングメソッド
        opt_for_pl1['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        # layer4 畳込み層3
        M3      = kwargs.pop('M3',     32)            # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2)  # フィルタ高
        stride3 = kwargs.pop('stride3', 1)            # ストライド
        cl3_pad = kwargs.pop('cl3_pad', 0)            # パディング
        opt_for_cl3 = {}
        opt_for_cl3['activate']  = kwargs.pop('cl3_act', 'ReLU')   # 活性化関数 
        opt_for_cl3['optimize']  = kwargs.pop('cl3_opt', 'Adam')   # 最適化関数
        opt_for_cl3['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        opt_for_cl3['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl3['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl3['spctrnorm'] = kwargs.get('sn',        None)
        # layer5 畳込み層4
        M4      = kwargs.pop('M4',     32)            # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 2)  # フィルタ高
        stride4 = kwargs.pop('stride4', 1)            # ストライド
        cl4_pad = kwargs.pop('cl4_pad', 0)            # パディング
        opt_for_cl4 = {}
        opt_for_cl4['activate']  = kwargs.pop('cl4_act', 'ReLU')   # 活性化関数 
        opt_for_cl4['optimize']  = kwargs.pop('cl4_opt', 'Adam')   # 最適化関数
        opt_for_cl4['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl4['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl4['spctrnorm'] = kwargs.get('sn',        None)
        # layer6 プーリング層2
        opt_for_pl2 = {}
        pool2   = kwargs.pop('pool2',   2)            # プーリング
        pl2_pad = kwargs.pop('pl2_pad', 0)            # パディング
        method2 = kwargs.get('method', 'max')         # プーリングメソッド
        opt_for_pl2['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        # layer7 畳込み層5
        M5      = kwargs.pop('M5',     64)            # フィルタ数
        kernel_size5 = kwargs.pop('kernel_size5', 2)  # フィルタ高
        stride5 = kwargs.pop('stride5', 1)            # ストライド
        cl5_pad = kwargs.pop('cl5_pad', 0)            # パディング
        opt_for_cl5 = {}
        opt_for_cl5['activate']  = kwargs.pop('cl5_act', 'ReLU')   # 活性化関数 
        opt_for_cl5['optimize']  = kwargs.pop('cl5_opt', 'Adam')   # 最適化関数
        opt_for_cl5['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        opt_for_cl5['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl5['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl5['spctrnorm'] = kwargs.get('sn',        None)
        # layer8 畳込み層6
        M6      = kwargs.pop('M6',     64)            # フィルタ数
        kernel_size6 = kwargs.pop('kernel_size6', 2)  # フィルタ高
        stride6 = kwargs.pop('stride6', 1)            # ストライド
        cl6_pad = kwargs.pop('cl6_pad', 0)            # パディング
        opt_for_cl6 = {}
        opt_for_cl6['activate']  = kwargs.pop('cl6_act', 'ReLU')   # 活性化関数 
        opt_for_cl6['optimize']  = kwargs.pop('cl6_opt', 'Adam')   # 最適化関数
        opt_for_cl6['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl6['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl6['spctrnorm'] = kwargs.get('sn',        None)
        # layer9 プーリング層3
        opt_for_pl3 = {}
        pool3   = kwargs.pop('pool3',   2)            # プーリング
        pl3_pad = kwargs.pop('pl3_pad', 0)            # パディング
        method3 = kwargs.pop('method', 'max')         # プーリングメソッド
        opt_for_pl3['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        # layer10 全結合中間層 
        ml_nn  = kwargs.pop('ml_nn',   200)           # ニューロン数
        opt_for_ml = {}
        opt_for_ml['activate']   = kwargs.pop('ml_act',  'ReLU')   #活性化関数
        opt_for_ml['optimize']   = kwargs.pop('ml_opt',  'Adam')   #最適化関数
        opt_for_ml['dropout']    = kwargs.pop('dropout',   True)   # ドロップアウト
        opt_for_ml['batchnorm']  = kwargs.pop('bn',       False)    
        opt_for_ml['layernorm']  = kwargs.pop('ln',       False)    
        opt_for_ml['spctrnorm']  = kwargs.get('sn',        None)
        opt_for_ml['full_connection'] = True         
        # layer11 全結合出力層
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']   = kwargs.pop('ol_act', ol_act_cand) #活性化関数
        opt_for_ol['optimize']   = kwargs.pop('ol_opt',   'SGD')   #最適化関数
        opt_for_ol['spctrnorm']  = kwargs.pop('sn',        None)
        # kwargs に残ったものを結合
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_cl3.update(kwargs) 
        opt_for_cl4.update(kwargs) 
        opt_for_cl5.update(kwargs) 
        opt_for_cl6.update(kwargs) 
        opt_for_ml.update(kwargs)  
        opt_for_ol.update(kwargs)  

        # -- 各層の初期化 -- 
        self.conv_layer1    = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        self.conv_layer2    = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        self.pooling_layer1 = neuron.Pooling2dLayer(pool1, pl1_pad, method1, **opt_for_pl1)
        self.conv_layer3    = neuron.Conv2dLayer(M3, kernel_size3, stride3, cl3_pad, **opt_for_cl3)
        self.conv_layer4    = neuron.Conv2dLayer(M4, kernel_size4, stride4, cl4_pad, **opt_for_cl4)
        self.pooling_layer2 = neuron.Pooling2dLayer(pool2, pl2_pad, method2, **opt_for_pl2)
        self.conv_layer5    = neuron.Conv2dLayer(M5, kernel_size5, stride5, cl5_pad, **opt_for_cl5)
        self.conv_layer6    = neuron.Conv2dLayer(M6, kernel_size6, stride6, cl6_pad, **opt_for_cl6)
        self.pooling_layer3 = neuron.Pooling2dLayer(pool3, pl3_pad, method3, **opt_for_pl3)
        self.middle_layer   = neuron.NeuronLayer(ml_nn, **opt_for_ml)
        self.output_layer   = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.pooling_layer1)
        self.layers.append(self.conv_layer3)
        self.layers.append(self.conv_layer4)
        self.layers.append(self.pooling_layer2)
        self.layers.append(self.conv_layer5)
        self.layers.append(self.conv_layer6)
        self.layers.append(self.pooling_layer3)
        self.layers.append(self.middle_layer)    
        self.layers.append(self.output_layer)


class CNN_ccpccpccgp(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×２＋プーリング層
        ＋畳込み層×２＋プーリング層
        ＋畳込み層×２
        ＋プーリング層(global average pooling)
        ＋全結合層(中間層＋出力層)
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer0 畳込み層1
        M1      = kwargs.pop('M1',     32)#16)            # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2)  # フィルタ高
        stride1 = kwargs.pop('stride1', 1)            # ストライド
        cl1_pad = kwargs.pop('cl1_pad', 0)            # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')   # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')   # 最適化関数
        #opt_for_cl1['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        opt_for_cl1['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl1['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',        None)
        # layer1 畳込み層2
        M2      = kwargs.pop('M2',     32)#16)            # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2)  # フィルタ高
        stride2 = kwargs.pop('stride2', 1)            # ストライド
        cl2_pad = kwargs.pop('cl2_pad', 0)            # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')   # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')   # 最適化関数
        opt_for_cl2['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl2['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',        None)
        # layer2 プーリング層1
        opt_for_pl1 = {}
        pool1   = kwargs.pop('pool1',   2)            # プーリング
        pl1_pad = kwargs.pop('pl1_pad', 0)            # パディング
        method1 = kwargs.get('method', 'max')         # プーリングメソッド
        opt_for_pl1['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        # layer3 畳込み層3
        M3      = kwargs.pop('M3',     64)#32)            # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2)  # フィルタ高
        stride3 = kwargs.pop('stride3', 1)            # ストライド
        cl3_pad = kwargs.pop('cl3_pad', 0)            # パディング
        opt_for_cl3 = {}
        opt_for_cl3['activate']  = kwargs.pop('cl3_act', 'ReLU')   # 活性化関数 
        opt_for_cl3['optimize']  = kwargs.pop('cl3_opt', 'Adam')   # 最適化関数
        #opt_for_cl3['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        opt_for_cl3['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl3['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl3['spctrnorm'] = kwargs.get('sn',        None)
        # layer4 畳込み層4
        M4      = kwargs.pop('M4',     64)#32)            # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 2)  # フィルタ高
        stride4 = kwargs.pop('stride4', 1)            # ストライド
        cl4_pad = kwargs.pop('cl4_pad', 0)            # パディング
        opt_for_cl4 = {}
        opt_for_cl4['activate']  = kwargs.pop('cl4_act', 'ReLU')   # 活性化関数 
        opt_for_cl4['optimize']  = kwargs.pop('cl4_opt', 'Adam')   # 最適化関数
        opt_for_cl4['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl4['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl4['spctrnorm'] = kwargs.get('sn',        None)
        # layer5 プーリング層2
        opt_for_pl2 = {}
        pool2   = kwargs.pop('pool2',   2)            # プーリング
        pl2_pad = kwargs.pop('pl2_pad', 0)            # パディング
        method2 = kwargs.pop('method', 'max')         # プーリングメソッド
        opt_for_pl2['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        # layer6 畳込み層5
        M5      = kwargs.pop('M5',    128)#64)            # フィルタ数
        kernel_size5 = kwargs.pop('kernel_size5', 2)  # フィルタ高
        stride5 = kwargs.pop('stride5', 1)            # ストライド
        cl5_pad = kwargs.pop('cl5_pad', 0)            # パディング
        opt_for_cl5 = {}
        opt_for_cl5['activate']  = kwargs.pop('cl5_act', 'ReLU')   # 活性化関数 
        opt_for_cl5['optimize']  = kwargs.pop('cl5_opt', 'Adam')   # 最適化関数
        #opt_for_cl5['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        opt_for_cl5['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl5['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl5['spctrnorm'] = kwargs.get('sn',        None)
        # layer7 畳込み層6
        M6      = kwargs.pop('M6',    128)#64)            # フィルタ数
        kernel_size6 = kwargs.pop('kernel_size6', 2)  # フィルタ高
        stride6 = kwargs.pop('stride6', 1)            # ストライド
        cl6_pad = kwargs.pop('cl6_pad', 0)            # パディング
        opt_for_cl6 = {}
        opt_for_cl6['activate']  = kwargs.pop('cl6_act', 'ReLU')   # 活性化関数 
        opt_for_cl6['optimize']  = kwargs.pop('cl6_opt', 'Adam')   # 最適化関数
        opt_for_cl6['batchnorm'] = kwargs.pop('bn',       False)    
        opt_for_cl6['layernorm'] = kwargs.pop('ln',       False)    
        opt_for_cl6['spctrnorm'] = kwargs.get('sn',        None)
        # layer8 グローバルプーリング層
        opt_for_gpl = {}
        opt_for_gpl['dropout']   = kwargs.pop('dropout',   True)   # ドロップアウト
        # layer9 全結合出力層
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']   = kwargs.pop('ol_act', ol_act_cand) #活性化関数
        opt_for_ol['optimize']   = kwargs.pop('ol_opt',   'SGD')   #最適化関数
        opt_for_ol['spctrnorm']  = kwargs.pop('sn',        None)
        # kwargs に残ったものを結合
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_cl3.update(kwargs) 
        opt_for_cl4.update(kwargs) 
        opt_for_cl5.update(kwargs) 
        opt_for_cl6.update(kwargs) 
        opt_for_ol.update(kwargs)  

        # -- 各層の初期化 -- 
        self.conv_layer1    = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        self.conv_layer2    = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        self.pooling_layer1 = neuron.Pooling2dLayer(pool1, pl1_pad, method1, **opt_for_pl1)
        self.conv_layer3    = neuron.Conv2dLayer(M3, kernel_size3, stride3, cl3_pad, **opt_for_cl3)
        self.conv_layer4    = neuron.Conv2dLayer(M4, kernel_size4, stride4, cl4_pad, **opt_for_cl4)
        self.pooling_layer2 = neuron.Pooling2dLayer(pool2, pl2_pad, method2, **opt_for_pl2)
        self.conv_layer5    = neuron.Conv2dLayer(M5, kernel_size5, stride5, cl5_pad, **opt_for_cl5)
        self.conv_layer6    = neuron.Conv2dLayer(M6, kernel_size6, stride6, cl6_pad, **opt_for_cl6)
        self.global_pooling = neuron.GlobalAveragePooling(**opt_for_gpl)
        self.output_layer   = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.pooling_layer1)
        self.layers.append(self.conv_layer3)
        self.layers.append(self.conv_layer4)
        self.layers.append(self.pooling_layer2)
        self.layers.append(self.conv_layer5)
        self.layers.append(self.conv_layer6)
        self.layers.append(self.global_pooling)
        self.layers.append(self.output_layer)


class CNN_ccpccpccpccgp(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×２＋プーリング層
        ＋畳込み層×２＋プーリング層
        ＋畳込み層×２＋プーリング層
        ＋畳込み層×２
        ＋プーリング層(global_averave pooling)
        ＋全結合層(中間層＋出力層)
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer0 畳込み層1
        M1      = kwargs.pop('M1',     32)#16)            # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2)  # フィルタ高
        stride1 = kwargs.pop('stride1', 1)            # ストライド
        cl1_pad = kwargs.pop('cl1_pad', 0)            # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')   # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')   # 最適化関数
        #opt_for_cl1['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        opt_for_cl1['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl1['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',        None)
        # layer1 畳込み層2
        M2      = kwargs.pop('M2',     32)#16)            # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2)  # フィルタ高
        stride2 = kwargs.pop('stride2', 1)            # ストライド
        cl2_pad = kwargs.pop('cl2_pad', 0)            # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')   # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')   # 最適化関数
        opt_for_cl2['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl2['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',        None)
        # layer2 プーリング層1
        opt_for_pl1 = {}
        pool1   = kwargs.pop('pool1',   2)            # プーリング
        pl1_pad = kwargs.pop('pl1_pad', 0)            # パディング
        method1 = kwargs.get('method', 'max')         # プーリングメソッド
        opt_for_pl1['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        # layer3 畳込み層3
        M3      = kwargs.pop('M3',     64)#32)            # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2)  # フィルタ高
        stride3 = kwargs.pop('stride3', 1)            # ストライド
        cl3_pad = kwargs.pop('cl3_pad', 0)            # パディング
        opt_for_cl3 = {}
        opt_for_cl3['activate']  = kwargs.pop('cl3_act', 'ReLU')   # 活性化関数 
        opt_for_cl3['optimize']  = kwargs.pop('cl3_opt', 'Adam')   # 最適化関数
        #opt_for_cl3['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        opt_for_cl3['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl3['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl3['spctrnorm'] = kwargs.get('sn',        None)
        # layer4 畳込み層4
        M4      = kwargs.pop('M4',     64)#32)            # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 2)  # フィルタ高
        stride4 = kwargs.pop('stride4', 1)            # ストライド
        cl4_pad = kwargs.pop('cl4_pad', 0)            # パディング
        opt_for_cl4 = {}
        opt_for_cl4['activate']  = kwargs.pop('cl4_act', 'ReLU')   # 活性化関数 
        opt_for_cl4['optimize']  = kwargs.pop('cl4_opt', 'Adam')   # 最適化関数
        opt_for_cl4['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl4['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl4['spctrnorm'] = kwargs.get('sn',        None)
        # layer5 プーリング層2
        opt_for_pl2 = {}
        pool2   = kwargs.pop('pool2',   2)            # プーリング
        pl2_pad = kwargs.pop('pl2_pad', 0)            # パディング
        method2 = kwargs.get('method', 'max')         # プーリングメソッド
        opt_for_pl2['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        # layer6 畳込み層5
        M5      = kwargs.pop('M5',    128)#64)            # フィルタ数
        kernel_size5 = kwargs.pop('kernel_size5', 2)  # フィルタ高
        stride5 = kwargs.pop('stride5', 1)            # ストライド
        cl5_pad = kwargs.pop('cl5_pad', 0)            # パディング
        opt_for_cl5 = {}
        opt_for_cl5['activate']  = kwargs.pop('cl5_act', 'ReLU')   # 活性化関数 
        opt_for_cl5['optimize']  = kwargs.pop('cl5_opt', 'Adam')   # 最適化関数
        #opt_for_cl5['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        opt_for_cl5['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl5['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl5['spctrnorm'] = kwargs.get('sn',        None)
        # layer7 畳込み層6
        M6      = kwargs.pop('M6',    128)#64)            # フィルタ数
        kernel_size6 = kwargs.pop('kernel_size6', 2)  # フィルタ高
        stride6 = kwargs.pop('stride6', 1)            # ストライド
        cl6_pad = kwargs.pop('cl6_pad', 0)            # パディング
        opt_for_cl6 = {}
        opt_for_cl6['activate']  = kwargs.pop('cl6_act', 'ReLU')   # 活性化関数 
        opt_for_cl6['optimize']  = kwargs.pop('cl6_opt', 'Adam')   # 最適化関数
        opt_for_cl6['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl6['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl6['spctrnorm'] = kwargs.get('sn',        None)
        # layer8 プーリング層3
        opt_for_pl3 = {}
        pool3   = kwargs.pop('pool3',   2)            # プーリング
        pl3_pad = kwargs.pop('pl3_pad', 0)            # パディング
        method3 = kwargs.pop('method', 'max')         # プーリングメソッド
        opt_for_pl3['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        # layer9 畳込み層7
        M7      = kwargs.pop('M7',    256)#64)            # フィルタ数
        kernel_size7 = kwargs.pop('kernel_size7', 2)  # フィルタ高
        stride7 = kwargs.pop('stride7', 1)            # ストライド
        cl7_pad = kwargs.pop('cl7_pad', 0)            # パディング
        opt_for_cl7 = {}
        opt_for_cl7['activate']  = kwargs.pop('cl7_act', 'ReLU')   # 活性化関数 
        opt_for_cl7['optimize']  = kwargs.pop('cl7_opt', 'Adam')   # 最適化関数
        #opt_for_cl7['dropout']   = kwargs.get('dropout',   True)   # ドロップアウト
        opt_for_cl7['batchnorm'] = kwargs.get('bn',       False)    
        opt_for_cl7['layernorm'] = kwargs.get('ln',       False)    
        opt_for_cl7['spctrnorm'] = kwargs.get('sn',        None)
        # layer10 畳込み層8
        M8      = kwargs.pop('M8',    256)#64)            # フィルタ数
        kernel_size8 = kwargs.pop('kernel_size8', 2)  # フィルタ高
        stride8 = kwargs.pop('stride8', 1)            # ストライド
        cl8_pad = kwargs.pop('cl8_pad', 0)            # パディング
        opt_for_cl8 = {}
        opt_for_cl8['activate']  = kwargs.pop('cl8_act', 'ReLU')   # 活性化関数 
        opt_for_cl8['optimize']  = kwargs.pop('cl8_opt', 'Adam')   # 最適化関数
        opt_for_cl8['batchnorm'] = kwargs.pop('bn',       False)    
        opt_for_cl8['layernorm'] = kwargs.pop('ln',       False)    
        opt_for_cl8['spctrnorm'] = kwargs.get('sn',        None)
        # layer11 グローバルプーリング層
        opt_for_gpl = {}
        opt_for_gpl['dropout']   = kwargs.pop('dropout',   True)   # ドロップアウト
        # layer12 全結合出力層
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']   = kwargs.pop('ol_act', ol_act_cand) #活性化関数
        opt_for_ol['optimize']   = kwargs.pop('ol_opt',   'SGD')   #最適化関数
        opt_for_ol['spctrnorm']  = kwargs.pop('sn',        None)
        # kwargs に残ったものを結合
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_cl3.update(kwargs) 
        opt_for_cl4.update(kwargs) 
        opt_for_cl5.update(kwargs) 
        opt_for_cl6.update(kwargs) 
        opt_for_cl7.update(kwargs) 
        opt_for_cl8.update(kwargs) 
        opt_for_ol.update(kwargs)  

        # -- 各層の初期化 -- 
        self.conv_layer1    = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        self.conv_layer2    = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        self.pooling_layer1 = neuron.Pooling2dLayer(pool1, pl1_pad, method1, **opt_for_pl1)
        self.conv_layer3    = neuron.Conv2dLayer(M3, kernel_size3, stride3, cl3_pad, **opt_for_cl3)
        self.conv_layer4    = neuron.Conv2dLayer(M4, kernel_size4, stride4, cl4_pad, **opt_for_cl4)
        self.pooling_layer2 = neuron.Pooling2dLayer(pool2, pl2_pad, method2, **opt_for_pl2)
        self.conv_layer5    = neuron.Conv2dLayer(M5, kernel_size5, stride5, cl5_pad, **opt_for_cl5)
        self.conv_layer6    = neuron.Conv2dLayer(M6, kernel_size6, stride6, cl6_pad, **opt_for_cl6)
        self.pooling_layer3 = neuron.Pooling2dLayer(pool3, pl3_pad, method3, **opt_for_pl3)
        self.conv_layer7    = neuron.Conv2dLayer(M7, kernel_size7, stride7, cl7_pad, **opt_for_cl7)
        self.conv_layer8    = neuron.Conv2dLayer(M8, kernel_size8, stride8, cl8_pad, **opt_for_cl8)
        self.global_pooling = neuron.GlobalAveragePooling(**opt_for_gpl)
        self.output_layer   = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.pooling_layer1)
        self.layers.append(self.conv_layer3)
        self.layers.append(self.conv_layer4)
        self.layers.append(self.pooling_layer2)
        self.layers.append(self.conv_layer5)
        self.layers.append(self.conv_layer6)
        self.layers.append(self.pooling_layer3)
        self.layers.append(self.conv_layer7)
        self.layers.append(self.conv_layer8)
        self.layers.append(self.global_pooling)
        self.layers.append(self.output_layer)


class CNN_mud(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        M2           = kwargs.pop('M2',         16)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 4)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆プーリング
        Pl2    = kwargs.pop('Pl2',  2)            
        opt_for_l2 = {} 
        opt_for_l2['method']    = kwargs.pop('pl_method', 'average')
        # layer3 逆畳込み　　
        M3      = kwargs.pop('M3',      4)           # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2) # フィルタ高と幅　　
        stride3 = kwargs.pop('stride3', 2)           # ストライド　　　　
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('dcl_act',      'Mish')
        opt_for_l3['optimize']  = kwargs.pop('dcl_opt',   'RMSProp')
        opt_for_l3['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l3['layernorm'] = kwargs.pop('ln',        False)    
        # layer4 出力層
        opt_for_l4 = {} 
        opt_for_l4['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l4['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l4['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)
        opt_for_l4.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer    = neuron.NeuronLayer(M2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.unpooling_layer = neuron.UnPooling2dLayer(M2, image_size2, Pl2, 0, **opt_for_l2)
        # layer 3
        self.deconv_layer    = neuron.DeConv2dLayer(M3, kernel_size3, stride3, 0, **opt_for_l3)
        # layer 4
        self.output_layer    = neuron.NeuronLayer(Out, **opt_for_l4)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.unpooling_layer)
        self.layers.append(self.deconv_layer)    
        self.layers.append(self.output_layer)


class CNN_mddd(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        C2           = kwargs.pop('C2',         16)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 4)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆畳込み　　
        M2           = kwargs.pop('M2',          16) # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 3) # フィルタ高と幅　　
        stride2      = kwargs.pop('stride2',      2) # ストライド　　　　
        opt_for_l2 = {} 
        opt_for_l2['activate']  = kwargs.pop('dcl1_act',     'Mish')
        opt_for_l2['optimize']  = kwargs.pop('dcl1_opt',  'RMSProp')
        opt_for_l2['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l2['layernorm'] = kwargs.get('ln',        False)    
        # layer3 逆畳込み　　
        M3           = kwargs.pop('M3',           8) # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 3) # フィルタ高と幅　　
        stride3      = kwargs.pop('stride3',      2) # ストライド　　　　
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('dcl2_act',     'Mish')
        opt_for_l3['optimize']  = kwargs.pop('dcl2_opt',  'RMSProp')
        opt_for_l3['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l3['layernorm'] = kwargs.get('ln',        False)    
        # layer4 逆畳込み
        M4           = kwargs.pop('M4',           4) # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 3) # フィルタ高と幅　　
        stride4      = kwargs.pop('stride4',      2) # ストライド　　　　
        opt_for_l4 = {} 
        opt_for_l4['activate']  = kwargs.pop('dcl3_act',     'Mish')
        opt_for_l4['optimize']  = kwargs.pop('dcl3_opt',  'RMSProp')
        opt_for_l4['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l4['layernorm'] = kwargs.pop('ln',        False)    
        # layer5 出力層
        opt_for_l5 = {} 
        opt_for_l5['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l5['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l5['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)
        opt_for_l4.update(kwargs)
        opt_for_l5.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer  = neuron.NeuronLayer(C2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.deconv_layer1 = neuron.DeConv2dLayer(C2, image_size2, M2, kernel_size2, stride2, 0, **opt_for_l2)
        # layer 3
        self.deconv_layer2 = neuron.DeConv2dLayer(M3, kernel_size3, stride3, 0, **opt_for_l3)
        # layer 4
        self.deconv_layer3 = neuron.DeConv2dLayer(M4, kernel_size4, stride4, 0, **opt_for_l4)
        # layer 5
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_l5)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.deconv_layer1)
        self.layers.append(self.deconv_layer2)    
        self.layers.append(self.deconv_layer3)
        self.layers.append(self.output_layer)

class CNN_mddc(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        C2           = kwargs.pop('C2',         16)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 4)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆畳込み　　
        M2           = kwargs.pop('M2',          16) # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 3) # フィルタ高と幅　　
        stride2      = kwargs.pop('stride2',      2) # ストライド　　　　
        opt_for_l2 = {} 
        opt_for_l2['activate']  = kwargs.pop('dcl1_act',     'Mish')
        opt_for_l2['optimize']  = kwargs.pop('dcl1_opt',  'RMSProp')
        opt_for_l2['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l2['layernorm'] = kwargs.get('ln',        False)    
        # layer3 逆畳込み　　
        M3           = kwargs.pop('M3',           8) # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 3) # フィルタ高と幅　　
        stride3      = kwargs.pop('stride3',      2) # ストライド　　　　
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('dcl2_act',     'Mish')
        opt_for_l3['optimize']  = kwargs.pop('dcl2_opt',  'RMSProp')
        opt_for_l3['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l3['layernorm'] = kwargs.get('ln',        False)    
        # layer4 畳込み
        M4           = kwargs.pop('M4',           8) # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 2) # フィルタ高と幅　　
        stride4      = kwargs.pop('stride4',      1) # ストライド　　　　
        opt_for_l4 = {} 
        opt_for_l4['activate']  = kwargs.pop('dcl3_act',     'Mish')
        opt_for_l4['optimize']  = kwargs.pop('dcl3_opt',  'RMSProp')
        opt_for_l4['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l4['layernorm'] = kwargs.pop('ln',        False)    
        # layer5 出力層
        opt_for_l5 = {} 
        opt_for_l5['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l5['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l5['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)
        opt_for_l4.update(kwargs)
        opt_for_l5.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer  = neuron.NeuronLayer(C2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.deconv_layer1 = neuron.DeConv2dLayer(C2, image_size2, M2, kernel_size2, stride2, 0, **opt_for_l2)
        # layer 3
        self.deconv_layer2 = neuron.DeConv2dLayer(M3, kernel_size3, stride3, 0, **opt_for_l3)
        # layer 4
        self.conv_layer    = neuron.Conv2dLayer(M4, kernel_size4, stride4, 0, **opt_for_l4)
        # layer 5
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_l5)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.deconv_layer1)
        self.layers.append(self.deconv_layer2)    
        self.layers.append(self.conv_layer)
        self.layers.append(self.output_layer)


class CNN_md(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        C2           = kwargs.pop('C2',         16)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 4)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆畳込み　　
        M2           = kwargs.pop('M2',          16) # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 3) # フィルタ高と幅　　
        stride2      = kwargs.pop('stride2',      2) # ストライド　　　　
        opt_for_l2 = {} 
        opt_for_l2['activate']  = kwargs.pop('dcl_act',     'Mish')
        opt_for_l2['optimize']  = kwargs.pop('dcl_opt',  'RMSProp')
        opt_for_l2['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l2['layernorm'] = kwargs.pop('ln',        False)    
        # layer3 出力層
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l3['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l3['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer = neuron.NeuronLayer(C2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.deconv_layer = neuron.DeConv2dLayer(C2, image_size2, M2, kernel_size2, stride2, 0, **opt_for_l2)
        # layer 3
        self.output_layer = neuron.NeuronLayer(Out, **opt_for_l3)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.deconv_layer)
        self.layers.append(self.output_layer)


class CNN_mdc(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        C2           = kwargs.pop('C2',         16)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 4)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆畳込み　　
        M2           = kwargs.pop('M2',          16) # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 3) # フィルタ高と幅　　
        stride2      = kwargs.pop('stride2',      2) # ストライド　　　　
        opt_for_l2 = {} 
        opt_for_l2['activate']  = kwargs.pop('dcl_act',     'Mish')
        opt_for_l2['optimize']  = kwargs.pop('dcl_opt',  'RMSProp')
        opt_for_l2['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l2['layernorm'] = kwargs.get('ln',        False)    
        # layer3 畳込み　　
        M3           = kwargs.pop('M3',          16) # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2) # フィルタ高と幅　　
        stride3      = kwargs.pop('stride3',      1) # ストライド　　　　
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('cl_act',      'Mish')
        opt_for_l3['optimize']  = kwargs.pop('cl_opt',   'RMSProp')
        opt_for_l3['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l3['layernorm'] = kwargs.pop('ln',        False)    
        # layer4 出力層
        opt_for_l4 = {} 
        opt_for_l4['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l4['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l4['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)
        opt_for_l4.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer = neuron.NeuronLayer(C2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.deconv_layer = neuron.DeConv2dLayer(C2, image_size2, M2, kernel_size2, stride2, 0, **opt_for_l2)
        # layer 3
        self.conv_layer   = neuron.Conv2dLayer(M3, kernel_size3, stride3, 0, **opt_for_l3)
        # layer 4
        self.output_layer = neuron.NeuronLayer(Out, **opt_for_l4)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.deconv_layer)
        self.layers.append(self.conv_layer)    
        self.layers.append(self.output_layer)

class CNN_mdd(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        C2           = kwargs.pop('C2',         16)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 4)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆畳込み　　
        M2           = kwargs.pop('M2',          16) # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 3) # フィルタ高と幅　　
        stride2      = kwargs.pop('stride2',      2) # ストライド　　　　
        opt_for_l2 = {} 
        opt_for_l2['activate']  = kwargs.pop('dcl1_act',     'Mish')
        opt_for_l2['optimize']  = kwargs.pop('dcl1_opt',  'RMSProp')
        opt_for_l2['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l2['layernorm'] = kwargs.get('ln',        False)    
        # layer3 逆畳込み　　
        M3           = kwargs.pop('M3',           8) # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 3) # フィルタ高と幅　　
        stride3      = kwargs.pop('stride3',      2) # ストライド　　　　
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('dcl2_act',     'Mish')
        opt_for_l3['optimize']  = kwargs.pop('dcl2_opt',  'RMSProp')
        opt_for_l3['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l3['layernorm'] = kwargs.pop('ln',        False)    
        # layer4 出力層
        opt_for_l4 = {} 
        opt_for_l4['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l4['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l4['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)
        opt_for_l4.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer  = neuron.NeuronLayer(C2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.deconv_layer1 = neuron.DeConv2dLayer(C2, image_size2, M2, kernel_size2, stride2, 0, **opt_for_l2)
        # layer 3
        self.deconv_layer2 = neuron.DeConv2dLayer(M3, kernel_size3, stride3, 0, **opt_for_l3)
        # layer 4
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_l4)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.deconv_layer1)
        self.layers.append(self.deconv_layer2)    
        self.layers.append(self.output_layer)


class CNN_mudd(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        M2           = kwargs.pop('M2',         16)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 4)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆プーリング
        Pl2    = kwargs.pop('Pl2',  2)            
        opt_for_l2 = {} 
        opt_for_l2['method']    = kwargs.pop('pl1_method','average')
        # layer3 逆畳込み　　
        M3           = kwargs.pop('M3',           8) # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 3) # フィルタ高と幅　　
        stride3      = kwargs.pop('stride3',      2) # ストライド　　　　
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('dcl1_act',     'Mish')
        opt_for_l3['optimize']  = kwargs.pop('dcl1_opt',  'RMSProp')
        opt_for_l3['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l3['layernorm'] = kwargs.get('ln',        False)    
        # layer4 逆畳込み
        M4           = kwargs.pop('M4',           4) # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 3) # フィルタ高と幅　　
        stride4      = kwargs.pop('stride4',      2) # ストライド　　　　
        opt_for_l4 = {} 
        opt_for_l4['activate']  = kwargs.pop('dcl2_act',     'Mish')
        opt_for_l4['optimize']  = kwargs.pop('dcl2_opt',  'RMSProp')
        opt_for_l4['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l4['layernorm'] = kwargs.pop('ln',        False)    
        # layer5 出力層
        opt_for_l5 = {} 
        opt_for_l5['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l5['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l5['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)
        opt_for_l4.update(kwargs)
        opt_for_l5.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer    = neuron.NeuronLayer(M2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.unpooling_layer = neuron.UnPooling2dLayer(M2, image_size2, Pl2, 0, **opt_for_l2)
        # layer 3
        self.deconv_layer1   = neuron.DeConv2dLayer(M3, kernel_size3, stride3, 0, **opt_for_l3)
        # layer 4
        self.deconv_layer2   = neuron.DeConv2dLayer(M4, kernel_size4, stride4, 0, **opt_for_l4)
        # layer 5
        self.output_layer    = neuron.NeuronLayer(Out, **opt_for_l5)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.unpooling_layer)
        self.layers.append(self.deconv_layer1)    
        self.layers.append(self.deconv_layer2)
        self.layers.append(self.output_layer)


class CNN_mudud(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        M2           = kwargs.pop('M2',         16)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 4)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆プーリング
        Pl2    = kwargs.pop('Pl2',  2)            
        opt_for_l2 = {} 
        opt_for_l2['method']    = kwargs.pop('pl1_method','average')
        # layer3 逆畳込み　　
        M3           = kwargs.pop('M3',           7) # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2) # フィルタ高と幅　　
        stride3      = kwargs.pop('stride3',      2) # ストライド　　　　
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('dcl1_act',     'Mish')
        opt_for_l3['optimize']  = kwargs.pop('dcl1_opt',  'RMSProp')
        opt_for_l3['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l3['layernorm'] = kwargs.get('ln',        False)    
        # layer4 逆プーリング 
        Pl4    = kwargs.pop('Pl4',  2)
        opt_for_l4 = {} 
        opt_for_l4['method']    = kwargs.pop('pl2_method','average')
        # layer5 逆畳込み
        M5           = kwargs.pop('M5',           3) # フィルタ数
        kernel_size5 = kwargs.pop('kernel_size5', 2) # フィルタ高と幅　　
        stride5      = kwargs.pop('stride5',      2) # ストライド　　　　
        opt_for_l5 = {} 
        opt_for_l5['activate']  = kwargs.pop('dcl2_act',     'Mish')
        opt_for_l5['optimize']  = kwargs.pop('dcl2_opt',  'RMSProp')
        opt_for_l5['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l5['layernorm'] = kwargs.pop('ln',        False)    
        # layer6 出力層
        opt_for_l6 = {} 
        opt_for_l6['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l6['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l6['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)
        opt_for_l4.update(kwargs)
        opt_for_l5.update(kwargs)
        opt_for_l6.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer     = neuron.NeuronLayer(M2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.unpooling_layer1 = neuron.UnPooling2dLayer(M2, image_size2, Pl2, 0, **opt_for_l2)
        # layer 3
        self.deconv_layer1    = neuron.DeConv2dLayer(M3, kernel_size3, stride3, 0, **opt_for_l3)
        # layer 4
        self.unpooling_layer2 = neuron.UnPooling2dLayer(Pl4, 0, **opt_for_l4)
        # layer 5
        self.deconv_layer2    = neuron.DeConv2dLayer(M5, kernel_size5, stride5, 0, **opt_for_l5)
        # layer 6
        self.output_layer     = neuron.NeuronLayer(Out, **opt_for_l6)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.unpooling_layer1)
        self.layers.append(self.deconv_layer1)    
        self.layers.append(self.unpooling_layer2)
        self.layers.append(self.deconv_layer2)
        self.layers.append(self.output_layer)


class CNN_mdddc(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        C2           = kwargs.pop('C2',         16)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 3)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆畳込み　　
        M2           = kwargs.pop('M2',          16) # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 3) # フィルタ高と幅　　
        stride2      = kwargs.pop('stride2',      2) # ストライド　　　　
        opt_for_l2 = {} 
        opt_for_l2['activate']  = kwargs.pop('dcl1_act',     'Mish')
        opt_for_l2['optimize']  = kwargs.pop('dcl1_opt',  'RMSProp')
        opt_for_l2['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l2['layernorm'] = kwargs.get('ln',        False)    
        # layer3 逆畳込み　　
        M3           = kwargs.pop('M3',           8) # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 3) # フィルタ高と幅　　
        stride3      = kwargs.pop('stride3',      2) # ストライド　　　　
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('dcl2_act',     'Mish')
        opt_for_l3['optimize']  = kwargs.pop('dcl2_opt',  'RMSProp')
        opt_for_l3['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l3['layernorm'] = kwargs.get('ln',        False)    
        # layer4 逆畳込み
        M4           = kwargs.pop('M4',           4) # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 3) # フィルタ高と幅　　
        stride4      = kwargs.pop('stride4',      2) # ストライド　　　　
        opt_for_l4 = {} 
        opt_for_l4['activate']  = kwargs.pop('dcl3_act',     'Mish')
        opt_for_l4['optimize']  = kwargs.pop('dcl3_opt',  'RMSProp')
        opt_for_l4['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l4['layernorm'] = kwargs.get('ln',        False)    
        # layer5 畳込み
        M5           = kwargs.pop('M5',           3) # フィルタ数
        kernel_size5 = kwargs.pop('kernel_size5', 2) # フィルタ高と幅　　
        stride5      = kwargs.pop('stride5',      1) # ストライド　　　　
        opt_for_l5 = {} 
        opt_for_l5['activate']  = kwargs.pop('cl_act',       'Mish')
        opt_for_l5['optimize']  = kwargs.pop('cl_opt',    'RMSProp')
        opt_for_l5['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l5['layernorm'] = kwargs.pop('ln',        False)    
        # layer6 出力層
        opt_for_l6 = {} 
        opt_for_l6['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l6['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l6['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)
        opt_for_l4.update(kwargs)
        opt_for_l5.update(kwargs)
        opt_for_l6.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer   = neuron.NeuronLayer(C2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.deconv_layer1  = neuron.DeConv2dLayer(C2, image_size2, M2, kernel_size2, stride2, 0, **opt_for_l2)
        # layer 3
        self.deconv_layer2  = neuron.DeConv2dLayer(M3, kernel_size3, stride3, 0, **opt_for_l3)
        # layer 4
        self.deconv_layer3  = neuron.DeConv2dLayer(M4, kernel_size4, stride4, 0, **opt_for_l4)
        # layer 5
        self.conv_layer     = neuron.Conv2dLayer(M5, kernel_size5, stride5, 0, **opt_for_l5)
        # layer 6
        self.output_layer   = neuron.NeuronLayer(Out, **opt_for_l6)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.deconv_layer1)
        self.layers.append(self.deconv_layer2)    
        self.layers.append(self.deconv_layer3)
        self.layers.append(self.conv_layer)
        self.layers.append(self.output_layer)

class CNN_mddddc(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        C2           = kwargs.pop('C2',         16)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 3)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆畳込み　　
        M2           = kwargs.pop('M2',          16) # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 3) # フィルタ高と幅　　
        stride2      = kwargs.pop('stride2',      2) # ストライド　　　　
        opt_for_l2 = {} 
        opt_for_l2['activate']  = kwargs.pop('dcl1_act',     'Mish')
        opt_for_l2['optimize']  = kwargs.pop('dcl1_opt',  'RMSProp')
        opt_for_l2['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l2['layernorm'] = kwargs.get('ln',        False)    
        # layer3 逆畳込み　　
        M3           = kwargs.pop('M3',           8) # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 3) # フィルタ高と幅　　
        stride3      = kwargs.pop('stride3',      2) # ストライド　　　　
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('dcl2_act',     'Mish')
        opt_for_l3['optimize']  = kwargs.pop('dcl2_opt',  'RMSProp')
        opt_for_l3['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l3['layernorm'] = kwargs.get('ln',        False)    
        # layer4 逆畳込み
        M4           = kwargs.pop('M4',           4) # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 3) # フィルタ高と幅　　
        stride4      = kwargs.pop('stride4',      2) # ストライド　　　　
        opt_for_l4 = {} 
        opt_for_l4['activate']  = kwargs.pop('dcl3_act',     'Mish')
        opt_for_l4['optimize']  = kwargs.pop('dcl3_opt',  'RMSProp')
        opt_for_l4['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l4['layernorm'] = kwargs.get('ln',        False)    
        # layer5 逆畳込み
        M5           = kwargs.pop('M5',           4) # フィルタ数
        kernel_size5 = kwargs.pop('kernel_size5', 3) # フィルタ高と幅　　
        stride5      = kwargs.pop('stride5',      2) # ストライド　　　　
        opt_for_l5 = {} 
        opt_for_l5['activate']  = kwargs.pop('dcl4_act',     'Mish')
        opt_for_l5['optimize']  = kwargs.pop('dcl4_opt',  'RMSProp')
        opt_for_l5['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l5['layernorm'] = kwargs.get('ln',        False)    
        # layer6 畳込み
        M6           = kwargs.pop('M6',           3) # フィルタ数
        kernel_size6 = kwargs.pop('kernel_size6', 2) # フィルタ高と幅　　
        stride6      = kwargs.pop('stride6',      1) # ストライド　　　　
        opt_for_l6 = {} 
        opt_for_l6['activate']  = kwargs.pop('cl_act',       'Mish')
        opt_for_l6['optimize']  = kwargs.pop('cl_opt',    'RMSProp')
        opt_for_l6['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l6['layernorm'] = kwargs.pop('ln',        False)    
        # layer7 出力層
        opt_for_l7 = {} 
        opt_for_l7['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l7['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l7['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)
        opt_for_l4.update(kwargs)
        opt_for_l5.update(kwargs)
        opt_for_l6.update(kwargs)
        opt_for_l7.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer   = neuron.NeuronLayer(C2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.deconv_layer1  = neuron.DeConv2dLayer(C2, image_size2, M2, kernel_size2, stride2, 0, **opt_for_l2)
        # layer 3
        self.deconv_layer2  = neuron.DeConv2dLayer(M3, kernel_size3, stride3, 0, **opt_for_l3)
        # layer 4
        self.deconv_layer3  = neuron.DeConv2dLayer(M4, kernel_size4, stride4, 0, **opt_for_l4)
        # layer 5
        self.deconv_layer4  = neuron.DeConv2dLayer(M5, kernel_size5, stride5, 0, **opt_for_l5)
        # layer 6
        self.conv_layer     = neuron.Conv2dLayer(M6, kernel_size6, stride6, 0, **opt_for_l6)
        # layer 7
        self.output_layer   = neuron.NeuronLayer(Out, **opt_for_l7)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.deconv_layer1)
        self.layers.append(self.deconv_layer2)    
        self.layers.append(self.deconv_layer3)
        self.layers.append(self.deconv_layer4)
        self.layers.append(self.conv_layer)
        self.layers.append(self.output_layer)

class CNN_mdcdc(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        C2           = kwargs.pop('C2',         16)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 4)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆畳込み1　　
        M2           = kwargs.pop('M2',          16) # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 3) # フィルタ高と幅　　
        stride2      = kwargs.pop('stride2',      2) # ストライド　　　　
        opt_for_l2 = {} 
        opt_for_l2['activate']  = kwargs.pop('dcl1_act',     'Mish')
        opt_for_l2['optimize']  = kwargs.pop('dcl1_opt',  'RMSProp')
        opt_for_l2['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l2['layernorm'] = kwargs.get('ln',        False)    
        # layer3 畳込み1　　
        M3           = kwargs.pop('M3',          16) # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2) # フィルタ高と幅　　
        stride3      = kwargs.pop('stride3',      1) # ストライド　　　　
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('cl1_act',      'Mish')
        opt_for_l3['optimize']  = kwargs.pop('cl1_opt',   'RMSProp')
        opt_for_l3['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l3['layernorm'] = kwargs.get('ln',        False)    
        # layer4 逆畳込み2
        M4           = kwargs.pop('M4',           8) # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 3) # フィルタ高と幅　　
        stride4      = kwargs.pop('stride4',      2) # ストライド　　　　
        opt_for_l4 = {} 
        opt_for_l4['activate']  = kwargs.pop('dcl2_act',     'Mish')
        opt_for_l4['optimize']  = kwargs.pop('dcl2_opt',  'RMSProp')
        opt_for_l4['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l4['layernorm'] = kwargs.get('ln',        False)    
        # layer5 畳込み2
        M5           = kwargs.pop('M5',           8) # フィルタ数
        kernel_size5 = kwargs.pop('kernel_size5', 2) # フィルタ高と幅　　
        stride5      = kwargs.pop('stride5',      1) # ストライド　　　　
        opt_for_l5 = {} 
        opt_for_l5['activate']  = kwargs.pop('cl2_act',      'Mish')
        opt_for_l5['optimize']  = kwargs.pop('cl2_opt',   'RMSProp')
        opt_for_l5['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l5['layernorm'] = kwargs.pop('ln',        False)    
        # layer6 出力層
        opt_for_l6 = {} 
        opt_for_l6['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l6['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l6['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)
        opt_for_l4.update(kwargs)
        opt_for_l5.update(kwargs)
        opt_for_l6.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer  = neuron.NeuronLayer(C2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.deconv_layer1 = neuron.DeConv2dLayer(C2, image_size2, M2, kernel_size2, stride2, 0, **opt_for_l2)
        # layer 3
        self.conv_layer1   = neuron.Conv2dLayer(M3, kernel_size3, stride3, 0, **opt_for_l3)
        # layer 4
        self.deconv_layer2 = neuron.DeConv2dLayer(M4, kernel_size4, stride4, 0, **opt_for_l4)
        # layer 5
        self.conv_layer2   = neuron.Conv2dLayer(M5, kernel_size5, stride5, 0, **opt_for_l5)
        # layer 6
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_l6)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.deconv_layer1)
        self.layers.append(self.conv_layer1)    
        self.layers.append(self.deconv_layer2)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.output_layer)


class CNN_mdcdcdc(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        C2           = kwargs.pop('C2',         16)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 4)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆畳込み1　　
        M2           = kwargs.pop('M2',          16) # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 3) # フィルタ高と幅　　
        stride2      = kwargs.pop('stride2',      2) # ストライド　　　　
        opt_for_l2 = {} 
        opt_for_l2['activate']  = kwargs.pop('dcl1_act',     'Mish')
        opt_for_l2['optimize']  = kwargs.pop('dcl1_opt',  'RMSProp')
        opt_for_l2['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l2['layernorm'] = kwargs.get('ln',        False)    
        # layer3 畳込み1　　
        M3           = kwargs.pop('M3',          16) # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2) # フィルタ高と幅　　
        stride3      = kwargs.pop('stride3',      1) # ストライド　　　　
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('cl1_act',     'Mish')
        opt_for_l3['optimize']  = kwargs.pop('cl1_opt',  'RMSProp')
        opt_for_l3['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l3['layernorm'] = kwargs.get('ln',        False)    
        # layer4 逆畳込み2
        M4           = kwargs.pop('M4',           8) # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 3) # フィルタ高と幅　　
        stride4      = kwargs.pop('stride4',      2) # ストライド　　　　
        opt_for_l4 = {} 
        opt_for_l4['activate']  = kwargs.pop('dcl2_act',     'Mish')
        opt_for_l4['optimize']  = kwargs.pop('dcl2_opt',  'RMSProp')
        opt_for_l4['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l4['layernorm'] = kwargs.get('ln',        False)    
        # layer5 畳込み2
        M5           = kwargs.pop('M5',           8) # フィルタ数
        kernel_size5 = kwargs.pop('kernel_size5', 2) # フィルタ高と幅　　
        stride5      = kwargs.pop('stride5',      1) # ストライド　　　　
        opt_for_l5 = {} 
        opt_for_l5['activate']  = kwargs.pop('cl2_act',       'Mish')
        opt_for_l5['optimize']  = kwargs.pop('cl2_opt',    'RMSProp')
        opt_for_l5['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l5['layernorm'] = kwargs.get('ln',        False)    
        # layer6 逆畳込み3
        M6           = kwargs.pop('M6',           4) # フィルタ数
        kernel_size6 = kwargs.pop('kernel_size6', 3) # フィルタ高と幅　　
        stride6      = kwargs.pop('stride6',      2) # ストライド　　　　
        opt_for_l6 = {} 
        opt_for_l6['activate']  = kwargs.pop('dcl3_act',     'Mish')
        opt_for_l6['optimize']  = kwargs.pop('dcl3_opt',  'RMSProp')
        opt_for_l6['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l6['layernorm'] = kwargs.get('ln',        False)    
        # layer7 畳込み3
        M7           = kwargs.pop('M7',           4) # フィルタ数
        kernel_size7 = kwargs.pop('kernel_size7', 2) # フィルタ高と幅　　
        stride7      = kwargs.pop('stride7',      1) # ストライド　　　　
        opt_for_l7 = {} 
        opt_for_l7['activate']  = kwargs.pop('cl3_act',       'Mish')
        opt_for_l7['optimize']  = kwargs.pop('cl3_opt',    'RMSProp')
        opt_for_l7['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l7['layernorm'] = kwargs.pop('ln',        False)    
        # layer8 出力層
        opt_for_l8 = {} 
        opt_for_l8['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l8['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l8['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)
        opt_for_l4.update(kwargs)
        opt_for_l5.update(kwargs)
        opt_for_l6.update(kwargs)
        opt_for_l7.update(kwargs)
        opt_for_l8.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer  = neuron.NeuronLayer(C2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.deconv_layer1 = neuron.DeConv2dLayer(C2, image_size2, M2, kernel_size2, stride2, 0, **opt_for_l2)
        # layer 3
        self.conv_layer1   = neuron.Conv2dLayer(M3, kernel_size3, stride3, 0, **opt_for_l3)
        # layer 4
        self.deconv_layer2 = neuron.DeConv2dLayer(M4, kernel_size4, stride4, 0, **opt_for_l4)
        # layer 5
        self.conv_layer2   = neuron.Conv2dLayer(M5, kernel_size5, stride5, 0, **opt_for_l5)
        # layer 6
        self.deconv_layer3 = neuron.DeConv2dLayer(M6, kernel_size6, stride6, 0, **opt_for_l6)
        # layer 7
        self.conv_layer3   = neuron.Conv2dLayer(M7, kernel_size7, stride7, 0, **opt_for_l7)
        # layer 8
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_l8)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.deconv_layer1)
        self.layers.append(self.conv_layer1)    
        self.layers.append(self.deconv_layer2)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.deconv_layer3)
        self.layers.append(self.conv_layer3)
        self.layers.append(self.output_layer)

class CNN_mdddcc(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        # 入力(ノイズ)の大きさは最初のfowardで入力により決まる

        '''
        In, Out = super().__init__(*args, **kwargs)
        # -- 各層の初期化 --
        # layer1 中間層
        C2           = kwargs.pop('C2',         24)  # フィルタ数
        image_size2  = kwargs.pop('image_size2', 3)  # 画像サイズ
        Ih2, Iw2     = image_size2 if isinstance(image_size2, (tuple, list)) \
                                   else (image_size2, image_size2)
        opt_for_l1 = {} 
        opt_for_l1['activate']  = kwargs.pop('ml_act',       'Mish')
        opt_for_l1['optimize']  = kwargs.pop('ml_opt',    'RMSProp')
        opt_for_l1['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l1['layernorm'] = kwargs.get('ln',        False)    
        # layer2 逆畳込み　　
        M2           = kwargs.pop('M2',          12) # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 3) # フィルタ高と幅　　
        stride2      = kwargs.pop('stride2',      2) # ストライド　　　　
        opt_for_l2 = {} 
        opt_for_l2['activate']  = kwargs.pop('dcl1_act',     'Mish')
        opt_for_l2['optimize']  = kwargs.pop('dcl1_opt',  'RMSProp')
        opt_for_l2['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l2['layernorm'] = kwargs.get('ln',        False)    
        # layer3 逆畳込み　　
        M3           = kwargs.pop('M3',           8) # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 3) # フィルタ高と幅　　
        stride3      = kwargs.pop('stride3',      2) # ストライド　　　　
        opt_for_l3 = {} 
        opt_for_l3['activate']  = kwargs.pop('dcl2_act',     'Mish')
        opt_for_l3['optimize']  = kwargs.pop('dcl2_opt',  'RMSProp')
        opt_for_l3['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l3['layernorm'] = kwargs.get('ln',        False)    
        # layer4 逆畳込み
        M4           = kwargs.pop('M4',           4) # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 3) # フィルタ高と幅　　
        stride4      = kwargs.pop('stride4',      2) # ストライド　　　　
        opt_for_l4 = {} 
        opt_for_l4['activate']  = kwargs.pop('dcl3_act',     'Mish')
        opt_for_l4['optimize']  = kwargs.pop('dcl3_opt',  'RMSProp')
        opt_for_l4['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l4['layernorm'] = kwargs.get('ln',        False)    
        # layer5 畳込み
        M5           = kwargs.pop('M5',           3) # フィルタ数
        kernel_size5 = kwargs.pop('kernel_size5', 2) # フィルタ高と幅　　
        stride5      = kwargs.pop('stride5',      1) # ストライド　　　　
        opt_for_l5 = {} 
        opt_for_l5['activate']  = kwargs.pop('dcl4_act',     'Mish')
        opt_for_l5['optimize']  = kwargs.pop('dcl4_opt',  'RMSProp')
        opt_for_l5['batchnorm'] = kwargs.get('bn',        False)    
        opt_for_l5['layernorm'] = kwargs.get('ln',        False)    
        # layer6 畳込み
        M6           = kwargs.pop('M6',           3) # フィルタ数
        kernel_size6 = kwargs.pop('kernel_size6', 2) # フィルタ高と幅　　
        stride6      = kwargs.pop('stride6',      1) # ストライド　　　　
        opt_for_l6 = {} 
        opt_for_l6['activate']  = kwargs.pop('cl_act',       'Mish')
        opt_for_l6['optimize']  = kwargs.pop('cl_opt',    'RMSProp')
        opt_for_l6['batchnorm'] = kwargs.pop('bn',        False)    
        opt_for_l6['layernorm'] = kwargs.pop('ln',        False)    
        # layer7 出力層
        opt_for_l7 = {} 
        opt_for_l7['activate']  = kwargs.pop('ol_act',       'Tanh')
        opt_for_l7['optimize']  = kwargs.pop('ol_opt',    'RMSProp')
        opt_for_l7['full_connection'] = True         
        # kwargsに残ったものを結合
        opt_for_l1.update(kwargs)
        opt_for_l2.update(kwargs)
        opt_for_l3.update(kwargs)
        opt_for_l4.update(kwargs)
        opt_for_l5.update(kwargs)
        opt_for_l6.update(kwargs)
        opt_for_l7.update(kwargs)

        # -- 各層の初期化 -- 
        # layer 1
        self.middle_layer  = neuron.NeuronLayer(C2*Ih2*Iw2, **opt_for_l1)
        # layer 2
        self.deconv_layer1 = neuron.DeConv2dLayer(C2, image_size2, M2, kernel_size2, stride2, 0, **opt_for_l2)
        # layer 3
        self.deconv_layer2 = neuron.DeConv2dLayer(M3, kernel_size3, stride3, 0, **opt_for_l3)
        # layer 4
        self.deconv_layer3 = neuron.DeConv2dLayer(M4, kernel_size4, stride4, 0, **opt_for_l4)
        # layer 5
        self.conv_layer1   = neuron.Conv2dLayer(M5, kernel_size5, stride5, 0, **opt_for_l5)
        # layer 6
        self.conv_layer2   = neuron.Conv2dLayer(M6, kernel_size6, stride6, 0, **opt_for_l6)
        # layer 7
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_l7)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.deconv_layer1)
        self.layers.append(self.deconv_layer2)    
        self.layers.append(self.deconv_layer3)
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.output_layer)


class CNN_c2pm(CNN_ccpm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class CNN_c2pc2pm(CNN_ccpccpm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class CNN_c2pc2pc2pm(CNN_ccpccpccpm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CNN_c(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層＋プーリング層、全結合層なし
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層
        M      = kwargs.pop('M',      16)          # フィルタ数
        kernel_size = kwargs.pop('kernel_size', 2) # フィルタ高
        stride = kwargs.pop('stride',  1)          # ストライド
        cl_pad = kwargs.pop('cl_pad',  0)          # パディング
        opt_for_cl = {}
        opt_for_cl['activate']  = kwargs.pop('cl_act', 'ReLU')    # 活性化関数 
        opt_for_cl['optimize']  = kwargs.pop('cl_opt', 'Adam')    # 最適化関数
        opt_for_cl['batchnorm'] = kwargs.pop('bn',      False)    # bn は中間層でも使う
        opt_for_cl['layernorm'] = kwargs.pop('ln',      False)    # ln は中間層でも使う
        opt_for_cl['spctrnorm'] = kwargs.get('sn',      None)
        # layer3 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['optimize']  = kwargs.pop('ol_opt', 'SGD')     # 最適化関数
        opt_for_ol['spctrnorm'] = kwargs.pop('sn',      None)
        opt_for_ol['full_connection'] = True 
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl.update(kwargs) 
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        self.conv_layer   = neuron.Conv2dLayer(M, kernel_size, stride, cl_pad, **opt_for_cl)
        self.output_layer = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer)
        self.layers.append(self.output_layer)


class CNN_cp(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層＋プーリング層＋全結合層
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層
        M      = kwargs.pop('M',      16)          # フィルタ数
        kernel_size = kwargs.pop('kernel_size', 2) # フィルタ高
        stride = kwargs.pop('stride',  1)          # ストライド
        cl_pad = kwargs.pop('cl_pad',  0)          # パディング
        opt_for_cl = {}
        opt_for_cl['activate']  = kwargs.pop('cl_act', 'ReLU')    # 活性化関数 
        opt_for_cl['optimize']  = kwargs.pop('cl_opt', 'Adam')    # 最適化関数
        opt_for_cl['batchnorm'] = kwargs.pop('bn',      False)    # bn は中間層でも使う
        opt_for_cl['layernorm'] = kwargs.pop('ln',      False)    # ln は中間層でも使う
        opt_for_cl['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 プーリング層
        pool   = kwargs.pop('pool',           2)   # プーリング
        pl_pad = kwargs.pop('pl_pad',         0)   # パディング
        method = kwargs.pop('method',     'max')   # プーリングメソッド 
        # layer3 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['optimize']  = kwargs.pop('ol_opt', 'SGD')     # 最適化関数
        opt_for_ol['spctrnorm'] = kwargs.pop('sn',      None)
        opt_for_ol['full_connection'] = True 
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl.update(kwargs) 
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        # layer 1
        self.conv_layer   = neuron.Conv2dLayer(M, kernel_size, stride, cl_pad, **opt_for_cl)
        # layer 2
        self.pooling_layer = neuron.Pooling2dLayer(pool, pl_pad, method)
        # layer 3
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer)
        self.layers.append(self.pooling_layer)
        self.layers.append(self.output_layer)


class CNN_ccp(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×２＋プーリング層、全結合層なし
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層1
        M1      = kwargs.pop('M1',      16)          # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2) # フィルタ高
        stride1 = kwargs.pop('stride1',  1)          # ストライド
        cl1_pad = kwargs.pop('cl1_pad',  0)          # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')    # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')    # 最適化関数
        opt_for_cl1['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl1['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 畳込み層2
        M2      = kwargs.pop('M2',      16)          # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2) # フィルタ高
        stride2 = kwargs.pop('stride2',  1)          # ストライド
        cl2_pad = kwargs.pop('cl2_pad',  0)          # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')    # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')    # 最適化関数
        opt_for_cl2['batchnorm'] = kwargs.pop('bn',      False)    # bn は中間層でも使う
        opt_for_cl2['layernorm'] = kwargs.pop('ln',      False)    # ln は中間層でも使う
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',      None)
        # layer3 プーリング層
        pool   = kwargs.pop('pool',           2)     # プーリング
        pl_pad = kwargs.pop('pl_pad',         0)     # パディング
        method = kwargs.pop('method',     'max')     # プーリングメソッド 
        # layer4 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['optimize']  = kwargs.pop('ol_opt', 'SGD')     # 最適化関数
        opt_for_ol['spctrnorm'] = kwargs.pop('sn',      None)
        opt_for_ol['full_connection'] = True 
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        # layer 1
        self.conv_layer1   = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        # layer 2
        self.conv_layer2   = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        # layer 3
        self.pooling_layer = neuron.Pooling2dLayer(pool, pl_pad, method)
        # layer 4
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.pooling_layer)
        self.layers.append(self.output_layer)


class CNN_cccp(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×3＋プーリング層、全結合層なし
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層1
        M1      = kwargs.pop('M1',      12)          # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2) # フィルタ高
        stride1 = kwargs.pop('stride1',  1)          # ストライド
        cl1_pad = kwargs.pop('cl1_pad',  0)          # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')    # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')    # 最適化関数
        opt_for_cl1['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl1['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 畳込み層2
        M2      = kwargs.pop('M2',      12)          # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2) # フィルタ高
        stride2 = kwargs.pop('stride2',  1)          # ストライド
        cl2_pad = kwargs.pop('cl2_pad',  0)          # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')    # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')    # 最適化関数
        opt_for_cl2['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl2['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',      None)
        # layer3 畳込み層3
        M3      = kwargs.pop('M3',      12)          # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2) # フィルタ高
        stride3 = kwargs.pop('stride3',  1)          # ストライド
        cl3_pad = kwargs.pop('cl3_pad',  0)          # パディング
        opt_for_cl3 = {}
        opt_for_cl3['activate']  = kwargs.pop('cl3_act', 'ReLU')    # 活性化関数 
        opt_for_cl3['optimize']  = kwargs.pop('cl3_opt', 'Adam')    # 最適化関数
        opt_for_cl3['batchnorm'] = kwargs.pop('bn',      False)    # bn は中間層でも使う
        opt_for_cl3['layernorm'] = kwargs.pop('ln',      False)    # ln は中間層でも使う
        opt_for_cl3['spctrnorm'] = kwargs.get('sn',      None)
        # layer4 プーリング層
        pool   = kwargs.pop('pool',           2)     # プーリング
        pl_pad = kwargs.pop('pl_pad',         0)     # パディング
        method = kwargs.pop('method',     'max')     # プーリングメソッド 
        # layer5 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['optimize']  = kwargs.pop('ol_opt', 'SGD')     # 最適化関数
        opt_for_ol['spctrnorm'] = kwargs.pop('sn',      None)
        opt_for_ol['full_connection'] = True 
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_cl3.update(kwargs) 
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        # layer 1
        self.conv_layer1   = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        # layer 2
        self.conv_layer2   = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        # layer 3
        self.conv_layer3   = neuron.Conv2dLayer(M3, kernel_size3, stride3, cl3_pad, **opt_for_cl3)
        # layer 4
        self.pooling_layer = neuron.Pooling2dLayer(pool, pl_pad, method)
        # layer 5
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.conv_layer3)
        self.layers.append(self.pooling_layer)
        self.layers.append(self.output_layer)


class CNN_ccccp(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×4＋プーリング層、全結合層なし
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層1
        M1      = kwargs.pop('M1',      12)          # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2) # フィルタ高
        stride1 = kwargs.pop('stride1',  1)          # ストライド
        cl1_pad = kwargs.pop('cl1_pad',  0)          # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')    # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')    # 最適化関数
        opt_for_cl1['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl1['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 畳込み層2
        M2      = kwargs.pop('M2',      12)          # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2) # フィルタ高
        stride2 = kwargs.pop('stride2',  1)          # ストライド
        cl2_pad = kwargs.pop('cl2_pad',  0)          # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')    # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')    # 最適化関数
        opt_for_cl2['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl2['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',      None)
        # layer3 畳込み層3
        M3      = kwargs.pop('M3',      12)          # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2) # フィルタ高
        stride3 = kwargs.pop('stride3',  1)          # ストライド
        cl3_pad = kwargs.pop('cl3_pad',  0)          # パディング
        opt_for_cl3 = {}
        opt_for_cl3['activate']  = kwargs.pop('cl3_act', 'ReLU')    # 活性化関数 
        opt_for_cl3['optimize']  = kwargs.pop('cl3_opt', 'Adam')    # 最適化関数
        opt_for_cl3['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl3['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl3['spctrnorm'] = kwargs.get('sn',      None)
        # layer4 畳込み層4
        M4      = kwargs.pop('M4',      12)          # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 2) # フィルタ高
        stride4 = kwargs.pop('stride4',  1)          # ストライド
        cl4_pad = kwargs.pop('cl4_pad',  0)          # パディング
        opt_for_cl4 = {}
        opt_for_cl4['activate']  = kwargs.pop('cl4_act', 'ReLU')    # 活性化関数 
        opt_for_cl4['optimize']  = kwargs.pop('cl4_opt', 'Adam')    # 最適化関数
        opt_for_cl4['batchnorm'] = kwargs.pop('bn',      False)    # bn は中間層でも使う
        opt_for_cl4['layernorm'] = kwargs.pop('ln',      False)    # ln は中間層でも使う
        opt_for_cl4['spctrnorm'] = kwargs.get('sn',      None)
        # layer5 プーリング層
        pool   = kwargs.pop('pool',           2)     # プーリング
        pl_pad = kwargs.pop('pl_pad',         0)     # パディング
        method = kwargs.pop('method',     'max')     # プーリングメソッド 
        # layer6 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['optimize']  = kwargs.pop('ol_opt', 'SGD')     # 最適化関数
        opt_for_ol['spctrnorm'] = kwargs.pop('sn',      None)
        opt_for_ol['full_connection'] = True 
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_cl3.update(kwargs) 
        opt_for_cl4.update(kwargs) 
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        # layer 1
        self.conv_layer1   = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        # layer 2
        self.conv_layer2   = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        # layer 3
        self.conv_layer3   = neuron.Conv2dLayer(M3, kernel_size3, stride3, cl3_pad, **opt_for_cl3)
        # layer 4
        self.conv_layer4   = neuron.Conv2dLayer(M4, kernel_size4, stride4, cl4_pad, **opt_for_cl4)
        # layer 5
        self.pooling_layer = neuron.Pooling2dLayer(pool, pl_pad, method)
        # layer 6
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_ol); print('###debug', Out)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.conv_layer3)
        self.layers.append(self.conv_layer4)
        self.layers.append(self.pooling_layer)
        self.layers.append(self.output_layer)


class CNN_ccpccgp(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×4＋グローバルプーリング層
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層1
        M1      = kwargs.pop('M1',     32)            # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2)  # フィルタ高
        stride1 = kwargs.pop('stride1', 1)            # ストライド
        cl1_pad = kwargs.pop('cl1_pad', 0)            # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')    # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')    # 最適化関数
        opt_for_cl1['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl1['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 畳込み層2
        M2      = kwargs.pop('M2',     32)            # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2)  # フィルタ高
        stride2 = kwargs.pop('stride2', 1)            # ストライド
        cl2_pad = kwargs.pop('cl2_pad', 0)            # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')    # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')    # 最適化関数
        opt_for_cl2['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl2['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',      None)
        # layer3 プーリング層
        pool   = kwargs.pop('pool',       2)          # プーリング
        pl_pad = kwargs.pop('pl_pad',     0)          # パディング
        method = kwargs.pop('method', 'max')          # プーリングメソッド 
        # layer4 畳込み層3
        M3      = kwargs.pop('M3',     64)            # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2)  # フィルタ高
        stride3 = kwargs.pop('stride3', 1)            # ストライド
        cl3_pad = kwargs.pop('cl3_pad', 0)            # パディング
        opt_for_cl3 = {}
        opt_for_cl3['activate']  = kwargs.pop('cl3_act', 'ReLU')   # 活性化関数 
        opt_for_cl3['optimize']  = kwargs.pop('cl3_opt', 'Adam')   # 最適化関数
        opt_for_cl3['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl3['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl3['spctrnorm'] = kwargs.get('sn',      None)
        # layer5 畳込み層4
        M4      = kwargs.pop('M4',     64)            # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 2)  # フィルタ高
        stride4 = kwargs.pop('stride4', 1)            # ストライド
        cl4_pad = kwargs.pop('cl4_pad', 0)            # パディング
        opt_for_cl4 = {}
        opt_for_cl4['activate']  = kwargs.pop('cl4_act', 'ReLU')   # 活性化関数 
        opt_for_cl4['optimize']  = kwargs.pop('cl4_opt', 'Adam')   # 最適化関数
        opt_for_cl4['batchnorm'] = kwargs.pop('bn',      False)    # bn は中間層でも使う
        opt_for_cl4['layernorm'] = kwargs.pop('ln',      False)    # ln は中間層でも使う
        opt_for_cl4['spctrnorm'] = kwargs.get('sn',      None)
        # layer6 グローバルプーリング層
        # layer7 全結合出力層
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']   = kwargs.pop('ol_act', ol_act_cand) #活性化関数
        opt_for_ol['optimize']   = kwargs.pop('ol_opt', 'SGD')     #最適化関数
        opt_for_ol['spctrnorm']  = kwargs.pop('sn',      None)
        
        # kwargs に残ったものを結合
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_cl3.update(kwargs) 
        opt_for_cl4.update(kwargs) 
        opt_for_ol.update(kwargs)  
        opt_for_ol['full_connection'] = True
        # -- 各層の初期化 -- 
        self.conv_layer1    = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        self.conv_layer2    = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        self.pooling_layer  = neuron.Pooling2dLayer(pool, pl_pad, method)
        self.conv_layer3    = neuron.Conv2dLayer(M3, kernel_size3, stride3, cl3_pad, **opt_for_cl3)
        self.conv_layer4    = neuron.Conv2dLayer(M4, kernel_size4, stride4, cl4_pad, **opt_for_cl4)
        self.global_pooling = neuron.GlobalAveragePooling()
        self.output_layer   = neuron.NeuronLayer(Out, **opt_for_ol); print('###debug', Out)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.pooling_layer)
        self.layers.append(self.conv_layer3)
        self.layers.append(self.conv_layer4)
        self.layers.append(self.global_pooling)
        self.layers.append(self.output_layer)

class CNN_cccc(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×4＋プーリング層、全結合層なし
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層1
        M1      = kwargs.pop('M1',      16)          # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2) # フィルタ高
        stride1 = kwargs.pop('stride1',  1)          # ストライド
        cl1_pad = kwargs.pop('cl1_pad',  0)          # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')    # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')    # 最適化関数
        opt_for_cl1['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl1['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 畳込み層2
        M2      = kwargs.pop('M2',      16)          # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2) # フィルタ高
        stride2 = kwargs.pop('stride2',  1)          # ストライド
        cl2_pad = kwargs.pop('cl2_pad',  0)          # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')    # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')    # 最適化関数
        opt_for_cl2['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl2['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',      None)
        # layer3 畳込み層3
        M3      = kwargs.pop('M3',      16)          # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2) # フィルタ高
        stride3 = kwargs.pop('stride3',  1)          # ストライド
        cl3_pad = kwargs.pop('cl3_pad',  0)          # パディング
        opt_for_cl3 = {}
        opt_for_cl3['activate']  = kwargs.pop('cl3_act', 'ReLU')    # 活性化関数 
        opt_for_cl3['optimize']  = kwargs.pop('cl3_opt', 'Adam')    # 最適化関数
        opt_for_cl3['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl3['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl3['spctrnorm'] = kwargs.get('sn',      None)
        # layer4 畳込み層4
        M4      = kwargs.pop('M4',      16)          # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 2) # フィルタ高
        stride4 = kwargs.pop('stride4',  1)          # ストライド
        cl4_pad = kwargs.pop('cl4_pad',  0)          # パディング
        opt_for_cl4 = {}
        opt_for_cl4['activate']  = kwargs.pop('cl4_act', 'ReLU')    # 活性化関数 
        opt_for_cl4['optimize']  = kwargs.pop('cl4_opt', 'Adam')    # 最適化関数
        opt_for_cl4['batchnorm'] = kwargs.pop('bn',      False)    # bn は中間層でも使う
        opt_for_cl4['layernorm'] = kwargs.pop('ln',      False)    # ln は中間層でも使う
        opt_for_cl4['spctrnorm'] = kwargs.get('sn',      None)
        # layer5 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['optimize']  = kwargs.pop('ol_opt', 'SGD')     # 最適化関数
        opt_for_ol['spctrnorm'] = kwargs.pop('sn',      None)
        opt_for_ol['full_connection'] = True 
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_cl3.update(kwargs) 
        opt_for_cl4.update(kwargs) 
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        # layer 1
        self.conv_layer1   = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        # layer 2
        self.conv_layer2   = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        # layer 3
        self.conv_layer3   = neuron.Conv2dLayer(M3, kernel_size3, stride3, cl3_pad, **opt_for_cl3)
        # layer 4
        self.conv_layer4   = neuron.Conv2dLayer(M4, kernel_size4, stride4, cl4_pad, **opt_for_cl4)
        # layer 5
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.conv_layer3)
        self.layers.append(self.conv_layer4)
        self.layers.append(self.output_layer)

class CNN_ccpccp(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×4＋プーリング層、全結合層なし
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層1
        M1      = kwargs.pop('M1',      16)          # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2) # フィルタ高
        stride1 = kwargs.pop('stride1',  1)          # ストライド
        cl1_pad = kwargs.pop('cl1_pad',  0)          # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')    # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')    # 最適化関数
        opt_for_cl1['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl1['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 畳込み層2
        M2      = kwargs.pop('M2',      16)          # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2) # フィルタ高
        stride2 = kwargs.pop('stride2',  1)          # ストライド
        cl2_pad = kwargs.pop('cl2_pad',  0)          # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')    # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')    # 最適化関数
        opt_for_cl2['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl2['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',      None)
        # layer3 プーリング層
        pool1   = kwargs.pop('pool1',         2)     # プーリング
        pl1_pad = kwargs.pop('pl1_pad',       0)     # パディング
        method1 = kwargs.get('method',    'max')     # プーリングメソッド 
        # layer4 畳込み層3
        M3      = kwargs.pop('M3',      24)          # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2) # フィルタ高
        stride3 = kwargs.pop('stride3',  1)          # ストライド
        cl3_pad = kwargs.pop('cl3_pad',  0)          # パディング
        opt_for_cl3 = {}
        opt_for_cl3['activate']  = kwargs.pop('cl3_act', 'ReLU')    # 活性化関数 
        opt_for_cl3['optimize']  = kwargs.pop('cl3_opt', 'Adam')    # 最適化関数
        opt_for_cl3['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl3['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl3['spctrnorm'] = kwargs.get('sn',      None)
        # layer5 畳込み層4
        M4      = kwargs.pop('M4',      24)          # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 2) # フィルタ高
        stride4 = kwargs.pop('stride4',  1)          # ストライド
        cl4_pad = kwargs.pop('cl4_pad',  0)          # パディング
        opt_for_cl4 = {}
        opt_for_cl4['activate']  = kwargs.pop('cl4_act', 'ReLU')    # 活性化関数 
        opt_for_cl4['optimize']  = kwargs.pop('cl4_opt', 'Adam')    # 最適化関数
        opt_for_cl4['batchnorm'] = kwargs.pop('bn',      False)    # bn は中間層でも使う
        opt_for_cl4['layernorm'] = kwargs.pop('ln',      False)    # ln は中間層でも使う
        opt_for_cl4['spctrnorm'] = kwargs.get('sn',      None)
        # layer6 プーリング層
        pool2   = kwargs.pop('pool2',         2)    # プーリング
        pl2_pad = kwargs.pop('pl2_pad',       0)    # パディング
        method2 = kwargs.pop('method',    'max')    # プーリングメソッド 
        # layer7 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['optimize']  = kwargs.pop('ol_opt', 'SGD')     # 最適化関数
        opt_for_ol['spctrnorm'] = kwargs.pop('sn',      None)
        opt_for_ol['full_connection'] = True 
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_cl3.update(kwargs) 
        opt_for_cl4.update(kwargs) 
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        # layer 1
        self.conv_layer1   = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        # layer 2
        self.conv_layer2   = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        # layer 3
        self.pooling_layer1 = neuron.Pooling2dLayer(pool1, pl1_pad, method1)
        # layer 4
        self.conv_layer3   = neuron.Conv2dLayer(M3, kernel_size3, stride3, cl3_pad, **opt_for_cl3)
        # layer 5
        self.conv_layer4   = neuron.Conv2dLayer(M4, kernel_size4, stride4, cl4_pad, **opt_for_cl4)
        # layer 6
        self.pooling_layer2 = neuron.Pooling2dLayer(pool2, pl2_pad, method2)
        # layer 5
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.pooling_layer1)
        self.layers.append(self.conv_layer3)
        self.layers.append(self.conv_layer4)
        self.layers.append(self.pooling_layer2)
        self.layers.append(self.output_layer)

class CNN_ccpccpccp(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×4＋プーリング層、全結合層なし
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer1 畳込み層1
        M1      = kwargs.pop('M1',      16)          # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2) # フィルタ高
        stride1 = kwargs.pop('stride1',  1)          # ストライド
        cl1_pad = kwargs.pop('cl1_pad',  0)          # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')    # 活性化関数 
        opt_for_cl1['optimize']  = kwargs.pop('cl1_opt', 'Adam')    # 最適化関数
        opt_for_cl1['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl1['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl1['spctrnorm'] = kwargs.get('sn',      None)
        # layer2 畳込み層2
        M2      = kwargs.pop('M2',      16)          # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2) # フィルタ高
        stride2 = kwargs.pop('stride2',  1)          # ストライド
        cl2_pad = kwargs.pop('cl2_pad',  0)          # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')    # 活性化関数 
        opt_for_cl2['optimize']  = kwargs.pop('cl2_opt', 'Adam')    # 最適化関数
        opt_for_cl2['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl2['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl2['spctrnorm'] = kwargs.get('sn',      None)
        # layer3 プーリング層1
        pool1   = kwargs.pop('pool1',         2)     # プーリング
        pl1_pad = kwargs.pop('pl1_pad',       0)     # パディング
        method1 = kwargs.get('method',    'max')     # プーリングメソッド 
        # layer4 畳込み層3
        M3      = kwargs.pop('M3',      24)          # フィルタ数
        kernel_size3 = kwargs.pop('kernel_size3', 2) # フィルタ高
        stride3 = kwargs.pop('stride3',  1)          # ストライド
        cl3_pad = kwargs.pop('cl3_pad',  0)          # パディング
        opt_for_cl3 = {}
        opt_for_cl3['activate']  = kwargs.pop('cl3_act', 'ReLU')    # 活性化関数 
        opt_for_cl3['optimize']  = kwargs.pop('cl3_opt', 'Adam')    # 最適化関数
        opt_for_cl3['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl3['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl3['spctrnorm'] = kwargs.get('sn',      None)
        # layer5 畳込み層4
        M4      = kwargs.pop('M4',      24)          # フィルタ数
        kernel_size4 = kwargs.pop('kernel_size4', 2) # フィルタ高
        stride4 = kwargs.pop('stride4',  1)          # ストライド
        cl4_pad = kwargs.pop('cl4_pad',  0)          # パディング
        opt_for_cl4 = {}
        opt_for_cl4['activate']  = kwargs.pop('cl4_act', 'ReLU')    # 活性化関数 
        opt_for_cl4['optimize']  = kwargs.pop('cl4_opt', 'Adam')    # 最適化関数
        opt_for_cl4['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl4['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl4['spctrnorm'] = kwargs.get('sn',      None)
        # layer6 プーリング層2
        pool2   = kwargs.pop('pool2',         2)    # プーリング
        pl2_pad = kwargs.pop('pl2_pad',       0)    # パディング
        method2 = kwargs.pop('method',    'max')    # プーリングメソッド 
        # layer7 畳込み層5
        M5      = kwargs.pop('M5',      32)          # フィルタ数
        kernel_size5 = kwargs.pop('kernel_size5', 2) # フィルタ高
        stride5 = kwargs.pop('stride5',  1)          # ストライド
        cl5_pad = kwargs.pop('cl5_pad',  0)          # パディング
        opt_for_cl5 = {}
        opt_for_cl5['activate']  = kwargs.pop('cl5_act', 'ReLU')    # 活性化関数 
        opt_for_cl5['optimize']  = kwargs.pop('cl5_opt', 'Adam')    # 最適化関数
        opt_for_cl5['batchnorm'] = kwargs.get('bn',      False)    # bn は中間層でも使う
        opt_for_cl5['layernorm'] = kwargs.get('ln',      False)    # ln は中間層でも使う
        opt_for_cl5['spctrnorm'] = kwargs.get('sn',      None)
        # layer8 畳込み層6
        M6      = kwargs.pop('M6',      32)          # フィルタ数
        kernel_size6 = kwargs.pop('kernel_size6', 2) # フィルタ高
        stride6 = kwargs.pop('stride6',  1)          # ストライド
        cl6_pad = kwargs.pop('cl6_pad',  0)          # パディング
        opt_for_cl6 = {}
        opt_for_cl6['activate']  = kwargs.pop('cl6_act', 'ReLU')    # 活性化関数 
        opt_for_cl6['optimize']  = kwargs.pop('cl6_opt', 'Adam')    # 最適化関数
        opt_for_cl6['batchnorm'] = kwargs.pop('bn',      False)    # bn は中間層でも使う
        opt_for_cl6['layernorm'] = kwargs.pop('ln',      False)    # ln は中間層でも使う
        opt_for_cl6['spctrnorm'] = kwargs.get('sn',      None)
        # layer9 プーリング層3
        pool3   = kwargs.pop('pool3',         2)    # プーリング
        pl3_pad = kwargs.pop('pl3_pad',       0)    # パディング
        method3 = kwargs.pop('method',    'max')    # プーリングメソッド 
        # layer10 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['optimize']  = kwargs.pop('ol_opt', 'SGD')     # 最適化関数
        opt_for_ol['spctrnorm'] = kwargs.pop('sn',      None)
        opt_for_ol['full_connection'] = True 
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_cl3.update(kwargs) 
        opt_for_cl4.update(kwargs) 
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        # layer 1
        self.conv_layer1   = neuron.Conv2dLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        # layer 2
        self.conv_layer2   = neuron.Conv2dLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        # layer 3
        self.pooling_layer1 = neuron.Pooling2dLayer(pool1, pl1_pad, method1)
        # layer 4
        self.conv_layer3   = neuron.Conv2dLayer(M3, kernel_size3, stride3, cl3_pad, **opt_for_cl3)
        # layer 5
        self.conv_layer4   = neuron.Conv2dLayer(M4, kernel_size4, stride4, cl4_pad, **opt_for_cl4)
        # layer 6
        self.pooling_layer2 = neuron.Pooling2dLayer(pool2, pl2_pad, method2)
        # layer 7
        self.conv_layer5   = neuron.Conv2dLayer(M5, kernel_size5, stride5, cl5_pad, **opt_for_cl5)
        # layer 8
        self.conv_layer6   = neuron.Conv2dLayer(M6, kernel_size6, stride6, cl6_pad, **opt_for_cl6)
        # layer 9
        self.pooling_layer3 = neuron.Pooling2dLayer(pool3, pl3_pad, method3)
        # layer 10
        self.output_layer  = neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.pooling_layer1)
        self.layers.append(self.conv_layer3)
        self.layers.append(self.conv_layer4)
        self.layers.append(self.pooling_layer2)
        self.layers.append(self.conv_layer5)
        self.layers.append(self.conv_layer6)
        self.layers.append(self.pooling_layer3)
        self.layers.append(self.output_layer)



if __name__=='__main__':
    print('\n#### all cast ####')
    import inspect
    import sys
    current_module = sys.modules[__name__]
    classes = map(lambda x:x[0],inspect.getmembers(current_module,inspect.isclass))
    classes = list(classes)
    for c in classes:
        print(c)
        if c not in('NN_CNN_Base') and c[-4:]!='bkup':
            model = eval(c)()
