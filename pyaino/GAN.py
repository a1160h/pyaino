# GAN 各種
# 2025.10.15 A.Inoue　
from pyaino.Config import *
from pyaino import common_function as cf
from pyaino import LossFunctions as lf

def set_dtype(value):
    ''' neuronのwやbの精度指定 '''
    neuron.set_dtype(value)

# -- 各層の初期化 --
class GANBase:
    def __init__(self, **kwargs):
        self.gen = None
        self.dsc = None
        self.title = None

        # 損失関数 
        loss    = kwargs.pop('loss',    None)
        sumup   = kwargs.pop('sumup',  False) # 誤差の勾配算出の際にバッチ内で足しこむ
        enhance = kwargs.pop('enhance',    1) # 誤差の勾配算出の際にかける倍率
        if loss=='MM2':   # NS
            self.loss_function = lf.LossFunctionForGAN2(sumup, enhance)
        elif loss=='MM3': # basic + NS
            self.loss_function = lf.LossFunctionForGAN3(sumup, enhance)
        elif loss=='MM4': # Wasserstein
            self.loss_function = lf.LossFunctionForGAN4(sumup, enhance)
        else:                 # basic mini-max
            self.loss_function = lf.LossFunctionForGAN(sumup, enhance)
        # kwargsをgenとdscに振分ける
        self.opt_for_gen, self.opt_for_dsc = {}, {}
        keys_kwargs = list(kwargs.keys()) # iterator(ループ内変更不可)
        for k in keys_kwargs: 
            if k[:4]=='gen_':
                self.opt_for_gen[k[4:]]=kwargs.pop(k) 
            if k[:4]=='dsc_':
                self.opt_for_dsc[k[4:]]=kwargs.pop(k)
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        self.opt_for_gen.update(kwargs)
        self.opt_for_dsc.update(kwargs) 
        print('options for gen', self.opt_for_gen)
        print('options for dsc', self.opt_for_dsc)
                                    
    # -- 順伝播 --
    def gen_forward(self, x, **kwargs):
        return self.gen.forward(x, **kwargs)

    def dsc_forward(self, x, t=None, **kwargs):
        y = self.dsc.forward(x, **kwargs)
        if t is None:
            return y
        else:
            l = self.loss_function.forward(y, t)
            return y, l 
    
    def dsc_loss(self, y, t): # 旧版互換
        return self.loss_function.forward(y, t)

    def loss_forward(self, y, t):
        return self.loss_function.forward(y, t)

    def loss_backward(self):
        return self.loss_function.backward()

    def loss_forward_gen(self, y): 
        return self.loss_function.forward_for_gen(y)

    def loss_backward_gen(self): 
        return self.loss_function.backward_for_gen()

    def get_accuracy(self, y, t):
        correct = np.sum(np.where(y<0.5, 0, 1) == t)
        return correct / len(y)

    # -- 逆伝播 --
    def dsc_backward(self, gy=None, gl=1):
        if gy is not None:
            pass
        elif self.loss_function.t is not None:
            gy = self.loss_function.backward(gl)
        else:
            raise Exception("Can't get gradient by backward.")
        gy = self.dsc.backward(gy)
        return gy

    def gen_backward(self, gy):
        return self.gen.backward(gy)
    
    # -- パラメータの更新 --
    def gen_update(self, eta, **kwargs):
        self.gen.update(eta=eta, **kwargs)

    def dsc_update(self, eta, **kwargs):
        self.dsc.update(eta=eta, **kwargs)
    
    def dsc_backup(self):
        self.dsc.backup()
    
    def dsc_recover(self):
        self.dsc.recover()
    
    def summary(self):
        print('～'*39)
        print(self.__class__.__name__)
        self.gen.summary()
        self.dsc.summary()
        print('loss_function =', self.loss_function.__class__.__name__, end=' (')
        print('sumup =', self.loss_function.sumup, end='  ')
        print('enhance =', self.loss_function.enhance, ')\n')
        

    def export_params(self):
        params = {}
        gen_params = self.gen.export_params()
        dsc_params = self.dsc.export_params()
        for k in gen_params.keys():
            params['gen_'+k] = gen_params[k]
        for k in dsc_params.keys():
            params['dsc_'+k] = dsc_params[k] 
        return params
    
    def import_params(self, params):
        gen_params, dsc_params = {}, {}
        print('source', params.keys())
        for k in params.keys():
            if k[:4]=='gen_':
                gen_params[k[4:]]=params[k]
            if k[:4]=='dsc_':
                dsc_params[k[4:]]=params[k]
        self.gen.import_params(gen_params)
        self.dsc.import_params(dsc_params)
        print('gen', gen_params.keys())
        print('dsc', dsc_params.keys())

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
    
# -- 外部から直接アクセス(共通) --
def build(GAN_name, n, **kwargs):
    print(GAN_name, n, kwargs)
    global model
    model = GAN_name(n, **kwargs)

def gen_forward(x, **kwargs):
    return model.gen_forward(x, **kwargs)

def dsc_forward(x, t=None, **kwargs):
    return model.dsc_forward(x, t, **kwargs)

def dsc_loss(y, t): # 旧版互換
    return model.loss_forward(y, t)

def loss_forward(y, t):
    return model.loss_forward(y, t)

def loss_backward():
    return model.loss_backward()

def loss_forward_gen(y):
    return model.loss_forward_gen(y)

def loss_backward_gen():
    return model.loss_backward_gen()

def dsc_backward(gy=None, gl=1):
    return model.dsc_backward(gy, gl)

def gen_backward(gy):
    return model.gen_backward(gy)

def gen_update(eta, **kwargs):
    model.gen_update(eta, **kwargs)

def dsc_update(eta, **kwargs):
    model.dsc_update(eta, **kwargs)

def dsc_backup():
    model.dsc_backup()
   
def dsc_recover():
    model.dsc_recover()
   
# -- 学習結果の保存 --
def save_parameters(file_name):
    model.save_parameters(file_name)

# -- 学習結果の継承 --
def load_parameters(file_name):
    return model.load_parameters(file_name) 

# -- 誤差を計算 --
def get_error(y, t):
    return model.dsc_loss(y, t)

# -- 正解率を計算 --
def get_accuracy(y, t):
    return model.get_accuracy(y, t)

def summary():
    model.summary()

#### GAN各モデル #### 
from pyaino import NN_CNN
class GAN_m_m(GANBase):
    def __init__(self, n, gml=256, dml=256, **kwargs):
        '''
        Geneaotor    : 中間層 + 出力層
        dscriminator : 中間層 + 出力層
        n: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        gml: gen 中間層の大きさ
        dml: dsc 中間層の大きさ
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_m_m'
        self.gen = NN_CNN.NN_m(n, ml_nn=gml, \
                                 ml_act='ReLU', ml_opt='RMSProp', \
                                 ol_act='Tanh', ol_opt='RMSProp', \
                                 **self.opt_for_gen)
        
        self.dsc = NN_CNN.NN_m(1, ml_nn=dml, \
                                 full_connection=True,      
                                 ml_act='LReLU', ml_opt='RMSProp', \
                                 ol_act='Sigmoid', ol_opt='SGD', \
                                 **self.opt_for_dsc)
class GAN_mm_mm(GANBase):
    def __init__(self, n, gml1=64, gml2=128, dml1=128, dml2=64, **kwargs):
        '''
        Geneaotor    : 中間層 + 中間層 + 出力層
        dscriminator : 中間層 + 中間層 + 出力層
        n: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        gml: gen 中間層の大きさ
        dml: dsc 中間層の大きさ
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mm_mm'
        self.gen = NN_CNN.NN_mm(n, ml1_nn=gml1, ml2_nn=gml2, \
                                 ml1_act='ReLU', ml1_opt='RMSProp',\
                                 ml2_act='ReLU', ml2_opt='RMSProp',\
                                 ol_act='Tanh', ol_opt='RMSProp', \
                                 **self.opt_for_gen)
        
        self.dsc = NN_CNN.NN_mm(1, ml1_nn=dml1, ml2_nn=dml2, \
                                 full_connection=True,      
                                 ml1_act='LReLU', ml1_opt='RMSProp', \
                                 ml2_act='LReLU', ml2_opt='RMSProp', \
                                 ol_act='Sigmoid', ol_opt='SGD', \
                                 **self.opt_for_dsc)
                                     
      
class GAN_mdd_cpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋出力層
        dscriminator : 畳込み層＋プーリング層＋中間層＋出力層
        C, Ih, Iw: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mdd_cpm'
        self.gen = NN_CNN.CNN_mdd(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_cpm(1 \
                                 , M =16, ml_nn=64
                                 , cl_act='Mish', cl_opt='RMSProp'
                                 , ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mdc_cpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋畳込み層＋出力層
        dscriminator : 畳込み層＋プーリング層＋中間層＋出力層
        C, Ih, Iw: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mdc_cpm'
        self.gen = NN_CNN.CNN_mdc(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_cpm(1 \
                                 , M =16, ml_nn=64
                                 , cl_act='Mish', cl_opt='RMSProp'
                                 , ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mud_cpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋逆畳込み層＋出力層
        dscriminator : 畳込み層＋畳込み層＋プーリング層＋中間層＋出力層
        C, Ih, Iw: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mud_cpm'
        self.gen = NN_CNN.CNN_mud(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_cpm(1 \
                                 , M =16, ml_nn=64
                                 , cl_act='Mish', cl_opt='RMSProp'
                                 , ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mddc_cpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋畳込み層＋出力層
        dscriminator : 畳込み層＋プーリング層＋中間層＋出力層
        C, Ih, Iw: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mddc_cpm'
        self.gen = NN_CNN.CNN_mddc(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_cpm(1 \
                                 , M =16, ml_nn=64
                                 , cl_act='Mish', cl_opt='RMSProp'
                                 , ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mddc_ccpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋畳込み層＋出力層
        dscriminator : 畳込み層＋畳込み層＋プーリング層＋中間層＋出力層
        C, Ih, Iw: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mddc_ccpm'
        self.gen = NN_CNN.CNN_mddc(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_ccpm(1 \
                                 , M1 =16, M2 =16, ml_nn=64
                                 , cl1_act='Mish', cl1_opt='RMSProp'
                                 , cl2_act='Mish', cl2_opt='RMSProp'
                                 , ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mddd_ccpm2(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋逆畳込み層＋出力層
        dscriminator : 畳込み層＋畳込み層＋プーリング層＋中間層＋出力層
        C, Ih, Iw: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mddd_ccpm'
        self.gen = NN_CNN.CNN_mddd(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_ccpm(1 \
                                 , M1 =16, M2 =16, ml_nn=64
                                 , cl1_act='Mish', cl1_opt='RMSProp'
                                 , cl2_act='Mish', cl2_opt='RMSProp'
                                 , ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Identity', ol_opt='RMSProp'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mddd_ccpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋逆畳込み層＋出力層
        dscriminator : 畳込み層＋畳込み層＋プーリング層＋中間層＋出力層
        C, Ih, Iw: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mddd_ccpm'
        self.gen = NN_CNN.CNN_mddd(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_ccpm(1 \
                                 , M1 =16, M2 =16, ml_nn=64
                                 , cl1_act='Mish', cl1_opt='RMSProp'
                                 , cl2_act='Mish', cl2_opt='RMSProp'
                                 , ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mdddc_ccpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋逆畳込み層＋畳込み層＋出力層
        dscriminator : 畳込み層＋畳込み層＋プーリング層＋中間層＋出力層
        n: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mdddc_ccpm'
        self.gen = NN_CNN.CNN_mdddc(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_ccpm(1 \
                                 , M1 =16, M2 =16, ml_nn=64
                                 , cl1_act='Mish', cl1_opt='RMSProp'
                                 , cl2_act='Mish', cl2_opt='RMSProp'
                                 , ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mddddc_ccpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋逆畳込み層＋逆畳込み層
                     ＋畳込み層＋出力層
        dscriminator : 畳込み層＋畳込み層＋プーリング層＋中間層＋出力層
        n: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mddddc_ccpm'
        self.gen = NN_CNN.CNN_mddddc(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_ccpm(1 \
                                 , M1 =16, M2 =16, ml_nn=64
                                 , cl1_act='Mish', cl1_opt='RMSProp'
                                 , cl2_act='Mish', cl2_opt='RMSProp'
                                 , ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mddddc_cccpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋逆畳込み層＋逆畳込み層
                     ＋畳込み層＋出力層
        dscriminator : 畳込み層＋畳込み層＋畳込み層＋プーリング層＋中間層＋出力層
        C, Ih, Iw: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mddddc_cccpm'
        self.gen = NN_CNN.CNN_mddddc(n \
                                 , **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_cccpm(1 \
                                 , M1=12, M2=12, M3=12, ml_nn=64                 
                                 , cl1_act='Mish', cl1_opt='RMSProp'
                                 , cl2_act='Mish', cl2_opt='RMSProp'
                                 , cl3_act='Mish', cl3_opt='RMSProp'
                                 , ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mdddcc_ccpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋逆畳込み層＋逆畳込み層
                     ＋畳込み層＋出力層
        dscriminator : 畳込み層＋畳込み層＋プーリング層＋中間層＋出力層
        n: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mddddc_ccpm'
        self.gen = NN_CNN.CNN_mdddcc(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_ccpm(1 \
                                 , M1 =16, M2 =16, ml_nn=64
                                 , cl1_act='Mish', cl1_opt='RMSProp'
                                 , cl2_act='Mish', cl2_opt='RMSProp'
                                 , ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)
class GAN_mdcdc_ccpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋逆畳込み層＋逆畳込み層
                     ＋畳込み層＋出力層
        dscriminator : 畳込み層＋畳込み層＋プーリング層＋中間層＋出力層
        n: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mdcdc_ccpm'
        self.gen = NN_CNN.CNN_mdcdc(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_ccpm(1 \
                                 , M1 =16, M2 =16, ml_nn=64
                                 , cl1_act='Mish', cl1_opt='RMSProp'
                                 , cl2_act='Mish', cl2_opt='RMSProp'
                                 , ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mddddc_ccpccpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋逆畳込み層＋逆畳込み層
                     ＋畳込み層＋出力層
        dscriminator : 畳込み層＋畳込み層＋プーリング層＋畳込み層＋畳込み層＋プーリング層
                     ＋中間層＋出力層
        n: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mddddc_ccpccpm'
        self.gen = NN_CNN.CNN_mddddc(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_ccpccpm(1 \
                                 , cl1_act='Mish', cl1_opt='RMSProp'
                                 , cl2_act='Mish', cl2_opt='RMSProp'
                                 , cl3_act='Mish', cl3_opt='RMSProp'
                                 , cl4_act='Mish', cl4_opt='RMSProp'
                                 , ml_nn=64, ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mddddc_ccpccpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋逆畳込み層＋逆畳込み層
                     ＋畳込み層＋出力層
        dscriminator : 畳込み層＋畳込み層＋プーリング層＋畳込み層＋畳込み層＋プーリング層
                     ＋中間層＋出力層
        n: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mddddc_ccpccpm'
        self.gen = NN_CNN.CNN_mddddc(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_ccpccpm(1 \
                                 , cl1_act='Mish', cl1_opt='RMSProp'
                                 , cl2_act='Mish', cl2_opt='RMSProp'
                                 , cl3_act='Mish', cl3_opt='RMSProp'
                                 , cl4_act='Mish', cl4_opt='RMSProp'
                                 , ml_nn=64, ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)

class GAN_mddd_ccpccpccpm(GANBase):
    def __init__(self, n, **kwargs):
        '''
        Geneaotor    : 中間層＋逆畳込み層＋逆畳込み層＋逆畳込み層＋出力層
        dscriminator : 畳込み層＋畳込み層＋プーリング層
                     ＋畳込み層＋畳込み層＋プーリング層
                     ＋畳込み層＋畳込み層＋プーリング層＋中間層＋出力層
        C, Ih, Iw: 出力(画像)の大きさ
        入力(ノイズ)の大きさは最初のfowardで入力により決まる
        '''
        super().__init__(**kwargs)
        self.title = 'GAN_mddd_ccpccpccpm'
        self.gen = NN_CNN.CNN_mddd(n, **self.opt_for_gen)
        
        self.dsc = NN_CNN.CNN_ccpccpccpm(1 \
                                 , cl1_act='Mish', cl1_opt='RMSProp'
                                 , cl2_act='Mish', cl2_opt='RMSProp'
                                 , cl3_act='Mish', cl3_opt='RMSProp'
                                 , cl4_act='Mish', cl4_opt='RMSProp'
                                 , cl5_act='Mish', cl5_opt='RMSProp'
                                 , cl6_act='Mish', cl6_opt='RMSProp'
                                 , ml_nn=64, ml_act='Mish', ml_opt='RMSProp'
                                 , ol_act='Sigmoid', ol_opt='SGD'
                                 , method='average'
                                 , **self.opt_for_dsc)




