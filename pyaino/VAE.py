# VAE 各種
# 20251121 A.Inoue　
from pyaino.Config import *
import matplotlib.pyplot as plt
from pyaino import Neuron as neuron
from pyaino import common_function as cf
from pyaino import LossFunctions
from pyaino import Activators


class VAEBase():
    def __init__(self, rate=1.0, kld=None, mil=None, **kwargs):
        self.encoder = None
        self.decoder = None
        self.title = self.__class__.__name__
        
        # kwargsをencoderとdecoderに振分ける
        self.opt_for_enc, self.opt_for_dec = {}, {}
        keys_kwargs = list(kwargs.keys()) # iterator(ループ内変更不可)
        for k in keys_kwargs: 
            if k[:4]=='enc_':
                self.opt_for_enc[k[4:]]=kwargs.pop(k) 
            if k[:4]=='dec_':
                self.opt_for_dec[k[4:]]=kwargs.pop(k)
        self.opt_for_enc['ol_act'] = self.opt_for_enc.pop('ol_act', 'Identity')  # encoder出力層の活性化関数      
        # VAE全体に関わるoption
        loss_function_name  = kwargs.pop('loss', None)
        loss_function_options = {'axis': -1} # 二乗和誤差の場合も含めて末尾の軸に沿って算出し合算
        #loss_function_options['sumup']   = kwargs.pop('sumup',  False) # 誤差の勾配算出の際にバッチ内で足しこむ
        #loss_function_options['enhance'] = kwargs.pop('enhance',   10) # 誤差の勾配算出の際にかける倍率
        self.loss_function = cf.eval_in_module(loss_function_name, LossFunctions, **loss_function_options)

        # Normalization
        normalize = kwargs.pop('normalize', False)
        self.normalize = neuron.Normalization(axis=-1) if normalize else Activators.Identity()
        # サンプル層 潜在変数の大きさはencoderの出力
        self.sampling = neuron.LatentSampling(rate, kld, mil, **kwargs)

        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        self.opt_for_enc.update(kwargs)
        self.opt_for_dec.update(kwargs)

        print('options for encoder', self.opt_for_enc)
        print('options for decoder', self.opt_for_dec)
                          
    def summary(self):
        print('～'*39)
        print(self.__class__.__name__)
        self.encoder.summary()
        if self.normalize.__class__ == neuron.Normalization:
            print('-'*72)
            print(self.normalize.__class__.__name__)
        if self.sampling is not None:
            print('-'*72)
            print('sampling', self.sampling.__class__.__name__)
            if hasattr(self.sampling, 'rate'):
                print(' spread rate of sampling =',  self.sampling.rate, end='')
            if hasattr(self.sampling, 'kld') and self.sampling.kld is not None:
                print('\n kld =',   self.sampling.kld.__class__.__name__,
                      'rate =', self.sampling.r_kld, end='')
            if hasattr(self.sampling, 'mil') and self.sampling.mil is not None:
                print('\n mil =',   self.sampling.mil.__class__.__name__,
                      'rate =', self.sampling.r_mil, end='')
            print('\n'+'-'*72)
        self.decoder.summary()
        print('loss_function', self.loss_function.__class__.__name__, end=' ')
        if hasattr(self.loss_function, 'sumup'):
            print(' sumup =', self.loss_function.sumup, end=' ')
            print(' enhance =', self.loss_function.enhance, end=' ')
        print('\n'+'～'*39)
        print()

    def forward(self, x, enc_dropout=0.0, dec_dropout=0.0, **kwargs): #, kld=False, r_kld=1.0):
        y = self.encoder.forward(x, dropout=enc_dropout)
        y = self.normalize.forward(y)
        if not (self.sampling.kld or self.sampling.mil):
            z = self.sampling.forward(y); kll, mi = 0, 0
        else:    
            z, kll, mi = self.sampling.forward(y)
        self.z = z # 潜在変数を保存
        y = self.decoder.forward(z, dropout=dec_dropout)
        #if not self.sampling.kld:
        #    return y
        l = self.loss_function.forward(y, x.reshape(*y.shape))
        rec_error = float(l)        # rec_errorはdecoder出力のxとの隔たり        　
        reg_error = float(kll)      # reg_errorはsamplingで得られたKullback_Leibler divergence
        loss = float(l + kll + mi)  # lossは上記をr_kldに応じて合わせた値
        #self.r_kld = r_kld
        return y, loss, rec_error, reg_error

    def backward(self, gy=None, gl=1, gkll=1, gmi=1, **kwargs):
        if gy is not None:
            pass
        elif self.loss_function.t is not None:
            gy = self.loss_function.backward(gl, **kwargs)
        else:
            raise Exception("Can't get gradient by backward.")
        gz = self.decoder.backward(gy)
        gx = self.sampling.backward(gz, gkll, gmi)
        gx = self.normalize.backward(gx)
        gx = self.encoder.backward(gx)
        return gx

    def backward_bkup(self):
        gz = self.dec_backward() 
        self.enc_backward(gz, self.r_kld)  # dlossdkll=r_kld

    def dec_backward(self, gy=None, gl=1, **kwargs):
        if gy is not None:
            pass
        elif self.loss_function.t is not None:
            gy = self.loss_function.backward(gl, **kwargs)
        else:
            raise Exception("Can't get gradient by backward.")
        gx = self.decoder.backward(gy)
        return gx

    def enc_backward(self, gz, gkll=1, gmi=1):
        gx = self.sampling.backward(gz, gkll, gmi)
        gx = self.encoder.backward(gx)
        return gx

    def update(self, eta=0.0003, **kwargs):
        self.encoder.update(eta=eta, **kwargs)
        self.decoder.update(eta=eta, **kwargs)

    def enc_update(self, eta=0.0003, **kwargs):
        self.encoder.update(eta=eta, **kwargs)
       
    def dec_update(self, eta=0.0003, **kwargs):
        self.decoder.update(eta=eta, **kwargs)

    def get_image_size(self):
        input_config = self.encoder.layers[0].config
        if len(input_config) == 2:      # 全結合層を想定 
           pic_size = input_config[0]
           C = 1                        # カラーか白黒かはわからないが白黒にしておく
           Ih = Iw = int(pic_size**0.5) # 形状は正方形を想定　
        else:                           # 畳込み層を想定   
           C, Ih, Iw = input_config[:3]
        return C, Ih, Iw

    def get_rec_error(self, y, t):
        error = self.loss_function.forward(y, t)
        return float(error)

    def get_reg_error(self):
        ''' Kullback-Leibler divergence '''
        mu = self.sampling.mu
        log_var = self.sampling.log_var
        error = self.sampling.kld.forward(mu, log_var)
        return float(error)

    def check_latent_variables(self, z):
        nz = z.shape[1]
        x = np.zeros(nz)
        for i in range(nz):
            x[i] = np.mean(z[:, i])
            var  = np.var(z[:, i].astype('f4'))
            print('{:3d}番目の潜在変数　平均値 {:6.3f} 分散 {:6.3f}'\
                  .format(i, float(x[i]), float(var)))


    def export_params(self):
        params = {}
        enc_params = self.encoder.export_params()
        dec_params = self.decoder.export_params()
        for k in enc_params.keys():
            params['enc_'+k] = enc_params[k]
        for k in dec_params.keys():
            params['dec_'+k] = dec_params[k] 
        return params
    
    def import_params(self, params):
        enc_params, dec_params = {}, {}
        print('source', params.keys())
        for k in params.keys():
            if k[:4]=='enc_':
                enc_params[k[4:]]=params[k]
            if k[:4]=='dec_':
                dec_params[k[4:]]=params[k]
        self.encoder.import_params(enc_params)
        self.decoder.import_params(dec_params)
        print('encoder', enc_params.keys())
        print('decoder', dec_params.keys())

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

    # -- 画像の表示 --　
    def show_images(self, x, n=10, offset=0, epsilon=0):
        C, Ih, Iw = self.get_image_size()
        print(C, Ih, Iw)
        # 表示のパラメタ確定
        if C == 1:
            shape = Ih, Iw    # 画像の元形状
            trnsp = 0, 1      # 軸交換
            cmap  = 'Greys_r' # 色指示
        if C > 2:
            shape = C, Ih, Iw # 画像の元形状　
            trnsp = 1, 2, 0   # 軸交換 
            cmap  = None      # 色指示 
        # 潜在変数と出力画像を生成   
        y = self.encoder.forward(x)
        y = self.normalize.forward(y)
        z_etc = self.sampling.forward(y, epsilon=epsilon) # zも表示する
        z = z_etc[0] if type(z_etc) is tuple else z_etc
        y = self.decoder.forward(z)
        # 表示用に値の範囲を補正
        x = (x - np.min(x))/(np.max(x) - np.min(x))
        y = (y - np.min(y))/(np.max(y) - np.min(y))
        plt.figure(figsize=(15, 7))#, dpi=50)
        # 潜在変数の表示形状
        zs = z.shape[-1]
        zh = int(zs**0.5)
        zw = zs // zh
        z = z[:,:zh*zw]
        # 表示 
        for i in range(n):
            j = i + offset
            # 入力
            ax = plt.subplot(3, n, i+1)
            plt.imshow(x[j].reshape(shape).transpose(trnsp).tolist(), cmap=cmap)
            ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
            # 中間層 　　　
            ax = plt.subplot(3, n, i+1+n)
            plt.imshow(z[j].reshape(zh, zw).tolist(), cmap='Greys_r')
            ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False) 
            # 出力    　　
            ax = plt.subplot(3, n, i+1+2*n)
            plt.imshow(y[j].reshape(shape).transpose(trnsp).tolist(), cmap=cmap)
            ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
        plt.show()

    # -- 潜在変数を与え、その画像を表示する --
    def picture_image(self, x, pil=False, save=False):
        C, Ih, Iw = self.get_image_size()
        cf.picture_image(self.decoder.forward, np.array(x), C, Ih, Iw, pil=pil, save=save)

    # -- 画像を生成して表示 --
    def generate_random_images(self, *args, **kwargs):
        C, Ih, Iw = self.get_image_size()
        nz  = self.decoder.layers[0].config[0]
        cf.generate_random_images(self.decoder.forward, nz, C, Ih, Iw, *args, **kwargs)

    def picture_varying_latent_variables_bkup(self, *args, **kwargs):
        C, Ih, Iw = self.get_image_size()
        cf.picture_varying_latent_variables(self.decoder.forward, self.z, C, Ih, Iw, *args, **kwargs)

    def picture_varying_latent_variables(self, *args, **kwargs):
        C, Ih, Iw = self.get_image_size()
        nz  = self.decoder.layers[0].config[0]
        z1, z2 = cf.picture_varying_latent_variables(self.decoder.forward, nz, C, Ih, Iw, *args, **kwargs)
        return z1, z2

    # -- 潜在変数を平面にプロット --
    def plot_latent_variables(self, x, t, axis=(0,1)):
        def func(x):
            y = self.encoder.forward(x)
            y = self.normalize.forward(y)
            z_etc = self.sampling.forward(y)
            z = z_etc[0] if type(z_etc) is tuple else z_etc
            return z
        cf.plot_latent_variables(func, x, t, axis)


# -- 外部から直接アクセス --
def build(Class, *args, **kwargs):
    print(Class, args, kwargs)
    global model
    model = Class(*args, **kwargs)

def summary():
    model.summary()

def forward(x, t=None, **kwargs):
    return model.forward(x, t, **kwargs)

def enc_forward(x, **kwargs):
    return model.enc_forward(x, **kwargs)

def dec_forward(x, t=None, **kwargs):
    return model.dec_forward(x, t, **kwargs)

def backward(gy=None, gl=1):
    return model.backward(gy, gl)

def dec_backward(gy=None, gl=1, **kwargs):
    return model.dec_backward(gy, gl, **kwargs)

def enc_backward(grad_z):
    return model.enc_backward(grad_z)

def update(**kwargs):
    model.update(**kwargs)

def enc_update(**kwargs):
    model.enc_update(**kwargs)

def dec_update(**kwargs):
    model.dec_update(**kwargs)

# -- 学習結果の保存 --
def save_parameters(file_name):
    model.save_parameters(file_name)

# -- 学習結果の継承 --
def load_parameters(file_name):
    return model.load_parameters(file_name) 

# -- 誤差を計算 --
def get_error(y, t):
    eps = 1e-7
    return -np.sum(t*np.log(y+eps) + (1-t)*np.log(1-y+eps)) / len(y)  # 二値の交差エントロピー誤差

# -- 正解率を計算 --
def get_accuracy(y, t):
    correct = np.sum(np.where(y<0.5, 0, 1) == t)
    return correct / len(y)

# -- 誤差を計算 --
def get_rec_error(y, t):
    return model.get_rec_error(y, t) 

def get_reg_error():
    return model.get_reg_error() 

def get_reg_error2():
    return model.get_reg_error2() 

# -- 潜在変数を与え、その画像を表示する --
def picture_image(x, pil=False, save=False):
    model.picture_image(x, pil, save)

# -- 画像を生成して表示 --
def generate_random_images(*args, **kwargs):
    model.generate_random_images(*args, **kwargs)
   
def picture_varying_latent_variables(*args, **kwargs):
    z1, z2 = model.picture_varying_latent_variables(*args, **kwargs)
    return z1, z2
    
def check_latent_variables():
    model.check_latent_variables()

# -- 潜在変数を平面にプロット --
def plot_latent_variables(x, t, axis=(0,1)):
    cf.plot_latent_variables(enc_forward, x, t, axis)

# -- 画像の表示 --　
def show_images(x, n=10, offset=0):
    model.show_images(x, n, offset)
    

def show_images_bkup(x):
    import random
    y = forward(x)
    n = 10
    idx = list(range(len(x)))
    Ih = Iw = int(x.shape[1]**0.5)
    #zh = int(nz**0.5); zw = nz // zh
    plt.figure(figsize=(15, 7))#, dpi=50)
    for i in range(n):
        j = random.choice(idx) # i + 65 # index_of_data
        # 入力
        ax = plt.subplot(4, n, i+1)
        plt.imshow(x[j].reshape(Ih, Iw).tolist(), cmap='Greys_r')
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
        # 中間層 　　　
        ax = plt.subplot(4, n, i+1+n)
        #plt.imshow(sampling.z[j].reshape(zh, zw).tolist(), cmap='Greys_r')
        plt.imshow(model.sampling.z[j].reshape(1, -1).tolist(), cmap='Greys_r')
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
        # 出力 1   　　
        ax = plt.subplot(4, n, i+1+2*n)
        plt.imshow(dec_y[j].reshape(Ih, Iw).tolist(), cmap='Greys_r')
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
    plt.show()

#### VAE 各モデル #### 
from pyaino import NN_CNN

class VAE_0_z_m(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.NN_0(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.NN_m(C*Ih*Iw, **self.opt_for_dec)

class VAE_c_z_md(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_c(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_md(C*Ih*Iw, **self.opt_for_dec)

class VAE_cp_z_mud(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_cp(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mud(C*Ih*Iw, **self.opt_for_dec)

class VAE_cpm_z_mdc(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_cpm(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mdc(C*Ih*Iw, **self.opt_for_dec)

class VAE_ccp_z_mudd(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_ccp(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mudd(C*Ih*Iw, **self.opt_for_dec)

class VAE_ccp_z_mddd(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_ccp(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mddd(C*Ih*Iw, **self.opt_for_dec)

class VAE_ccp_z_mdddc(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_ccp(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mdddc(C*Ih*Iw, **self.opt_for_dec)

class VAE_ccp_z_mddddc(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_ccp(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mddddc(C*Ih*Iw, **self.opt_for_dec)

class VAE_cccp_z_mddddc(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_cccp(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mddddc(C*Ih*Iw, **self.opt_for_dec)

class VAE_cccc_z_mddddc(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_cccc(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mddddc(C*Ih*Iw, **self.opt_for_dec)

class VAE_ccccp_z_mddddc(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_ccccp(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mddddc(C*Ih*Iw, **self.opt_for_dec)

class VAE_ccpccp_z_mddddc(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_ccpccp(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mddddc(C*Ih*Iw, **self.opt_for_dec)

class VAE_ccpccgp_z_mddddc(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_ccpccgp(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mddddc(C*Ih*Iw, **self.opt_for_dec)

class VAE_ccpccpccgp_z_mddddc(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_ccpccpccgp(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mddddc(C*Ih*Iw, **self.opt_for_dec)

class VAE_ccpccp_z_mdcdcdc(VAEBase):
    def __init__(self, C, Ih, Iw, z, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NN_CNN.CNN_ccpccp(2*z, **self.opt_for_enc)
        self.decoder = NN_CNN.CNN_mdcdcdc(C*Ih*Iw, **self.opt_for_dec)

if __name__=='__main__':
    print('\n#### all cast ####')
    import inspect
    import sys
    current_module = sys.modules[__name__]
    classes = map(lambda x:x[0],inspect.getmembers(current_module,inspect.isclass))
    classes = list(classes)
    print(classes)
        
