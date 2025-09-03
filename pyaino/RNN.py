# RNN
# 2025.09.03 A.Inoue
from pyaino.Config import *
from pyaino import Neuron as neuron, Functions as f
from pyaino import LossFunctions
from pyaino import common_function as cf
import copy, random

# ニューラルネットワークの構築
class RNN_Base: 
    def __init__(self, *args, **kwargs):
        self.title = self.__class__.__name__
        # 引数の判定(configで渡すだけだが一応見ておく,後でlやnをデータから判定)
        if len(args) == 4:
            v, l, m, n = args
        elif len(args) == 3:
            l, m, n = args; v = None
        else:
            v, l, m, n = None, None, None, None
        self.config = v, l, m, n # 語彙数,語ベクトル長,隠れ層ニューロン数,出力数
        print('Initialize', self.__class__.__name__, self.config)
        
        # 残差接続residual(未指定:False、数値:有効、その他:例外)
        res = kwargs.pop('residual', None)
        if res is None:
            self.residual = False
        elif isinstance(res, (int, float, np.floating)):
            self.residual = np.asarray(float(res), dtype=Config.dtype)
        else:
            raise TypeError("residual should be scalar number in range 0-1.")
            
        # 出力層(full_connection_layer、affine_layer)
        ol_act_cand = 'Sigmoid' if n == 1 else 'Softmax'
        opt_for_ol = {}
        opt_for_ol['activate'] = kwargs.pop('ol_act', ol_act_cand)
        opt_for_ol.update(kwargs)    # kwargsに残ったものを結合
        self.opt_for_ol = opt_for_ol # subclassで出力層初期化時に使用

        self.layers = []
        self.forward_options = None  # RNN全体としてのCPTやDOの設定
        self.fopt_for_layers = None  # 層毎のCPTやDOの設定
        self.r0_target_layer = None  # 層毎のr0設定の有無  
        self.r0 = None               # 外部から与えられるr0(seq2seqで使用)
        self.r0_awaiting = False     # r0が設定待ち 
        self.r0_given = False        # r0がforwardの際に与えられた
        self.gr0 = None              # r0の勾配

        # 損失関数
        loss_function_name = kwargs.pop('loss', None)
        if loss_function_name is not None:
            ignore_label = kwargs.pop('ignore', -1)
            print(loss_function_name, ignore_label)
            self.loss_function = cf.eval_in_module(loss_function_name, LossFunctions)

        # 重み共有
        share_weight = kwargs.pop('share_weight', False) # Embeddingと全結合層の重み共有 
        if share_weight == True and l==m and v==n:       # 重み共有の場合には、D==H の必要がある
            print('embeddingとoutputで重みが共有されます')
            self.share_weight = True
        else:                                            
            self.share_weight = False
            
        self.error_layer = None # デバグ用    
        self.outputshape = {}   # デバグ用

    def configure_layers(self, **kwargs):
        """ クラス名に応じてRNNの層を構成する """
        base_name = self.__class__.__base__.__name__
        name = self.__class__.__name__
        print('Configure', name, 'based on', base_name)
        if name[:4] != 'RNN_':
            raise Exception()
        v, l, m, n = self.config
        layer_config = None
                
        for i, char in enumerate(name[4:]):
            if i==0:                 # 最初の層
                if char=='e':
                    layer_config = v, l
                else:
                    layer_config = l, m
                if self.share_weight and char!='e':
                    print('Share_weight is not accepted.')
                    self.share_weight = False
            elif i==len(name[4:])-1: # 最後の層
                if char=='f':
                    layer_config = layer_config[1], n # 左シフトして右にnを加える
                    kwargs = self.opt_for_ol
                else:
                    layer_config = layer_config[1], m # 左シフトして右にmを加える
            else:                    # 中間の層
                layer_config = layer_config[1], m     # 左シフトして右にmを加える
                
            print('type', char, 'layer_config =', layer_config)
            if   char == 'e':
                layer = neuron.Embedding(*layer_config, **kwargs)
            elif char == 'r':
                layer = neuron.RNN(*layer_config, **kwargs)
            elif char == 'l':
                layer = neuron.LSTM(*layer_config, **kwargs)
            elif char == 'g':
                layer = neuron.GRU(*layer_config, **kwargs)
            elif char == 'f':
                layer = neuron.NeuronLayer(*layer_config, **kwargs) # 仮opt_for_ol未対応
            elif char == 'n':
                layer = neuron.L2Normalize()
            elif char == 's':
                layer = neuron.ContextualSelfAttention(*layer_config, **kwargs)
            elif char == 'a':
                if base_name!='RNN_With_Attention_Base':
                    raise Exception('Cannot configure', name, 'on', base_name)
                layer = neuron.AttentionUnit(**kwargs) # Attentionはconfig等の指定不要　
                # 下流の層に渡すのは、1つ上流の層の出力＋Attentionの出力
                #if layer_config[0] is not None and layer_config[1] is not None:
                if not None in layer_config:
                    layer_config = layer_config[0], layer_config[0]+layer_config[1]
            else:    
                raise Exception(name, base_name, "Layer cannot be configured.", i, char)
            self.layers.append(layer)

            self.set_r0_target_layer()

            

    def summary(self):
        print('～～ model summary of ' + str(self.__class__.__name__) + str(self.config)
              + ' ～～～～～～～～～～～～')
        for i, layer in enumerate(self.layers):
            print('layer', i, layer.__class__.__name__, end=' ')
            if layer.__class__.__base__.__name__=='RnnBaseLayer':
                print(' configuration =', layer.config[:2], 'stateful =', layer.config[2])
            else:
                print(' configuration =', layer.config)
            if hasattr(layer, 'method'):
                print(' method =', layer.method, end=' ')
            if hasattr(layer, 'activator'):
                print(' activate =', layer.activator.__class__.__name__, end=' ')
            if hasattr(layer, 'optimizer_w'):
                print(' optimize =', layer.optimizer_w.__class__.__name__, end='')
            if hasattr(layer, 'DO') and i<len(self.layers)-1:
                print('  dropout applicable.', end='')
            if getattr(layer, 'cell_normalization', False):
                print('\n cell_normalization = True', end='')
            if getattr(layer, 'batchnorm', False):
                print('\n batch_normalization = True', end='')
            if hasattr(layer, 'update') and hasattr(layer, 'w_decay') and layer.w_decay!=0:     
                print('\n weight_decay_lambda =', layer.w_decay, end=' ')
            if getattr(layer, 'WCw', None) is not None:
                print('   weight_clipping = ', layer.WCw.clip, end='')
            if getattr(layer, 'WC2w', None) is not None:
                print('   weight_clipping2 = ', layer.WC2w.clip, end='')
            if self.residual and i>0 and layer.__class__.__base__.__name__=='RnnBaseLayer':
                print(' residual connection =', self.residual, end='')
            print('\n------------------------------------------------------------------------')
        #print(' except for output layer, dropout rate is given when forward propagation')
        if hasattr(self, 'loss_function'):
            print(' loss_function =', self.loss_function.__class__.__name__)
        if self.share_weight:
            print(' sharing parameter(weight) of embedding layer and output layer')
        print('～～ end of summary ～～～～～～～～～～～～～～～～～～～～～～～～～～～～\n')

    def reset_state(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_state'):
                layer.reset_state()

    def set_state(self, r0): # 最後のRNN層のr0をセット Seq2seqのdecoderで使う
        self.reset_state() # 仮処置20240726
        self.r0 = r0
        self.r0_awaiting = True

        # 以下はforward中でも良いはず
        #for layer, r0t in zip(self.layers, self.r0_target_layer):
        #    if r0t and self.r0_awaiting:
        #        layer.set_state(self.r0)
        #        self.r0_awaiting = False
      
        #for layer in reversed(self.layers):
        #    if hasattr(layer, 'set_state'):
        #        layer.set_state(r0)
        #        break

    def get_grad_r0(self):
        #for layer, r0t in zip(reversed(self.layers), reversed(self.r0_target_layer)):
        #    if r0t:# and hasattr(layer, 'grad_r0'):
        #        return layer.grad_r0

        #for layer in reversed(self.layers):
        #    if hasattr(layer, 'grad_r0'):
        #        return layer.grad_r0
        
        #print(self.__class__.__name__, 'get_grad_r0', self.gr0 is not None)
        return self.gr0
            

    def update(self, **kwargs):
        # 始めと終わりの層は重み共有を判定して更新
        if self.share_weight and self.layers[0].__class__.__name__ == 'Embedding':
            # embedding_layerをoutput_layerの勾配を加味して更新
            self.layers[0].grad_w += self.layers[-1].grad_w.T
            self.layers[0].update(**kwargs)
            # output_layerはembedding_layerのwに合わせる
            self.layers[-1].w = self.layers[0].w.T
        else:
            self.layers[0].update(**kwargs)
            self.layers[-1].update(**kwargs)
        # 重み共有に関係ない層を一括更新    
        for layer in self.layers[1:-1]:
            if hasattr(layer, 'update'):
                layer.update(**kwargs)

    def set_r0_target_layer(self):
        """ r0設定対象の層とそうでない層を順にリストで示す """
        self.r0_target_layer = []; target_set = False
        for layer in reversed(self.layers): # 下流側から 
            r0t = False  # 各層のr0設定の有無
            if layer.__class__.__base__.__name__=='RnnBaseLayer' and not target_set:
                r0t = True
                target_set = True
            self.r0_target_layer.append(r0t)  # 該層のr0設定フラグをリストに加える
        self.r0_target_layer.reverse()

    def set_fopt_for_layers(self, CPT, dropout):
        """ forwardの引数に従い各層に順伝播時の設定を行う """
        self.fopt_for_layers = []
        cpt_set = False
        for layer in reversed(self.layers): 
            opt = {}     # 各層のオプション
            if hasattr(layer, 'CPT') and not cpt_set: # CPTはRNN層の出口で行う　
                opt['CPT'] = CPT
                cpt_set = True
            if hasattr(layer, 'DO') and layer.__class__.__name__ != 'NeuronLayer':
                opt['dropout'] = dropout
            self.fopt_for_layers.append(opt)
        self.fopt_for_layers.reverse() # 逆順を正順に戻す
        self.forward_options = (CPT, dropout)
        #print(self.__class__.__name__, self.forward_options, self.fopt_for_layers)

    def loss(self, y, t):
        if hasattr(self, 'loss_function'):
            l = self.loss_function.forward(y, t)
            return l
        else:
            raise Exception('No loss_function defined.')
        
    def select_category(self, x, stochastic=False, beta=2):
        return cf.select_category(x, stochastic, beta)

    def forward(self, x, t=None, *, mask=None, CPT=None, dropout=0.0):
        if self.forward_options != (CPT, dropout): # dropoutかCPTの指定が変わったとき 
            self.set_fopt_for_layers(CPT, dropout)
        y = x
        for i, (layer, opt, r0t) in \
                enumerate(zip(self.layers, self.fopt_for_layers, self.r0_target_layer)):
            self.error_layer = layer                                      # デバグ用 

            last_x = y.copy()
            
            if hasattr(layer, 'mask'): # Emmbeddingや親がRnnBaseLayerの場合のmask
                opt['mask'] = mask
                
            if r0t and self.r0_awaiting: # RNNのr0を外から与える場合
                y = layer.forward(y, self.r0, **opt) # r0は位置引数、optはキーワード引数
                self.r0_awaiting = False
                self.r0_given = True
            else:
                y = layer.forward(y, **opt) # optは辞書型

            #print(i, layer.__class__.__name__, last_x.shape, y.shape)  
            if self.residual and i>0 and layer.__class__.__base__.__name__=='RnnBaseLayer':
                y = y + self.residual * last_x 
            
            self.outputshape[str(i) + layer.__class__.__name__] = y.shape # デバグ用

        if t is None:
            return y
        elif hasattr(self, 'loss_function'):
            l = self.loss_function.forward(y, t)
            return y, l
        else:
            raise Exception("Can't get loss by forward.")

    # -- 逆伝播 --
    def backward(self, gy=None, gl=1):
        if gy is not None:
            pass
        elif self.loss_function.t is not None:
            gy = self.loss_function.backward(gl)
        else:
            print(self.__class__.__name__, 'gy =', gy, 'gl =', gl)
            raise Exception("Can't get gradient for backward.")

        gx = gy; gr0 = None
        for layer, r0t in \
            enumerate(zip(reversed(self.layers), reversed(self.r0_target_layer))):
            self.error_layer = layer

            last_gx = gx.copy()
            
            if r0t and self.r0_given:
                gx, gr0 = layer.backward(gx)
                self.r0_given = False
            else:    
                gx = layer.backward(gx)

            
            #print(layer.__class__.__name__, type(last_gx), type(gx))
            #print(layer.__class__.__name__, last_gx.shape)#, gx.shape)
            if self.residual and i > len(self.layers) -1 \
                             and layer.__class__.__base__.__name__=='RnnBaseLayer':
                gx += self.residual * last_gx
                
        self.gr0 = gr0        
        return gx

    def generate2(self, seed, length=200,
                 stochastic=True, beta=2, skip_ids=None, end_id=None, reset=True):
        """ seedから順に次時刻のデータをlengthになるまで生成 """
        gen_data = seed if isinstance(seed, np.ndarray) else np.array(seed)
        if gen_data.ndim==0:
            gen_data = gen_data.reshape(-1)
        
        if gen_data.ndim==2:
            T, l = gen_data.shape
        else:
            T = len(gen_data)
            l = 1

        if any(l.__class__.__name__=='Embedding' for l in self.layers):
            x_shape = 1, 1
            y_shape = -1
            categorical = True # 取り敢えずembedding層有りで決める
        else:
            x_shape = 1, 1, l
            y_shape = 1, -1
            categorical = False

        if reset:
            self.reset_state()
        for j in range(length-1):
            x = gen_data[j].reshape(x_shape)
            y = self.forward(x)
            y = y.reshape(y_shape)

            if j+1 < T: # seedの範囲は書込まない
                continue

            if not categorical: # 非カテゴリカルならば、そのまま綴って次へ　
                gen_data = np.concatenate((gen_data, y))
                continue
            
            # -- 以下はカテゴリカルな処理 --
            next_one = cf.select_category(y, stochastic, beta) 
            if (skip_ids is not None) and (next_one in skip_ids):
                continue # skip_idは含めない
            gen_data = np.concatenate((gen_data, next_one))
            if end_id is not None and next_one==end_id: 
                break    # end_idまで含んで終わる

        return gen_data

    def generate(self, *args, **kwargs):
        """ seed から順に次時刻のデータを leng になるまで生成 """
        if any(l.__class__.__name__ == 'Embedding' for l in self.layers):
            return self.generate_id_from_id(*args, **kwargs)
        else:
            return self.generate_data_from_data(*args, **kwargs)

    def generate_data_from_data(self, seed, length=200, reset=True):
        """
        seed から順に次時刻のデータを size になるまで生成(画像などの値) 
        embedding を伴わない RNN で画像など、値から値を生成する場合に適合

        """
        T, l = seed.shape
        gen_data = np.zeros((length, l))
        gen_data[:T] = seed # seedの範囲を生成データにセット
        if reset:
            self.reset_state()
        for j in range(length - 1):
            x = gen_data[j]
            y = self.forward(x.reshape(1, 1, l))
            if j+1 < T:     # seedの範囲は書込まない
                continue
            gen_data[j+1] = y.reshape(-1)
        return gen_data

    def generate_id_from_id(self, seed, length=200,
               stochastic=True, beta=2, skip_ids=None, end_id=None, reset=True):
        """
        seed から順に次時刻のデータを length になるまで生成(文字列などのid)
        embedding を備えた RNN で文字列生成をする場合に適合

        """
        gen_data = seed if isinstance(seed, np.ndarray) else np.array(seed)
        if gen_data.ndim == 0:
            gen_data = gen_data.reshape(-1)
        #print('### debug', gen_data, type(gen_data))
        T = len(gen_data) 
        if reset:
            self.reset_state()

        for j in range(length - 1):
            x = gen_data[j]
            y = self.forward(x.reshape(1, 1)) # embeddingへの入力形状は(B, T) 
            if j+1 < T:   # seedの範囲はgen_dataに出力を加えない
                continue
            next_one = cf.select_category(y, stochastic, beta)
            if skip_ids is not None and next_one in skip_ids:
                continue
            gen_data = np.concatenate((gen_data, next_one))
            if end_id is not None and next_one==end_id: # end_idが出現したら打切り　
                break
        return gen_data

    # -- seed から順に次時刻のデータを size になるまで生成(画像などの値) --
    # embedding を伴わない RNN で画像など、値から値を生成する場合に適合
    def generate_data_from_data2(self, seed, length=200,
                                verbose=False, extension=False):
        """ 末尾に一つ追加しながら頭は変えずに伸ばしていく """
        T, l = seed.shape
        gen_data = np.zeros((length, l))
        gen_data[0:T, :] = seed
        x_record, y_record = [], []
        for j in range(length - T):                   
            self.reset_state()
            x = gen_data[0:j+T, :] if extension else gen_data[j:j+T, :]
            y = self.forward(x.reshape(1, -1, l))
            gen_data[j+T, :] = y[0, -1, :]
            x_record.append(x)
            y_record.append(y)
        if verbose:
            return gen_data, x_record, y_record
        return gen_data

    # -- seed から順に次時刻のデータを length になるまで生成(文字列などのid) --
    # embedding を備えた RNN で文字列生成をする場合に適合
    def generate_id_from_id2(self, seed, length=200,
               beta=2, skip_ids=None, end_id=None, stochastic=True, extension=False):
        """ 末尾に一つ追加しながら頭は変えずに伸ばしていく """
        T = len(seed)
        gen_data = np.array(seed)
        for i in range(length - T):
            self.reset_state()
            x = gen_data if extension else gen_data[i:]
            y = self.forward(x.reshape(1, -1))
            y = y[0,-1,:]                 # 非バッチ処理の末尾の時刻
            if stochastic: # 確率的に
                p = np.empty_like(y, dtype='f4') # 極小値多数でエラーしないため精度が必要
                p[...] = y ** beta
                p = p / np.sum(p)
                next_one = np.random.choice(len(p), size=1, p=p)
            else:         # 確定的に
                next_one = np.argmax(y)
            if end_id is not None and next_one==end_id:  # end_idが出現したら打切り　
                break
            if (skip_ids is None) or (next_one not in skip_ids):
                gen_data = np.concatenate((gen_data, next_one))

        return gen_data

    def generate_text(self, x_to_id, id_to_x, length=100,
                      seed=None, stop=None, print_text=True, end='', 
                      stochastic=True, beta=2):
        """ 文字列の生成 x:文字または語 x_chain:その連なり key:次の索引キー """
        self.reset_state()
        x_chain = ""
        vocab_size = len(x_to_id)

        if seed is None:      
            seed = random.choices(list(x_to_id)) 
            
        for j in range(length):
            # -- 書出しはseedから、その後は前回出力から、文字または語とその識別子を得る
            if j < len(seed): # 書出しはseedから 
                x = seed[j]        
                key = x_to_id[x]   
            else:             # その後は前回出力から
                key = self.select_category(y, stochastic=stochastic, beta=beta) 
                x = id_to_x[key]    
            # -- 文字または語を綴る --
            x_chain += x
            if print_text:
                print(x, end=end)
            if x==stop or len(x_chain)>length:
                break
            # -- 次に備える --
            if hasattr(self, 'embedding_layer'):
                key = np.array(key).reshape(1, 1)
            else:
                key = np.eye(vocab_size)[key]
                key = key.reshape(1, 1, -1)
            y = self.forward(key)

        return x_chain    


    # -- パラメタから辞書 --
    def export_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w'):
                params['layer'+str(i)+'_w'] = np.array(layer.w)
            if hasattr(layer, 'v'):
                params['layer'+str(i)+'_v'] = np.array(layer.v)
            if hasattr(layer, 'b'):
                params['layer'+str(i)+'_b'] = np.array(layer.b)
        return params

    # -- 辞書からパラメタ --
    def import_params(self, params):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w'):
                layer.w = np.array(params['layer'+str(i)+'_w']) 
            if hasattr(layer, 'v'):
                layer.v = np.array(params['layer'+str(i)+'_v']) 
            if hasattr(layer, 'b'):
                layer.b = np.array(params['layer'+str(i)+'_b'])
            
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

class RNN_With_Attention_Base(RNN_Base):

    def forward(self, x, z, t=None, *, mask=None, CPT=None, dropout=0.0): # zはAttentionへの入力
        if self.forward_options != (CPT, dropout):  # dropoutかCPTの指定が変わったとき 
            self.set_fopt_for_layers(CPT, dropout)
        y = x
        for i, (layer, opt, r0t) in \
                enumerate(zip(self.layers, self.fopt_for_layers, self.r0_target_layer)):
            self.error_layer = layer

            last_x = y.copy() 

            if hasattr(layer, 'mask'): # Emmbeddingや親がRnnBaseLayerの場合のmask
                opt['mask'] = mask
                
            if layer.__class__.__name__ == 'AttentionUnit':
                #c = layer.forward(z, z, y, **opt) # z:key&value, y:query, optは辞書型
                c = layer.forward(z, z, y, **opt) # 仮20250202AI # z:key&value, y:query, optは辞書型

                #y = np.concatenate((c, y), axis=-1) # <<< これが問題か？ 
                y = f.concatenate(c, y, axis=-1) # <<< これが問題か？
                
            elif r0t and self.r0_awaiting:  # RNNのr0を外から与える場合
                y = layer.forward(y, self.r0, **opt) # r0は位置引数、optはキーワード引数
                self.r0_awaiting = False
                self.r0_given = True
            else:
                y = layer.forward(y, **opt) # optは辞書型

            #print(i, layer.__class__.__name__, last_x.shape, y.shape)  
            if self.residual and i>0 and layer.__class__.__base__.__name__=='RnnBaseLayer':
                y = y + self.residual * last_x 
            
            self.outputshape[str(i) + layer.__class__.__name__] = y.shape # デバグ用
               
        if t is None:
            return y
        elif hasattr(self, 'loss_function'):
            l = self.loss_function.forward(y, t)
            return y, l
        else:
            raise Exception("Can't get loss by forward.")

    def backward(self, gy=None, gl=1):
        if gy is not None:
            pass
        elif self.loss_function.t is not None:
            gy = self.loss_function.backward(gl)
        else:
            print(self.__class__.__name__, 'gy =', gy, 'gl =', gl)
            raise Exception("Can't get gradient for backward.")

        gx = gy; gr0 = None
        for i, (layer, r0t) in \
            enumerate(zip(reversed(self.layers), reversed(self.r0_target_layer))):
            self.error_layer = layer
            
            last_gx = gx.copy()
            
            if layer.__class__.__name__ == 'AttentionUnit':
                B, T, H2 = gx.shape; H = H2//2
                gv, gk, gq = layer.backward(gx[:,:,:H]) # gv:valueの勾配、gk:keyの勾配、gq:queryの勾配
                gz = gv + gk          # gz:key&valueの勾配
                gx = gx[:,:,H:] + gq  # forwardでattentionとaffineに配っていることに対応
            elif r0t and self.r0_given:
                gx, gr0 = layer.backward(gx)
                self.r0_given = False
            else:    
                gx = layer.backward(gx)

            #print(i, layer.__class__.__name__, type(last_gx), type(gx))
            #print(i, layer.__class__.__name__, last_gx.shape)#, gx.shape)
            if self.residual and i > len(self.layers) -1 \
                             and layer.__class__.__base__.__name__=='RnnBaseLayer':
                gx += self.residual * last_gx
                
        self.gr0 = gr0
        return gx, gz

    def generate_data_from_data(self, seed, z, length=200, reset=True):
        """
        seed から順に次時刻のデータを size になるまで生成(画像などの値) 
        embedding を伴わない RNN で画像など、値から値を生成する場合に適合

        """
        T, l = seed.shape
        gen_data = np.zeros((length, l))
        gen_data[:T] = seed # seedの範囲を生成データにセット
        if reset:
            self.reset_state()
        for j in range(length - 1):
            x = gen_data[j]
            y = self.forward(x.reshape(1, 1, l), z) # zはAttentionへの入力
            if j+1 < T:     # seedの範囲は書込まない
                continue
            gen_data[j+1] = y.reshape(-1)
        return gen_data

    def generate_id_from_id(self, seed, z, length=200,
               stochastic=True, beta=2, skip_ids=None, end_id=None, reset=True):
        """
        seed から順に次時刻のデータを length になるまで生成(文字列などのid)
        embedding を備えた RNN で文字列生成をする場合に適合

        """
        gen_data = seed if isinstance(seed, np.ndarray) else np.array(seed)
        if gen_data.ndim == 0:
            gen_data = gen_data.reshape(-1)
        #print('### debug', gen_data, type(gen_data))
        T = len(gen_data) 
        if reset:
            self.reset_state()

        for j in range(length - 1):
            x = gen_data[j]
            y = self.forward(x.reshape(1, 1), z) # zはAttentionへの入力 
            if j+1 < T:   # seedの範囲はgen_dataに出力を加えない
                continue
            next_one = cf.select_category(y, stochastic, beta) 
            if skip_ids is not None and next_one in skip_ids:
                continue
            gen_data = np.concatenate((gen_data, next_one))
            if end_id is not None and next_one==end_id: # end_idが出現したら打切り　
                break
        return gen_data


# -- 外部から直接アクセス --
def select_category(x, stochastic=False, beta=2):
    return cf.select_category(x, stochastic, beta)

def generate(func, seed, length=200):
    """ seedに続いて一つずつ生成していく """
    gen_data = np.array(seed)
    T, l = gen_data.shape
    for j in range(length-1):
        x = gen_data[j]
        y = func(x)
        if j+1 < T: # seedの範囲は書込まない
            continue
        next_one = y.reshape(1, l)
        gen_data = np.concatenate((gen_data, next_one))
    return gen_data

def generate_text(func, x_to_id, id_to_x, length=100,
                  seed=None, stop=None, print_text=True, end='', 
                  stochastic=True, beta=2):
    """ 文字列の生成 x:文字または語 x_chain:その連なり key:次の索引キー """
    x_chain = ""
    vocab_size = len(x_to_id)
    y = None
    # -- seed無指定なら辞書からランダムに選ぶ --
    if seed is None:      
        seed = random.choices(list(x_to_id)) 
        
    for j in range(length):
        # -- 書出しはseedから、その後はfunc出力から --
        if j < len(seed): # seedの範囲 
            x = seed[j]
            try:
                key = x_to_id[x]
            except:
                key = None
        elif y is None:   # seedを超えてもyがNoneのまま
            print(' => No valid seed specified.')
            break
        else:             # seedの範囲を超えた
            key = cf.select_category(y, stochastic, beta=beta)
            x = id_to_x[int(key)]
        # -- 綴る --
        x_chain += x
        if print_text:
            print(x, end=end)
        if x==stop or len(x_chain)>length:
            break
        # -- 次に備える --
        if key is not None:
            y = func(key)     # seedの範囲を含めて順伝播

    return x_chain    


def build(Class, *args, **kwargs):
    print(Class, args, kwargs)
    global model
    model = Class(*args, **kwargs)
    
def summary():
    model.summary()

def reset_state():
    model.reset_state()
    
def forward(*args, **kwargs):#d x, t=None, CPT=None, dropout=0.0):
    return model.forward(*args, **kwargs)#x, t, CPT, dropout)

def backward(*args, **kwargs):#grad_y=None, gl=1):
    model.backward(*args, **kwargs)#grad_y, gl)

def update(**kwargs):
    model.update(**kwargs)

def loss(y, t):
    return model.loss(y, t)

def loss_function(y, t):
    return model.loss(y, t)

def generate2(seed, length=200,
        beta=2, skip_ids=None, end_id=None, stochastic=True, verbose=False, extension=False):
    return model.generate(seed, length,
                          beta, skip_ids, end_id, stochastic, verbose, extension)

# -- 学習結果の保存 --
def save_parameters(file_name):
    model.save_parameters(file_name)
    
# -- 学習結果の継承 --
def load_parameters(file_name):
    return model.load_parameters(file_name)

class RNN_r(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_l(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_g(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_rf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_lf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_gf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_rr(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_ll(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_gg(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_rrf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_llf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_ggf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_rrr(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_lll(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_ggg(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_rrrf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_lllf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_gggf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_erf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)
        
class RNN_elf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)
        
class RNN_egf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)
        
class RNN_errf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_ellf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)
        
class RNN_eggf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)
        
class RNN_elllf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)
        
class RNN_egggf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_ellllf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_er(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_err(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_el(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_ell(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_elll(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_laf(RNN_With_Attention_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)
        
class RNN_llaf(RNN_With_Attention_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_lllaf(RNN_With_Attention_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_els(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_ells(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_ellls(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_elsf(RNN_With_Attention_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)
   
class RNN_ellsf(RNN_With_Attention_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)
        
class RNN_elllsf(RNN_With_Attention_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)
        
class RNN_elaf(RNN_With_Attention_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)
   
class RNN_ellaf(RNN_With_Attention_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)
        
class RNN_elllaf(RNN_With_Attention_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_llllllllf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_ellllllf(RNN_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_layers(**kwargs)

class RNN_ela(RNN_Base):
    """ 従来互換維持のため """
    def __init__(self, *args, **kwargs):
        print("Use RNN_els instead of RNN_ela", args, kwargs)
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = 'RNN_els' # 仮に置き換え
        self.configure_layers(**kwargs)
        self.__class__.__name__ = 'RNN_ela' # 元に戻す
        
class RNN_ella(RNN_Base):
    """ 従来互換維持のため """
    def __init__(self, *args, **kwargs):
        print("Use RNN_ells instead of RNN_ella")
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = 'RNN_ells' # 仮に置き換え
        self.configure_layers(**kwargs)
        self.__class__.__name__ = 'RNN_ella' # 元に戻す


if __name__=='__main__':
    print('\n#### all cast ####')
    import inspect
    import sys
    current_module = sys.modules[__name__]
    classes = map(lambda x:x[0],inspect.getmembers(current_module,inspect.isclass))
    classes = list(classes)
    for c in classes:
        print(c)
        #if c[:4] == 'RNN_' and c[-2]!='a' and c[-5:]!='_bkup':
        if c[:4] == 'RNN_' and c[-5:]!='_bkup':
            pass
            model = eval(c)()
            model.summary()
