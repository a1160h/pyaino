# seq2seq sequence to sequence
# Base functions for seq2seq and each seq2seq is defined
# 20250901 A.Inoue
from pyaino.Config import *
from pyaino import common_function as cf
from pyaino import RNN2 as RNN
from pyaino import LossFunctions

class Seq2seq_Base:
    def __init__(self, V, D, H, Di=None, Do=None, **kwargs):
        print(V,D,H,Di,Do,kwargs)
        # V:vocab_size, D:wordvec_size, H:hidden_size)
        if Di is None: # decoderの入力ベクトル幅
            Di = D
        if Do is None: # decoderの出力ベクトル幅 
            Do = Di    

        self.config = V, D, Di, Do, H 
          
        self.share_weight = kwargs.pop('share_weight', False)
        if self.share_weight == True and D == Di == Do == H:
            print('encoderとdecoderの各embeddingで重みが共有されます')
        else:
            self.share_weight = False

        kwargs['optimize'] = kwargs.pop('optimize', 'Adam') # 'Adam'を既定にする  
        option_for_encoder, option_for_decoder = {}, {} 

        # encoderに関わるオプション
        self.CPT = 1 # encoderの出力は通常のSeq2seqでは最終時刻のみ

        # decoderに関わるオプション
        option_for_decoder['stateful'] = kwargs.pop('stateful',      True)
        option_for_decoder['ol_act']   = kwargs.pop('ol_act',   'Softmax')#WithLoss')

        option_for_encoder.update(kwargs)
        option_for_decoder.update(kwargs)

        self.configure_encoder_decoder(option_for_encoder, option_for_decoder)

        # seq2seq全体に関わるオプション
        loss_function_name = kwargs.pop('loss', 'MeanSquaredError')
        if loss_function_name is not None:
            ignore_label = kwargs.pop('ignore', -1)
            print(loss_function_name, ignore_label)
            self.loss_function = cf.eval_in_module(loss_function_name, LossFunctions,
                                                   ignore=ignore_label)

    def configure_encoder_decoder(self, option_for_encoder, option_for_decoder):
        """ クラス名に応じてRNNの層を構成する """
        base_name = self.__class__.__base__.__name__
        name = self.__class__.__name__
        print('Configure', name, 'based on', base_name)
        V, D, Di, Do, H = self.config
        parts = name.split('_')
        if len(parts) != 3:
            raise ValueError('文字列が期待された形式ではありません．')
        if parts[0]!='seq2seq':
            raise ValueError('Name should start with seq2seq_')
        self.encoder = cf.eval_in_module('RNN_'+parts[1], RNN,
                                         V, D, H, V, **option_for_decoder)
        self.decoder = cf.eval_in_module('RNN_'+parts[2], RNN,
                                         V, D, H, V, **option_for_decoder)
 
    def summary(self):
        print('～'*39)
        print(self.__class__.__name__)
        self.encoder.summary()
        self.decoder.summary()
        print('loss_function =', self.loss_function.__class__.__name__, end =' ')
        if hasattr(self.loss_function, 'ignore_label'):
            print('ignore =', self.loss_function.ignore_label)
        else:
            print()
        print()

    def forward(self, xe, xd, t=None, *args, **kwargs): # xe:Encoder入力 xd:Decoder入力
        r0 = self.encoder.forward(xe, *args, CPT=self.CPT, **kwargs) # CPTはキーワード引数
        self.decoder.set_state(r0)  # デコーダにr0を設定     
        y = self.decoder.forward(xd, *args, **kwargs)
        if t is None:
            return y
        l = self.loss_function.forward(y, t)
        return y, l

    def backward(self, gy=None, gl=1):
        if gy is not None:
            pass
        elif self.loss_function.t is not None:
            gy = self.loss_function.backward(gl)
        else:
            raise Exception("Can't get gradient for backward." \
                            , 'gy =', gy, 'gl =', gl)
        gxd = self.decoder.backward(gy) # CPT=1の逆伝播で勾配は時系列内にbroadcastされている
        gr0 = self.decoder.get_grad_r0() # デーコーダ逆伝播の結果得られたgrad_r0を取得
        gxe = self.encoder.backward(gr0)
        return gxe, gxd

    def update(self, **kwargs):
        if self.share_weight == True:
            # layers[0]はEmbedding
            self.encoder.layers[0].grad_w += self.decoder.layers[0].grad_w
            self.encoder.update(**kwargs)
            self.decoder.update(**kwargs)
            self.decoder.layers[0].w = self.encoder.layers[0].w
        else:
            self.encoder.update(**kwargs)
            self.decoder.update(**kwargs)

    # 損失関数
    def loss(self, y, t):
        return self.loss_function.forward(y, t)

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()

    def generate(self, *args, **kwargs):
        if any(l.__class__.__name__=='Embedding' for l in self.decoder.layers):
            return self.generate_id(*args, **kwargs)
        else:
            return self.generate_data(*args, **kwargs)
    
    def generate_data(self, source, runup, length):
        """ encoderにsourceを入れr0を得て、decoderでtargetを紡ぐ """
        #source = np.array(source) sourceは、はじめからndarrayであるべき
        r0 = self.encoder.forward(source, CPT=self.CPT)
        self.decoder.set_state(r0)   # はじめにr0をセットして生成中はリセットしない
        target = self.decoder.generate(runup, length+len(runup), reset=False)
        #target = self.decoder.generate_data_from_data(runup, length+1, reset=False)
        return target[len(runup):] # 先頭はrunupなので外す

    def generate_id(self, question, start_id, end_id, length, stocastic=False, beta=2):
        """ encoderにquestionを入れr0を得て、decoderでanswerを紡ぐ """
        question = np.array(question).reshape(1, -1) 
        r0 = self.encoder.forward(question, CPT=self.CPT)
        self.decoder.set_state(r0)   # はじめにr0をセットして生成中はリセットしない
        answer = self.decoder.generate(start_id, length+1,
                                       stocastic, beta, None, end_id, reset=False)
        #answer = self.decoder.generate_id_from_id(start_id, length+1,
        #                               beta, None, end_id, stocastic, reset=False)
        return answer[1:] # 先頭はstart_idなので外す

    
    def generate_id2(self, question, start_id, end_id, length, stocastic=False, beta=2):
        """ encoderにquestionを入れr0を得て、decoderでanswerを紡ぐ """
        question = np.array(question).reshape(1, -1) 
        r0 = self.encoder.forward(question, CPT=self.CPT)
        self.decoder.set_state(r0)           # はじめにr0をセットして、
        next_one = np.array(start_id)        # start_idから紡ぎ始める　
        answer = []
        for _ in range(length): # 起動マークから始めて終了マークまで一文字ずつ綴っていく
            x = next_one.reshape((1, 1))     # 前の出力を入力に戻し、
            y = self.decoder.forward(x)      # 紡いでいく 
            # 2回目以降はr0をセットしない(statefullの動作)
            next_one = cf.select_category(y, stocastic, beta)
            answer.append(int(next_one))
            if end_id is not None and next_one==end_id: # end_idでら打切り　
                break
        return np.array(answer)

    def evaluate(self, question, correct, id_to_char, char_to_id,
                 verbose=False, is_reverse=False, language=None):
        return evaluate(self, question, correct, id_to_char, char_to_id,
                        verbose, is_reverse, language)
   
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
            print('!!構成が一致しないためパラメータは継承されません!!', title, title_f)
        return params
          
class Seq2seq_Base_with_Attention(Seq2seq_Base):
    '''
    Attentionへのkey&valueの入力のためforward,backword,generateがいずれも
    Seq2seq_Baseが使えないため、別途定義
    '''
    def forward(self, xe, xd, t=None, *args, **kwargs): # xe:Encoder入力 xd:Decoder入力
        z = self.encoder.forward(xe, *args, CPT=None, **kwargs) # CPTはキーワード引数
        r0 = z[:, -1, :]
        self.decoder.set_state(r0)
        y = self.decoder.forward(xd, z, *args, **kwargs) # z:key&value, xd:query 順序に注意
        if t is None:
            return y
        l = self.loss_function.forward(y, t)
        return y, l

    def backward(self, gy=None, gl=1):
        if gy is not None:
            pass
        elif self.loss_function.t is not None:
            gy = self.loss_function.backward(gl)
        else:
            raise Exception("Can't get gradient for backward." \
                            , 'gy =', gy, 'gl =', gl)
        gxd, gz = self.decoder.backward(gy) # gz:Attentionのkey&valueの勾配
        gr0 = self.decoder.get_grad_r0() # デーコーダ逆伝播の結果得られたgrad_r0を取得
        gz[:, -1, :] += gr0 # decoderのgr0はencoderの最終時刻出力の勾配に加算
        gxe = self.encoder.backward(gz)
        return gxe, gxd

    def generate_data(self, source, runup, length):
        """ encoderにsourceを入れzとr0を得て、decoderでtargetを紡ぐ """
        #source = np.array(source) sourceは、はじめからndarrayであるべき
        z = self.encoder.forward(source, CPT=None)
        r0 = z[:, -1, :]
        self.decoder.set_state(r0)   # はじめにr0をセットして生成中はリセットしない
        target = self.decoder.generate(runup, z, length+len(runup), reset=False)
        #target = self.decoder.generate_data_from_data(runup, length+1, reset=False)
        return target[len(runup):] # 先頭はrunupなので外す

    def generate_id(self, question, start_id, end_id, length, stocastic=False, beta=2):
        """ encoderにquestionを入れzとr0を得て、decoderでanswerを紡ぐ """
        question = np.array(question).reshape(1, -1)
        z = self.encoder.forward(question, CPT=None) # 質問文をセット　
        r0 = z[:, -1, :]
        self.decoder.set_state(r0)         # はじめにr0をセットして、
        answer = self.decoder.generate_id_from_id(start_id, z, length+1,
                                           stocastic, beta, None, end_id, reset=False)
        return answer[1:]

    def generate_id_bkup(self, question, start_id, end_id, length, stocastic=False, beta=2):
        """ encoderにquestionを入れr0とzを得て、decoderでanswerを紡ぐ """
        question = np.array(question).reshape(1, -1) 
        z = self.encoder.forward(question, CPT=None) # 質問文をセット　
        r0 = z[:, -1, :]
        self.decoder.set_state(r0)         # はじめにr0をセットして、
        next_one = np.array(start_id)      # start_idから紡ぎ始める　
        answer = []               
        for _ in range(length): # 起動マークから始めて終了マークまで一文字ずつ綴っていく
            x = next_one.reshape((1, 1))   # xは出力を戻して紡いでいく
            y = self.decoder.forward(x, z) # その間中zは変わらない
            # 2回目以降はr0をセットしない(statefullの動作)
            next_one = cf.select_category(y, stocastic, beta)
            answer.append(int(next_one))
            if end_id is not None and next_one==end_id: # end_idで打切り　
                break

        return np.array(answer)

class Seq2seq_Base_with_SelfAttention(Seq2seq_Base):
    """ エンコーダにSelfAttentionを備えたseq2seq """
    #encoderはSelfAttentionがあるためCPTを指定しない
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CPT = None # CPTによらずencoderの出力はSelfAttentionにより一時刻のみ 

class Seq2seq_Base_Data_Out_with_SelfAttention(Seq2seq_Base):
    """ データ入出力のデコーダを備えたseq2seq """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CPT = None # CPTによらずencoderの出力はSelfAttentionにより一時刻のみ 

    def configure_encoder_decoder(self, option_for_encoder, option_for_decoder):
        """ クラス名に応じてRNNの層を構成する """
        base_name = self.__class__.__base__.__name__
        name = self.__class__.__name__
        print('Configure', name, 'based on', base_name)
        V, D, Di, Do, H = self.config
        parts = name.split('_')
        if len(parts) != 3:
            raise ValueError('文字列が期待された形式ではありません．')
        if parts[0]!='seq2seq':
            raise ValueError('Name should start with seq2seq_')
        self.encoder = cf.eval_in_module('RNN_'+parts[1], RNN,
                                         V, D, H, V, **option_for_decoder)
        self.decoder = cf.eval_in_module('RNN_'+parts[2], RNN,
                                         Di, H, Do, **option_for_decoder)

class Seq2seq_Base_Data_Out_with_Attention(Seq2seq_Base_with_Attention):
    """ データ入出力のデコーダを備えたseq2seq """
    def configure_encoder_decoder(self, option_for_encoder, option_for_decoder):
        """ クラス名に応じてRNNの層を構成する """
        base_name = self.__class__.__base__.__name__
        name = self.__class__.__name__
        print('Configure', name, 'based on', base_name)
        V, D, Di, Do, H = self.config
        parts = name.split('_')
        if len(parts) != 3:
            raise ValueError('文字列が期待された形式ではありません．')
        if parts[0]!='seq2seq':
            raise ValueError('Name should start with seq2seq_')
        self.encoder = cf.eval_in_module('RNN_'+parts[1], RNN,
                                         V, D, H, V, **option_for_decoder)
        self.decoder = cf.eval_in_module('RNN_'+parts[2], RNN,
                                         Di, H, Do, **option_for_decoder)

# -- 外部から直接アクセス(共通) --
def build(Class, V, D, H, **kwargs):
    print(Class, V, D, H, kwargs)
    global model
    model = Class(V, D, H, **kwargs)

def forward(xe, xd, *args, **kwargs):
    return model.forward(xe, xd, *args, **kwargs)

def backward(*args, **kwargs):
    return model.backward(*args, **kwargs)

def update(**kwargs):
    return model.update(**kwargs)

def loss(y, t):
    return model.loss(y, t)
   
def reset_state():
    model.reset_state()

# -- 学習結果の保存 --
def save_parameters(file_name):
    model.save_parameters(file_name)

# -- 学習結果の継承 --
def load_parameters(file_name):
    return model.load_parameters(file_name) 

def summary():
    model.summary()

def print_func(data, id_to_char, en):
    if en:
        cf.print_en_texts(data, id_to_char)
    else:
        data = ''.join([id_to_char[int(c)] for c in data])
        print(data)

def evaluate(model, question, correct, id_to_char, char_to_id,
             verbose=False, is_reverse=False, language=None):

    if language in ('J','j','japanese','Japanese','Japan', 'japan'):
        flag = evaluate_language_model(model, question, correct,
                                       id_to_char, char_to_id,
                                       verbose, is_reverse, False)
    elif language in ('E','e','english','English'):
        flag = evaluate_language_model(model, question, correct,
                                       id_to_char, char_to_id,
                                       verbose, is_reverse, True)
    else:
        print('Not supported yet for specified case.')
        flag = False
        
    return flag
    
def evaluate_language_model(model, question, correct, id_to_char, char_to_id,
             verbose=False, is_reverse=False, en=False):
     
    correct = np.array(correct)   # cupy対応
    start_id = correct[:, 0]      # 頭の区切り文字
    correct = correct[:, 1:].reshape(-1)
    end_id = char_to_id.get('\n') # 無い場合はNone
    guess = model.generate(question, start_id, end_id, len(correct), False)

    # 正解値をend_idまでの範囲に成形
    if end_id is not None:
        idx, = np.where(correct==end_id)
        if len(idx) > 0:
            idx = int(idx[0])         # 複数個のend_idに対応
            correct = correct[:idx+1] # idxそのものを含む
            
    flag = str(correct) == str(guess) # 長さの違いも含めて文字列比較
    
    if verbose:
        question= question.reshape(-1)
        if is_reverse:
            question = question[::-1]
        
        print('Q', end=' ')
        print_func(question, id_to_char, en)
        print('T', end=' ')
        print_func(correct, id_to_char, en)
        mark = 'O' if flag else 'X'
        print(mark, end=' ')
        print_func(guess, id_to_char, en)
        print('---', correct.size, guess.size, '---')

    return flag


class seq2seq_r_rf(Seq2seq_Base):
    pass

class seq2seq_l_lf(Seq2seq_Base):
    pass

class seq2seq_er_erf(Seq2seq_Base):
    pass

class seq2seq_err_errf(Seq2seq_Base):
    pass

class seq2seq_el_elf(Seq2seq_Base):
    pass

class seq2seq_ell_ellf(Seq2seq_Base):
    pass

class seq2seq_el_elaf(Seq2seq_Base_with_Attention):
    pass

class seq2seq_el_elnaf(Seq2seq_Base_with_Attention):
    pass

class seq2seq_ell_ellaf(Seq2seq_Base_with_Attention):
    pass

class seq2seq_elll_elllaf(Seq2seq_Base_with_Attention):
    pass

class seq2seq_els_elf(Seq2seq_Base_with_SelfAttention):
    pass

class seq2seq_ells_ellf(Seq2seq_Base_with_SelfAttention):
    pass

class seq2seq_el_laf(Seq2seq_Base_Data_Out_with_Attention):
    pass

class seq2seq_ell_llaf(Seq2seq_Base_Data_Out_with_Attention):
    pass

class seq2seq_elll_lllaf(Seq2seq_Base_Data_Out_with_Attention):
    pass

class seq2seq_ells_llf(Seq2seq_Base_Data_Out_with_SelfAttention):
    pass

if __name__=='__main__':
    print('\n#### all cast ####')
    import inspect
    import sys
    current_module = sys.modules[__name__]
    classes = map(lambda x:x[0],inspect.getmembers(current_module,inspect.isclass))
    for c in list(classes):
        if c[:7]=='seq2seq':
            print(c)
