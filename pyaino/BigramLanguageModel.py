# BigramLanguageModel
# 20261126 A.Inoue

from pyaino.Config import *
#set_np('numpy'); np=Config.np
from pyaino import Neuron
from pyaino import stems_blocks_heads as sbh
from pyaino import Activators
from pyaino import LossFunctions as lf
from pyaino import common_function as cf
import copy
import matplotlib.pyplot as plt


class ModelBase:
    """ 共通ベース """
    def __init__(self, vocab_size=10000, block_size=500, emb_dim=64, n_layer=4, n_head=4, unify=False,
                 ffwd='ffn', intermediate=None, expansion=4,
                 rms=False, ignore=-1,
                 positional_embedding=True, rope=False,
                 bias=(False, True), lm_head_bias=True, **kwargs):

        self.emb_dim = emb_dim
        self.intermediate = intermediate
        self.expansion = expansion
        self.rms = rms
        kwargs.setdefault('optimize', 'AdamT')
        kwargs.setdefault('w_decay',    0.001)
        self.kwargs = kwargs
        chunk_size = kwargs.pop('chunk_size', None) # Attention用

        # -- Embedding -------------------------------------------------
        if positional_embedding:
            self.embed = Neuron.PositionalEmbedding(
                vocab_size, block_size, emb_dim, **kwargs)
        else:
            self.embed = Neuron.Embedding(
                vocab_size, emb_dim, **kwargs)
        
        # -- Transformer blocks ----------------------------------------
        self.blocks = Neuron.Sequential(
            *[sbh.TransformerBlock(
                emb_dim, n_head, 'tri', False, ffwd, intermediate, expansion,
                rms, chunk_size=chunk_size, rope=rope, bias=bias, **kwargs)
              for _ in range(n_layer)]
            )

        # -- Lm Head ---------------------------------------------------
        matmul = True                   
        tile_size = 1000 if vocab_size > 1000 else None 
        self.lm_head = sbh.LmHead(
            emb_dim, vocab_size, matmul, unify, rms, tile_size,
            bias=lm_head_bias, **kwargs)

        if not unify: # 以下2項は明示的に見せる必要がある
            self.softmax = Activators.Softmax()
            self.loss_function = self.lm_head.loss_function
        self.unify = unify
        
        # -- others ---------------------------------------------------
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.memory = []

    def generate(self, seed, max_tokens=1000,
                 stochastic=False, beta=2, skip_ids=None, end_ids=None,
                 memory_size=1000, flush=True):
        """
        入力されたidx列の末尾のblock_size長を入力してその次idxを得、
        それを結合したidx列からまたその末尾を繰り返してそのまた次、
        というようにして、入力されたidx列に続くidx列を生成する

        """
        block_size = self.block_size
        if not isinstance(self.memory, np.ndarray):
            self.memory = np.array(self.memory, dtype='int32')
        if not isinstance(seed, np.ndarray):
            seed = np.array(seed)
        if flush:
            self.memory = seed.copy()
        else:    
            self.memory = np.concatenate((self.memory, seed))
        if len(self.memory) > memory_size:
            self.memory = self.memory[-memory_size:]
        gen_data = seed.copy()

        for i in range(max_tokens - len(seed)):
            if self.unify:
                max_index, _ = self.forward(self.memory[-block_size:].reshape(1,-1)) 
                next_idx = max_index[:,-1] # 末尾を選ぶ
            else:
                logits = self.forward(self.memory[-block_size:].reshape(1,-1)) # バッチ軸追加して順伝播
                # 次の単語の予測確率を得る
                probs = self.softmax(logits[:, -1, :]) # (B, C)
                # 上記確率にもとづいて次のidxをサンプリング
                probs = probs.reshape(-1)
                
                # skip_idsの指すものを選択候補から除外
                if skip_ids is not None: 
                    probs[skip_ids] = 0
                    probs /= probs.sum()

                next_idx = cf.select_category(probs, stochastic, beta)
            gen_data    = np.concatenate((gen_data,    next_idx)) 
            self.memory = np.concatenate((self.memory, next_idx))
            if end_ids is not None and next_idx[0] in end_ids: # next_idxは1要素配列 end_idが出現したら打切り　
                break

        return gen_data


    def generate_bkup(self, seed, max_tokens=1000,
                 stochastic=False, beta=2, skip_ids=None, end_id=None,
                 memory_size=1000, flush=True):
        """
        入力されたidx列の末尾のblock_size長を入力してその次idxを得、
        それを結合したidx列からまたその末尾を繰り返してそのまた次、
        というようにして、入力されたidx列に続くidx列を生成する

        """
        block_size = self.block_size
        if not isinstance(self.memory, np.ndarray):
            self.memory = np.array(self.memory, dtype='int32')
        if not isinstance(seed, np.ndarray):
            seed = np.array(seed)
        if flush:
            self.memory = seed.copy()
        else:    
            self.memory = np.concatenate((self.memory, seed))
        if len(self.memory) > memory_size:
            self.memory = self.memory[-memory_size:]
        gen_data = seed.copy()

        skip_count = 0 ### 
        
        for i in range(max_tokens - len(seed)):
            if self.unify:
                max_index, _ = self.forward(self.memory[-block_size:].reshape(1,-1)) 
                next_idx = max_index[:,-1] # 末尾を選ぶ
            else:
                logits = self.forward(self.memory[-block_size:].reshape(1,-1)) # バッチ軸追加して順伝播
                # 次の単語の予測確率を得る
                probs = self.softmax(logits[:, -1, :]) # (B, C)
                # 上記確率にもとづいて次のidxをサンプリング
                probs = probs.reshape(-1)
                next_idx = cf.select_category(probs, stochastic, beta)
                
            if skip_ids is not None and next_idx in skip_ids:

                skip_count += 1 ###
                
                continue
            gen_data    = np.concatenate((gen_data,    next_idx)) 
            self.memory = np.concatenate((self.memory, next_idx))
            if end_id is not None and next_idx==end_id: # end_idが出現したら打切り　
                break

        print('generate loop =', i + 1, 'skip count =', skip_count) ###            
            
        return gen_data
  
    def get_sa_records(self, flatten=True):
        layer_records = []
        for bl in self.blocks.layers:
            regularizer = bl.sa.attention.regularizer
            if regularizer is not None:
                layer_records.append(regularizer.get_records())

        if not layer_records:
            return []

        n_record = max(len(records) for records in layer_records)
        sa_records = []

        for i in range(n_record):
            records = []

            for layer_record in layer_records:
                if i >= len(layer_record) or len(layer_record[i]) == 0:
                    continue

                record = np.array(layer_record[i])
                if flatten:
                    record = record.reshape(record.shape[0], -1)
                records.append(record)

            if not records:
                sa_records.append(None)
            elif flatten:
                sa_records.append(np.concatenate(records, axis=1))
            else:
                sa_records.append(np.array(records))

        return sa_records

    def accommodate(self):
        """モデルのEmbeddingと出力層を現在の語彙数へ拡張する。"""
        self.embed.accommodate()
        self.lm_head.accommodate()
    
class BigramLanguageModel(ModelBase):
    """ GPT：基本の構成（Embedding → Blocks → lm_head） """

    def forward(self, idx, targets=None, mask=None, dropout=0.0):
        x = self.embed.forward(idx)
        x = self.blocks.forward(x, mask=mask, dropout=dropout)
        y = self.lm_head(x, targets) # logits.shape=(B,T,vocab_size)
        return y

    def backward(self, gy=None, flush=True):
        gx = self.lm_head.backward(gy, flush=flush)
        gx = self.blocks.backward(gx, flush=flush)
        self.embed.backward(gx, flush=flush)

    def update(self, **kwargs):
        self.embed.update(**kwargs)
        self.blocks.update(**kwargs)
        self.lm_head.update(**kwargs)

class BigramLanguageModel2(ModelBase):
    """ GPT2：Embedding と最初の Block の間に LayerNorm + 2層FFN を挿入 """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        kwargs = self.kwargs        
        expansion = self.expansion
        emb_dim = self.emb_dim
        intermediate = self.intermediate

        if self.rms:
            self.ln_pf = Neuron.RMSNormalization(optimize=kwargs['optimize'])
        else:    
            self.ln_pf = Neuron.LayerNormalization(optimize=kwargs['optimize'])

        self.pffwd = sbh.FeedForward(emb_dim, intermediate, expansion, **kwargs)

    def forward(self, idx, targets=None, mask=None, dropout=0.0):
        x = self.embed.forward(idx)
        z = self.ln_pf.forward(x)
        z = self.pffwd.forward(z)
        x = z + x
        x = self.blocks.forward(x, mask=mask, dropout=dropout)
        y = self.lm_head(x, targets) # logits.shape=(B,T,vocab_size)
        return y
    
    def backward(self, gy=None, flush=True):
        gz = self.lm_head.backward(gy, flush=flush)
        gz = self.blocks.backward(gz, flush=flush)
        gx = self.pffwd.backward(gz, flush=flush)
        gx = self.ln_pf.backward(gx, flush=flush)
        gx += gz
        self.embed.backward(gx, flush=flush)

    def update(self, **kwargs):
        self.embed.update(**kwargs)
        self.ln_pf.update(**kwargs)
        self.pffwd.update(**kwargs)
        self.blocks.update(**kwargs)
        self.lm_head.update(**kwargs)

class BigramLanguageModel2a(BigramLanguageModel2):
    """ GPT2：Embedding と最初の Block の間に LayerNorm + 2層FFN を挿入 """
    def __init__(self, *args, **kwargs):

        ModelBase.__init__(self, *args, **kwargs)

        kwargs = self.kwargs
        expansion = self.expansion
        emb_dim = self.emb_dim
        intermediate = self.intermediate
        
        if self.rms:
            self.ln_pf = Neuron.RMSNormalization(optimize=kwargs['optimize'])
        else:    
            self.ln_pf = Neuron.LayerNormalization(optimize=kwargs['optimize'])

        self.pffwd = sbh.CausalConvFeedForward(emb_dim, intermediate, expansion, **kwargs)

class BigramLanguageModel3(BigramLanguageModel2):
    """ GPT3：Embedding と最初の Block の間に LayerNorm + 単層LinearLayer を挿入 """
    def __init__(self, *args, **kwargs):

        ModelBase.__init__(self, *args, **kwargs)
        
        kwargs = self.kwargs        
        emb_dim = self.emb_dim

        if self.rms:
            self.ln_pf = Neuron.RMSNormalization(optimize=kwargs['optimize'])
        else:    
            self.ln_pf = Neuron.LayerNormalization(optimize=kwargs['optimize'])
        self.pffwd = Neuron.LinearLayer(emb_dim, emb_dim, matmul=True, **kwargs)
   
class BigramLanguageModel4(BigramLanguageModel2):
    """ GPT4：Embedding と最初の Block の間に 簡易LN + 単層LinearLayer を挿入 """
    def __init__(self, *args, **kwargs):

        ModelBase.__init__(self, *args, **kwargs)

        kwargs = self.kwargs        
        emb_dim = self.emb_dim
        
        self.ln_pf = Neuron.LayerNormalization(axis=-1, scale_and_bias=False)
        self.pffwd = Neuron.LinearLayer(emb_dim, emb_dim, matmul=True, **kwargs)

class BigramLanguageModel5(ModelBase):
    """ GPT5：Embedding と最初の Block の間に単層LinearLayer を挿入 """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        kwargs = self.kwargs
        emb_dim = self.emb_dim
        
        self.pffwd = Neuron.LinearLayer(emb_dim, emb_dim, matmul=True, **kwargs)

    def forward(self, idx, targets=None, mask=None, dropout=0.0):
        x = self.embed.forward(idx)
        z = self.pffwd.forward(x)
        x = z + x
        x = self.blocks.forward(x, mask=mask, dropout=dropout)
        y = self.lm_head(x, targets) # logits.shape=(B,T,vocab_size)
        return y
    
    def backward(self, gy=None, flush=True):
        gz = self.lm_head.backward(gy, flush=flush)
        gz = self.blocks.backward(gz, flush=flush)
        gx = self.pffwd.backward(gz, flush=flush)
        gx += gz
        self.embed.backward(gx, flush=flush)

    def update(self, **kwargs):
        self.embed.update(**kwargs)
        self.pffwd.update(**kwargs)
        self.blocks.update(**kwargs)
        self.lm_head.update(**kwargs)


class QwenModel(ModelBase):
    """Qwen3構成。共通のModelBaseを使い、重み共有だけをこのクラスで管理する。"""
    def __init__(self, *args, **kwargs):

        model_settings = dict(
            vocab_size=151936, block_size=40960, emb_dim=1024,
            n_layer=28, n_head=(16, 8), head_dim=128,
            intermediate=3072, unify=True,
            ffwd='swiglu', activate=None,
            rms=True, scale_and_bias='scale', eps=1e-6,
            positional_embedding=False, rope=True,
            rope_mode='half', rope_base=1e6,
            qk_norm=True, qk_norm_eps=1e-6,
            bias=(False, False), lm_head_bias=False, scale=False,
        )

        # 位置引数もmodel_settingsに組み込む
        names = ('vocab_size', 'block_size', 'emb_dim', 'n_layer', 'n_head',
                 'unify', 'ffwd', 'intermediate')

        model_settings.update(zip(names, args))
        model_settings.update(kwargs)

        super().__init__(**model_settings)

        
    def tie_weights(self):
        w = self.embed.parameters()
        head = self.lm_head.linear_layer
        head.config = (w.shape[1], w.shape[0])
        head.parameters.w = w.T

    def merge_tied_gradient(self):
        e = self.embed.parameters
        h = self.lm_head.linear_layer.parameters
        if e.n_accum < 1 or e.n_accum != h.n_accum:
            raise RuntimeError('Tied gradient accumulation count mismatch')
        e.grad_w += h.grad_w.T

    def forward(self, idx, targets=None, mask=None, dropout=0.0):
        x = self.embed.forward(idx)
        x = self.blocks.forward(x, mask=mask, dropout=dropout)
        self.tie_weights()
        y = self.lm_head(x, targets)
        return y

    def backward(self, gy=None, flush=True):
        gx = self.lm_head.backward(gy, flush=flush)
        gx = self.blocks.backward(gx, flush=flush)
        self.embed.backward(gx, flush=flush)

    def update(self, **kwargs):
        self.merge_tied_gradient()
        self.embed.update(**kwargs)
        self.blocks.update(**kwargs)
        self.lm_head.ln_f.update(**kwargs)

    def accommodate(self):
        self.embed.accommodate()
        self.tie_weights()
        self.vocab_size = self.embed.parameters.w.shape[0]


def graph_plus(error_record=None, entropy_record=None, kldjsd_record=None,
               entropy_offset=0, kldjsd_offset=0, ncols=4, ylim=None, kldjsd='kld'):
    xmin = 0; xmax = None
    if error_record is not None:
        plt.plot(np.array(error_record).tolist(), label='error')
        xmax = len(error_record)

    if entropy_record is not None:
        entropy_recordn = np.array(entropy_record) + entropy_offset
        if xmax is None:
            xmax = len(entropy_record)
        if entropy_offset!=0:
            plt.hlines(entropy_offset, xmin, xmax, color='gray', label='entropy_offset')

        plt.plot(entropy_recordn.tolist(), label='entropy')

        if kldjsd_record is None: # entropy飲みの場合だけ表示
            entropy_std = np.std(entropy_recordn, axis=-1) + entropy_offset
            entropy_range = np.max(entropy_recordn, axis=-1) - np.min(entropy_recordn, axis=-1) \
                                + entropy_offset
            plt.plot(entropy_std.tolist(), label='std')
            plt.plot(entropy_range.tolist(), label='range')
    
    if kldjsd_record is not None:
        kldjsd_recordn = np.array(kldjsd_record) + kldjsd_offset
        if kldjsd == 'kld':
            labelo = 'kld_offset'
            label  = 'KLD'
        else:
            labelo = 'jsd_offset'
            label = 'JSD'
         
        if xmax is None:
            xmax = len(kldjsd_record)
        if kldjsd_offset!=0:
            plt.hlines(kldjsd_offset, xmin, xmax, color='gray', label=labelo)
        plt.plot(kldjsd_recordn.tolist(), label=label, linestyle='--')

    #plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0)#, fontsize=18)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(ncols=ncols)
    plt.show()

    
if __name__=='__main__':
    # -- 諸設定 --
    block_size = 7   
    emb_dim = 8
    n_head  = 4         
    n_layer = 1        
    epoch = 301

    # -- 訓練用データ --
    data = list(range(101))
    vocab_size=len(data)

    # -- 時系列に並んだデータ=>入力、その次の１つのデータ=>正解値 --
    input_data, correct_data = cf.arrange_time_data(data, block_size, step=1)
    x = np.array(input_data)
    t = np.array(correct_data)

    # -- 各層の初期化 --
    model = BigramLanguageModel2a(vocab_size, block_size, emb_dim, n_layer, n_head,
                                 #unify=True,
                                 ffwd='ccff', 
                                 rms=True,
                                 optimize='AdamT',
                                 regularizer='AttentionRegularizer()',
                                 positional_embedding=False,
                                 rope=True,
                                 w_decay=0.01,
                                 )
    cf.get_obj_info(model)

    error_record = []
    print('学習を開始')
    for i in range(epoch):
        #print(x.shape, t.shape)
        y, l = model.forward(x, t)
        
        model.backward()
        model.update(eta=0.01)#, g_clip=0.5)
       
        error_record.append(float(l))
        print('Epoch: {:3d} | Error {:9.6f}'.format(i, float(l)))
        if i % 20 == 0:
            #created_data = model.generate(np.arange(10), 20)
            created_data = model.generate(list(range(10)), 20)
            print(created_data.tolist())
            
    sa_records = model.get_sa_records()
    entropy_record = sa_records[0] if len(sa_records) > 0 else None
    if entropy_record is not None:
        cf.graph(entropy_record.tolist())
    cf.graph(error_record)
    error_record = np.array(error_record)
    if entropy_record is not None:
        record = np.concatenate([error_record.reshape(-1,1), entropy_record], axis=1)
        cf.graph(record.tolist())
    print('結果を確認')
    created_data = model.generate(np.arange(10), 101)
    print(created_data.tolist())
    print('Is all correct =', (created_data==np.array(data)).all())

    #cf.get_param_info(model, 'model')
    params = cf.export_parameters(model)
    #print(params.keys())
    entropies = {}
    for k in params.keys():
        if params[k] is not None:
            entropies[k] = cf.get_entropy(params[k])
            print(k, 'entropy =', entropies[k])
        
