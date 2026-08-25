# BigramLanguageModel
# 20260825 A.Inoue

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
                 expansion=4, rms=False, optimize='AdamT', w_decay=0.001, ignore=-1, **kwargs):

        kwargs['optimize']  = optimize
        #kwargs['decayrate'] = decayrate
        kwargs['w_decay']   = w_decay
        self.embed = Neuron.PositionalEmbedding(
            vocab_size, block_size, emb_dim, **kwargs)
        self.blocks = Neuron.Sequential(
            *[sbh.TransformerBlock(emb_dim, n_head, 'tri', False, expansion, rms, **kwargs)
              for _ in range(n_layer)]
            )
        matmul = True                   
        tile_size = 1000 if vocab_size > 1000 else None 
        self.lm_head = sbh.LmHead(emb_dim, vocab_size, matmul, unify, rms, tile_size, **kwargs)

        if not unify: # 以下2項は明示的に見せる必要がある
            self.softmax = Activators.Softmax()
            self.loss_function = self.lm_head.loss_function
        self.unify = unify

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.memory = []

    def generate(self, seed, max_tokens=1000,
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
                continue
            gen_data    = np.concatenate((gen_data,    next_idx)) 
            self.memory = np.concatenate((self.memory, next_idx))
            if end_id is not None and next_idx==end_id: # end_idが出現したら打切り　
                break
        return gen_data
   
    def get_sa_result1(self, flatten=True):
        sa_result1 = []
        for bl in self.blocks.layers:
            result1 = bl.sa.attention.result1
            if result1 is not None:
                sa_result1.append(result1)
        sa_result1 = np.array(sa_result1)
        if flatten:
            sa_result1 = sa_result1.reshape(-1)
        return sa_result1

    def get_sa_result2(self, flatten=True):
        sa_result2 = []
        for bl in self.blocks.layers:
            result2 = bl.sa.attention.result2
            if result2 is not None:
                sa_result2.append(result2)
        sa_result2 = np.array(sa_result2)
        if flatten:
            sa_result2 = sa_result2.reshape(-1)
        return sa_result2

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

    def backward(self, gy=None):
        gx = self.lm_head.backward(gy)
        gx = self.blocks.backward(gx)
        self.embed.backward(gx)

    def update(self, **kwargs):
        self.embed.update(**kwargs)
        self.blocks.update(**kwargs)
        self.lm_head.update(**kwargs)

class BigramLanguageModel2(ModelBase):
    """ GPT2：Embedding と最初の Block の間に LayerNorm + 2層FFN を挿入 """
    def __init__(self, vocab_size=10000, block_size=500, emb_dim=64, n_layer=4, n_head=4,
                 expansion=4, **kwargs):

        rms      = kwargs.pop('rms',        False)
        optimize = kwargs.pop('optimize', 'AdamT')
        w_decay  = kwargs.pop('w_decay',     0.01)

        super().__init__(vocab_size, block_size, emb_dim, n_layer, n_head,
                         expansion=expansion, rms=rms, optimize=optimize, w_decay=w_decay, **kwargs)
        
        if rms:
            self.ln_pf = Neuron.RMSNormalization(optimize=optimize)
        else:    
            self.ln_pf = Neuron.LayerNormalization(optimize=optimize)
        self.pffwd = sbh.FeedForward(emb_dim, expansion, optimize=optimize, w_decay=w_decay, **kwargs)

    def forward(self, idx, targets=None, mask=None, dropout=0.0):
        x = self.embed.forward(idx)
        z = self.ln_pf.forward(x)
        z = self.pffwd.forward(z)
        x = z + x
        x = self.blocks.forward(x, mask=mask, dropout=dropout)
        y = self.lm_head(x, targets) # logits.shape=(B,T,vocab_size)
        return y
    
    def backward(self, gy=None):
        gz = self.lm_head.backward(gy)
        gz = self.blocks.backward(gz)
        gx = self.pffwd.backward(gz)
        gx = self.ln_pf.backward(gx)
        gx += gz
        self.embed.backward(gx)

    def update(self, **kwargs):
        self.embed.update(**kwargs)
        self.ln_pf.update(**kwargs)
        self.pffwd.update(**kwargs)
        self.blocks.update(**kwargs)
        self.lm_head.update(**kwargs)

class BigramLanguageModel3(BigramLanguageModel2):
    """ GPT3：Embedding と最初の Block の間に LayerNorm + 単層LinearLayer を挿入 """
    def __init__(self, vocab_size=10000, block_size=500, emb_dim=64, n_layer=4, n_head=4,
                 expansion=4, **kwargs):

        rms      = kwargs.pop('rms',        False)
        optimize = kwargs.pop('optimize', 'AdamT')
        w_decay  = kwargs.pop('w_decay',     0.01)

        ModelBase.__init__(self, vocab_size, block_size, emb_dim, n_layer, n_head,
                           expansion=expansion, rms=rms, optimize=optimize, w_decay=w_decay, **kwargs)
        
        if rms:
            self.ln_pf = Neuron.RMSNormalization(optimize=optimize)
        else:    
            self.ln_pf = Neuron.LayerNormalization(optimize=optimize)
        self.pffwd = Neuron.LinearLayer(emb_dim, emb_dim,
                                        matmul=True, optimize=optimize, w_decay=w_decay, **kwargs)
   
class BigramLanguageModel4(BigramLanguageModel2):
    """ GPT4：Embedding と最初の Block の間に 簡易LN + 単層LinearLayer を挿入 """
    def __init__(self, vocab_size=10000, block_size=500, emb_dim=64, n_layer=4, n_head=4,
                 expansion=4, **kwargs):

        rms      = kwargs.pop('rms',        False)
        optimize = kwargs.pop('optimize', 'AdamT')
        w_decay  = kwargs.pop('w_decay',     0.01)

        ModelBase.__init__(self, vocab_size, block_size, emb_dim, n_layer, n_head,
                           expansion=expansion, rms=rms, optimize=optimize, w_decay=w_decay, **kwargs)
        
        self.ln_pf = Neuron.Normalization(axis=-1)
        self.pffwd = Neuron.LinearLayer(emb_dim, emb_dim,
                                        matmul=True, optimize=optimize, w_decay=w_decay, **kwargs)

class BigramLanguageModel5(ModelBase):
    """ GPT5：Embedding と最初の Block の間に単層LinearLayer を挿入 """
    def __init__(self, vocab_size=10000, block_size=500, emb_dim=64, n_layer=4, n_head=4,
                 expansion=4, **kwargs):

        rms      = kwargs.pop('rms',        False)
        optimize = kwargs.pop('optimize', 'AdamT')
        w_decay  = kwargs.pop('w_decay',     0.01)

        super().__init__(vocab_size, block_size, emb_dim, n_layer, n_head,
                         expansion=expansion, rms=rms, optimize=optimize, w_decay=w_decay, **kwargs)
        
        self.pffwd = Neuron.LinearLayer(emb_dim, emb_dim,
                                        matmul=True, optimize=optimize, w_decay=w_decay, **kwargs)

    def forward(self, idx, targets=None, mask=None, dropout=0.0):
        x = self.embed.forward(idx)
        z = self.pffwd.forward(x)
        x = z + x
        x = self.blocks.forward(x, mask=mask, dropout=dropout)
        y = self.lm_head(x, targets) # logits.shape=(B,T,vocab_size)
        return y
    
    def backward(self, gy=None):
        gz = self.lm_head.backward(gy)
        gz = self.blocks.backward(gz)
        gx = self.pffwd.backward(gz)
        gx += gz
        self.embed.backward(gx)

    def update(self, **kwargs):
        self.embed.update(**kwargs)
        self.pffwd.update(**kwargs)
        self.blocks.update(**kwargs)
        self.lm_head.update(**kwargs)


def graph_plus(error_record=None, entropy_record=None, kld_record=None,
               entropy_offset=0, kld_offset=0, ncols=4, ylim=None):
    xmin = 0; xmax = None
    if error_record is not None:
        plt.plot(np.array(error_record).tolist(), label='error')
        xmax = len(error_record)

    if entropy_record is not None:
        entropy_recordn = np.array(entropy_record) + entropy_offset
        entropy_std = np.std(entropy_recordn, axis=-1) + entropy_offset
        entropy_range = np.max(entropy_recordn, axis=-1) - np.min(entropy_recordn, axis=-1) \
                            + entropy_offset
        if xmax is None:
            xmax = len(entropy_record)
        if entropy_offset!=0:
            plt.hlines(entropy_offset, xmin, xmax, color='gray', label='entropy_offset')

        plt.plot(entropy_recordn.tolist(), label='entropy')
        plt.plot(entropy_std.tolist(), label='std')
        plt.plot(entropy_range.tolist(), label='range')
    
    if kld_record is not None:
        kld_recordn = np.array(kld_record) + kld_offset
        if xmax is None:
            xmax = len(kld_record)
        if kld_offset!=0:
            plt.hlines(kld_offset, xmin, xmax, color='gray', label='kld_offset')
        plt.plot(kld_recordn.tolist(), label='KLD', linestyle='--')

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
    model = BigramLanguageModel2(vocab_size, block_size, emb_dim, n_layer, n_head,
                                #unify=True,
                                rms=True,
                                optimize='AdamT',
                                regularizer='AttentionRegularizer()',
                                w_decay=0.01,
                                )
    cf.get_obj_info(model)

    error_record = []; entropy_record = []
    print('学習を開始')
    for i in range(epoch):
        #print(x.shape, t.shape)
        y, l = model.forward(x, t)
        
        model.backward()
        model.update(eta=0.01)#, g_clip=0.5)

        # ここでエントロピーを採取
        entropy_record.append(model.get_sa_result1())
        
        error_record.append(float(l))
        print('Epoch: {:3d} | Error {:9.6f}'.format(i, float(l)))
        if i % 20 == 0:
            #created_data = model.generate(np.arange(10), 20)
            created_data = model.generate(list(range(10)), 20)
            print(created_data.tolist())
            
    entropy_record = np.array(entropy_record)
    cf.graph(entropy_record.tolist())
    cf.graph(error_record)
    error_record = np.array(error_record)
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
        
