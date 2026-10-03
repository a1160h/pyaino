# RnnLanguageModel
# 20261003
#
# Recurrent language model for pyaino
# Embedding -> RNN/GRU/LSTM x N -> LmHead

from pyaino.Config import *
from pyaino import Neuron as nn
from pyaino import stems_blocks_heads as sbh
from pyaino import Activators
from pyaino import common_function as cf


class RnnLanguageModel:
    """
    Recurrent language model.

    Structure:
        Embedding
          -> RNN / GRU / LSTM x n_layer
          -> LmHead

    residual:
        x_next = rnn(x) + residual * x
        is applied to every recurrent layer when residual != 0.

    stateful:
        Passed to every recurrent layer.

    unify:
        True  -> LmHead uses LinearLayerCrossEntropy.
                 Integer targets can be supplied directly.
                 Inference is greedy because LmHead returns max_index/max_logit.
        False -> LmHead returns full logits and stochastic generation is available.
    """
    def __init__(self, vocab_size=10000,
                 emb_dim=64,
                 rnn_type='RNN',
                 n_layer=1,
                 residual=0.0,
                 stateful=True,
                 unify=True,
                 rms=True,
                 optimize='AdamT',
                 w_decay=0.001,
                 **kwargs):

        if not isinstance(n_layer, int) or n_layer < 1:
            raise ValueError(
                f'n_layer must be a positive integer, but {n_layer!r} was specified.')
        if not isinstance(residual, (int, float)):
            raise TypeError(
                f'residual must be int or float, but {type(residual).__name__} was specified.')

        options = dict(kwargs)
        options['optimize'] = optimize
        options['w_decay'] = w_decay

        self.embed = nn.Embedding(vocab_size, emb_dim, **options)

        self.rnns = [
            cf.eval_in_module(rnn_type, nn, emb_dim, emb_dim, stateful=stateful, **options)
            for _ in range(n_layer)
        ]

        self.residual_enabled = residual != 0

        tile_size = 1000 if vocab_size > 1000 else None 
        self.lm_head = sbh.LmHead(
            emb_dim, vocab_size,
            matmul=True,
            unify=unify,
            rms=rms,
            tile_size=tile_size,
            **options
        )

        if not unify:
            self.softmax = Activators.Softmax()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.rnn_type = rnn_type
        self.n_layer = n_layer
        self.residual = residual
        self.stateful = stateful
        self.unify = unify

        # common_function.save_parameters(verbose=True) からも見やすいように保持
        self.config = vocab_size, emb_dim, rnn_type, n_layer, residual, stateful


    def forward(self, idx, targets=None, mask=None, dropout=0.0):
        x = self.embed.forward(idx, mask=mask)

        for rnn in self.rnns:
            z = rnn.forward(x, mask=mask, dropout=dropout)

            if self.residual_enabled:
                x = z + self.residual * x
            else:
                x = z

        return self.lm_head(x, targets)


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


    def backward(self, gy=None):
        gx = self.lm_head.backward(gy)

        for rnn in reversed(self.rnns):

            # x_next = rnn(x) + residual*x
            # 下流から来た gx は両枝へ同じように流れる。
            gz = gx
            gx = rnn.backward(gz)

            if self.residual_enabled:
                gx = gx + self.residual * gz

        self.embed.backward(gx)


    def update(self, **kwargs):
        self.embed.update(**kwargs)

        for rnn in self.rnns:
            rnn.update(**kwargs)

        self.lm_head.update(**kwargs)


    def reset_state(self):
        for rnn in self.rnns:
            rnn.reset_state()


    def get_state(self, copy=True):
        """
        Return [(r0, c0), ...] for all recurrent layers.
        RNN/GRU do not use c0 internally, but RnnBaseLayer keeps the slot.
        """
        states = []

        for rnn in self.rnns:
            r0 = rnn.r0
            c0 = rnn.c0

            if copy:
                if r0 is not None:
                    r0 = r0.copy()
                if c0 is not None:
                    c0 = c0.copy()

            states.append((r0, c0))

        return states


    def accommodate(self):
        """Expand vocabulary-dependent input/output layers."""
        self.embed.accommodate()
        self.lm_head.accommodate()


    def summary(self):
        print(self.__class__.__name__)
        print('  vocab_size =', self.vocab_size)
        print('  emb_dim    =', self.emb_dim)
        print('  rnn_type   =', self.rnn_type)
        print('  n_layer    =', self.n_layer)
        print('  residual   =', self.residual)
        print('  stateful   =', self.stateful)
        print('  unify      =', self.unify)

        for i, rnn in enumerate(self.rnns):
            print(
                f'  layer[{i}]   = {rnn.__class__.__name__} '
                f'{rnn.config[:2]} residual={self.residual_enabled}'
            )
        
    def _select_next(self, y, stochastic=False, beta=2, skip_ids=None):
        """
        Select the next token from LmHead output.

        unify=True:
            y == (max_index, max_logit), so greedy generation only.
        unify=False:
            y == logits, so stochastic selection and skip_ids are available.
        """
        if self.unify:
            if stochastic:
                raise ValueError(
                    'stochastic=True requires unify=False because '
                    'LinearLayerCrossEntropy inference returns only max_index/max_logit.'
                )
            if skip_ids:
                raise ValueError(
                    'skip_ids requires unify=False because '
                    'LinearLayerCrossEntropy inference does not expose full logits.'
                )

            max_index, _ = y
            return int(max_index[0, -1])

        logits = y[:, -1, :].reshape(-1).copy()

        if skip_ids is not None:
            for idx in skip_ids:
                logits[int(idx)] = -np.inf

        probs = self.softmax(logits)
        return int(cf.select_category(probs, stochastic, beta))


    def generate(self, seed, max_tokens=1000,
                 stochastic=False, beta=2,
                 skip_ids=None, end_id=None,
                 flush=True):
        """
        Generate a single token sequence.

        max_tokens is the total length including seed, matching the convention
        used by pyaino's GPT language model.

        With stateful=True:
            seed is processed once, then one generated token at a time.

        With stateful=False:
            the whole generated prefix is re-evaluated each step.

        If end_id is predicted, it is included in the returned sequence.
        The recurrent state is left immediately before consuming end_id.
        This is convenient when the caller wants to continue a conversation
        without committing EOS to the recurrent state.
        """
        if not isinstance(seed, np.ndarray):
            seed = np.array(seed, dtype='int32')
        else:
            seed = seed.astype('int32', copy=False)

        if seed.ndim == 1:
            seed = seed.reshape(1, -1)

        if seed.ndim != 2 or seed.shape[0] != 1:
            raise ValueError(
                f'generate() supports one sequence only, but seed.shape={seed.shape}.'
            )

        if seed.shape[1] == 0:
            raise ValueError('seed must contain at least one token.')

        if max_tokens < seed.shape[1]:
            raise ValueError(
                f'max_tokens ({max_tokens}) must be >= seed length ({seed.shape[1]}).'
            )

        if flush:
            self.reset_state()

        gen_data = seed.reshape(-1).copy()

        # seed の末尾から最初の次tokenを得る
        y = self.forward(seed)
        next_idx = self._select_next(
            y,
            stochastic=stochastic,
            beta=beta,
            skip_ids=skip_ids,
        )

        while len(gen_data) < max_tokens:
            gen_data = np.append(
                gen_data,
                np.array([next_idx], dtype='int32')
            )

            if end_id is not None and next_idx == end_id:
                break

            if self.stateful:
                # 既存stateを使い、生成した1 tokenだけを追加処理する。
                x = np.array([[next_idx]], dtype='int32')
                y = self.forward(x)
            else:
                # stateを保持しない設定では、prefix全体を毎回処理する。
                self.reset_state()
                y = self.forward(gen_data.reshape(1, -1))

            next_idx = self._select_next(
                y,
                stochastic=stochastic,
                beta=beta,
                skip_ids=skip_ids,
            )

        return gen_data

