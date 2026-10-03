# ------------------------------------------------------------
# save / load round trip
# ------------------------------------------------------------

from pyaino.Config import *
import os
from pyaino import common_function as cf
from pyaino.RnnLanguageModel import RnnLanguageModel

set_seed(200)
np = Config.np

file_name = 'RnnLanguageModel_smoke.pkl'

model = RnnLanguageModel(
    vocab_size=32,
    emb_dim=8,
    hidden_size=8,
    rnn_type='GRU',
    n_layer=2,
    residual=0.5,
    stateful=True,
    unify=True,
    rms=True,
    optimize='AdamT',
    w_decay=0.01,
)

data = np.array([[1, 2, 3, 4, 5, 6]], dtype='int32')
t    = np.array([[2, 3, 4, 5, 6, 7]], dtype='int32')

model.reset_state()
pred, loss = model.forward(data, t)

model.backward()
model.update(eta=0.001)

model.reset_state()
pred1, loss1 = model.forward(data, t)

# 比較用にparameterを退避
embed_w = model.embed.parameters.w.copy()

rnn_params = []
for rnn in model.rnns:
    rnn_params.append((
        rnn.parameters.w.copy(),
        rnn.parameters.v.copy(),
        rnn.parameters.b.copy(),
    ))

head_w = model.lm_head.linear_layer.parameters.w.copy()

cf.save_parameters(file_name, model)


# ------------------------------------------------------------
# 別の初期値で同じ名前 model を作り直す
# ------------------------------------------------------------

del model

set_seed(123)

model = RnnLanguageModel(
    vocab_size=32,
    emb_dim=8,
    hidden_size=8,
    rnn_type='GRU',
    n_layer=2,
    residual=0.5,
    stateful=True,
    unify=True,
    rms=True,
    optimize='AdamT',
    w_decay=0.01,
)

cf.load_parameters(file_name, model)

model.reset_state()
pred2, loss2 = model.forward(data, t)

print('pred1 =', pred1)
print('pred2 =', pred2)
print('loss1 =', float(loss1))
print('loss2 =', float(loss2))

assert np.array_equal(pred1, pred2)
assert np.allclose(loss1, loss2, atol=1e-7)

assert np.allclose(
    embed_w,
    model.embed.parameters.w
)

for (w, v, b), rnn in zip(rnn_params, model.rnns):
    assert np.allclose(w, rnn.parameters.w)
    assert np.allclose(v, rnn.parameters.v)
    assert np.allclose(b, rnn.parameters.b)

assert np.allclose(
    head_w,
    model.lm_head.linear_layer.parameters.w
)

os.remove(file_name)

print('save/load round trip passed.')
