"""QwenModel(ModelBase) の実機確認。pyaino の test に置いて実行。"""
from pyaino.Config import Config, set_seed
from pyaino.BigramLanguageModel import ModelBase, BigramLanguageModel, QwenModel

np = Config.np


def check(name, actual, expected):
    assert bool(np.allclose(actual, expected, rtol=6e-4, atol=6e-5)), name
    print('PASS', name)


def test_qwen():
    set_seed(200)
    model = QwenModel(
        vocab_size=19, block_size=8, emb_dim=8, n_layer=1,
        n_head=(2, 1), head_dim=4, intermediate=16,
        optimize='SGD', w_decay=0.0,
    )
    assert isinstance(model, QwenModel)
    assert isinstance(model, ModelBase)

    block = model.blocks.layers[0]
    assert block.sa.config[2] == (2, 1, 1)
    assert block.sa.linear_i.bias is False
    assert block.sa.linear_o.bias is False
    assert block.sa.q_norm is not None and block.sa.k_norm is not None
    assert block.ffwd.linear_g.bias is False
    assert model.lm_head.ln_f.sb and not model.lm_head.ln_f.use_beta
    assert model.lm_head.ln_f.eps == 1e-6

    x1 = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int32)
    t1 = np.array([[2, 3, 4], [3, 4, 0]], dtype=np.int32)
    x2 = np.array([[1, 1, 3], [2, 2, 4]], dtype=np.int32)
    t2 = np.array([[3, 2, 4], [3, 4, 1]], dtype=np.int32)

    model.forward(x1, t1)
    emb = model.embed.parameters
    head = model.lm_head.linear_layer.parameters
    assert emb.w.shape == (19, 8) and head.w.shape == (8, 19)
    assert not head.bias and head.b is None and not head.scale
    a = emb.w[2, 3].copy()
    head.w[3, 2] = a + 0.125
    assert bool(emb.w[2, 3] == a + 0.125)
    emb.w[2, 3] = a
    print('PASS shared transpose view')

    model.forward(x1, t1)
    model.backward(flush=True)
    model.forward(x2, t2)
    model.backward(flush=False)
    assert emb.n_accum == head.n_accum == 2

    old_w = emb.w.copy()
    grad = emb.grad_w.copy() + head.grad_w.T
    norm = model.lm_head.ln_f
    assert norm.n_accum == 2
    old_gamma = norm.gamma.copy()
    gamma_grad = norm.ggamma.copy()

    model.update(eta=0.01)
    check('tied gradient and single embedding update', emb.w, old_w - 0.01 * grad / 2)
    check('final RMSNorm gamma update', norm.gamma, old_gamma - 0.01 * gamma_grad / 2)
    check('shared view after update', head.w, emb.w.T)


def test_old_model():
    set_seed(200)
    model = BigramLanguageModel(
        vocab_size=19, block_size=8, emb_dim=8,
        n_layer=1, n_head=2, unify=True,
        optimize='SGD', w_decay=0.0,
    )
    x = np.array([[1, 2, 3]], dtype=np.int32)
    t = np.array([[2, 3, 4]], dtype=np.int32)
    model.forward(x, t)
    model.backward()
    model.update(eta=0.01)
    assert model.lm_head.linear_layer.parameters.bias is True
    print('PASS existing BigramLanguageModel unchanged')


if __name__ == '__main__':
    print('Backend:', np.__name__, 'dtype:', Config.dtype)
    test_qwen()
    test_old_model()
    print('ALL TESTS PASSED')
