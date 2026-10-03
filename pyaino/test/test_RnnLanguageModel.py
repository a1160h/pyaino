# test_RnnLanguageModel
# 20261003
#
# Smoke tests for RnnLanguageModel

from pyaino.Config import *
from pyaino.RnnLanguageModel import RnnLanguageModel


def smoke_basic(rnn_type):
    print()
    print('--- basic', rnn_type, '---')

    set_seed(200)
    np = Config.np

    vocab_size = 32
    emb_dim = 8
    hidden_size = 8

    model = RnnLanguageModel(
        vocab_size,
        emb_dim=emb_dim,
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        n_layer=1,
        residual=0.0,
        stateful=True,
        unify=True,
        rms=True,
        optimize='AdamT',
        w_decay=0.01,
    )

    data = np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1],
    ], dtype='int32')

    x = data[:, :-1]
    t = data[:, 1:]

    model.reset_state()
    max_index, loss = model.forward(x, t)

    assert max_index.shape == t.shape
    assert model.rnns[0].r0.shape == (2, hidden_size)
    assert np.isfinite(loss)

    model.backward()

    assert model.embed.parameters.grad_w.shape == (vocab_size, emb_dim)

    factor = {'RNN': 1, 'GRU': 3, 'LSTM': 4}[rnn_type]

    assert model.rnns[0].parameters.grad_w.shape == (
        emb_dim, factor * hidden_size
    )
    assert model.rnns[0].parameters.grad_v.shape == (
        hidden_size, factor * hidden_size
    )
    assert model.rnns[0].parameters.grad_b.shape == (
        factor * hidden_size,
    )

    w0 = model.rnns[0].parameters.w.copy()
    model.update(eta=0.001)

    dw = float(np.max(np.abs(model.rnns[0].parameters.w - w0)))
    assert dw > 0

    model.reset_state()
    assert model.rnns[0].r0 is None
    assert model.rnns[0].c0 is None

    max_index, max_logit = model.forward(x)

    assert max_index.shape == x.shape
    assert max_logit.shape == x.shape

    print('loss =', float(loss))
    print('max |dw| =', dw)
    print('basic smoke passed:', rnn_type)


def smoke_stateful_split(rnn_type):
    print()
    print('--- stateful split', rnn_type, '---')

    set_seed(200)
    np = Config.np

    model = RnnLanguageModel(
        vocab_size=32,
        emb_dim=8,
        hidden_size=8,
        rnn_type=rnn_type,
        n_layer=2,
        residual=0.0,
        stateful=True,
        unify=True,
        rms=True,
        optimize='AdamT',
        w_decay=0.01,
    )

    data = np.array([[1, 2, 3, 4, 5, 6]], dtype='int32')

    # 一括
    model.reset_state()
    x = model.embed.forward(data)

    y_full = x
    for rnn in model.rnns:
        y_full = rnn.forward(y_full)

    state_full = model.get_state()

    # 2分割
    model.reset_state()

    xa = model.embed.forward(data[:, :3])
    ya = xa
    for rnn in model.rnns:
        ya = rnn.forward(ya)

    xb = model.embed.forward(data[:, 3:])
    yb = xb
    for rnn in model.rnns:
        yb = rnn.forward(yb)

    state_split = model.get_state()

    assert np.allclose(y_full[:, 3:, :], yb, atol=1e-6)

    for (rf, cf), (rs, cs) in zip(state_full, state_split):
        assert np.allclose(rf, rs, atol=1e-6)
        if cf is not None or cs is not None:
            assert np.allclose(cf, cs, atol=1e-6)

    print(
        'output diff =',
        float(np.max(np.abs(y_full[:, 3:, :] - yb)))
    )
    print('stateful split passed:', rnn_type)


def smoke_multilayer_residual(rnn_type):
    print()
    print('--- multilayer residual', rnn_type, '---')

    set_seed(200)
    np = Config.np

    model = RnnLanguageModel(
        vocab_size=32,
        emb_dim=8,
        hidden_size=8,
        rnn_type=rnn_type,
        n_layer=3,
        residual=0.5,
        stateful=True,
        unify=True,
        rms=True,
        optimize='AdamT',
        w_decay=0.01,
    )

    data = np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1],
    ], dtype='int32')

    x = data[:, :-1]
    t = data[:, 1:]

    model.reset_state()
    pred, loss = model.forward(x, t)

    assert pred.shape == t.shape
    assert model.residual_enabled is True

    model.backward()

    w0 = [rnn.parameters.w.copy() for rnn in model.rnns]
    model.update(eta=0.001)

    for i, (rnn, before) in enumerate(zip(model.rnns, w0)):
        dw = float(np.max(np.abs(rnn.parameters.w - before)))
        print('layer', i, 'max |dw| =', dw)
        assert dw > 0

    print('loss =', float(loss))
    print('multilayer residual passed:', rnn_type)


def smoke_generate():
    print()
    print('--- generate GRU ---')

    set_seed(200)
    np = Config.np

    model = RnnLanguageModel(
        vocab_size=32,
        emb_dim=8,
        hidden_size=8,
        rnn_type='GRU',
        n_layer=1,
        residual=0.0,
        stateful=True,
        unify=True,
        rms=True,
        optimize='AdamT',
        w_decay=0.01,
    )

    seed = np.array([1, 2, 3, 4], dtype='int32')

    gen1 = model.generate(
        seed,
        max_tokens=10,
        stochastic=False,
        flush=True,
    )
    state1 = model.get_state()

    # 手動逐次生成
    model.reset_state()

    y = model.forward(seed.reshape(1, -1))
    max_index, _ = y
    next_idx = int(max_index[0, -1])

    gen2 = seed.copy()

    while len(gen2) < 10:
        gen2 = np.append(
            gen2,
            np.array([next_idx], dtype='int32')
        )

        x = np.array([[next_idx]], dtype='int32')
        max_index, _ = model.forward(x)
        next_idx = int(max_index[0, -1])

    state2 = model.get_state()

    print('generate =', gen1)
    print('manual   =', gen2)

    assert np.array_equal(gen1, gen2)

    for (r1, c1), (r2, c2) in zip(state1, state2):
        assert np.allclose(r1, r2, atol=1e-6)
        if c1 is not None or c2 is not None:
            assert np.allclose(c1, c2, atol=1e-6)

    print('generate smoke passed.')


def tiny_overfit(rnn_type='GRU'):
    print()
    print('--- tiny overfit', rnn_type, '---')

    set_seed(200)
    np = Config.np

    model = RnnLanguageModel(
        vocab_size=16,
        emb_dim=8,
        hidden_size=8,
        rnn_type=rnn_type,
        n_layer=1,
        residual=0.0,
        stateful=True,
        unify=True,
        rms=True,
        optimize='AdamT',
        w_decay=0.0,
    )

    data = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype='int32')
    x = data[:, :-1]
    t = data[:, 1:]

    for i in range(300):
        model.reset_state()

        pred, loss = model.forward(x, t)
        model.backward()
        model.update(eta=0.01)

        if i % 50 == 0:
            print(i, float(loss), pred)

    print('final loss =', float(loss))
    print('prediction =', pred)
    print('target     =', t)

    assert np.array_equal(pred, t)

    gen = model.generate(
        np.array([1, 2, 3], dtype='int32'),
        max_tokens=7,
        stochastic=False,
        flush=True,
    )

    print('generated  =', gen)
    assert np.array_equal(
        gen,
        np.array([1, 2, 3, 4, 5, 6, 7], dtype='int32')
    )

    print('tiny overfit passed:', rnn_type)


if __name__ == '__main__':
    for rnn_type in ('RNN', 'GRU', 'LSTM'):
        smoke_basic(rnn_type)

    for rnn_type in ('RNN', 'GRU', 'LSTM'):
        smoke_stateful_split(rnn_type)

    for rnn_type in ('RNN', 'GRU', 'LSTM'):
        smoke_multilayer_residual(rnn_type)

    smoke_generate()

    # GRU版で、これまで通した過学習テストも回帰確認。
    tiny_overfit('GRU')

    print()
    print('All RnnLanguageModel smoke tests passed.')
