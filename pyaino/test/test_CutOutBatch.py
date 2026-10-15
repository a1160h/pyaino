from pyaino.Config import *
from pyaino import common_function as cf


def smoke_test(get_batch, name):
    print(f'\n--- {name} : shuffle={get_batch.shuffle} ---')
    # 従来のepoch区分（3,3,2,3,3,2,...）を維持する。
    expected_offsets = ((0, 1, 2), (0, 1, 2), (0, 1), (0, 1, 2))
    expected_starts = (
        (0, 3), (1, 4), (2, 5),
        (6, 9), (7, 10), (8, 11),
        (12, 15), (13, 16),
    )
    assert get_batch.n_batches == [3, 3, 2]
    assert get_batch.active_offsets == [0, 1, 2]
    assert len(get_batch.cycle) == 8

    i = 0
    for epoch, offsets in enumerate(expected_offsets):
        assert get_batch.epoch == epoch
        assert get_batch.iters == 0
        assert get_batch.n_batch == len(offsets)
        for batch, offset in enumerate(offsets):
            assert get_batch.iters == batch
            y = get_batch()
            assert y.shape == (2, 4)
            assert bool(np.all(np.diff(y, axis=1) == 1))
            assert bool(np.all(y[:, 0] % get_batch.step == offset))
            if not get_batch.shuffle:
                assert bool(np.array_equal(y[:, 0], np.array(expected_starts[i % 8])))
                print('epoch', epoch, 'offset', offset, '\n', y)
            i += 1
        assert get_batch.epoch == epoch + 1
        assert get_batch.iters == 0
    return True


def single_batch_test(get_batch, name):
    print(f'\n--- {name} : single full batch ---')
    assert get_batch.n_batches == [1, 0]
    assert get_batch.active_offsets == [0]
    assert len(get_batch.cycle) == 1
    for epoch in range(4):
        assert get_batch.epoch == epoch
        assert get_batch.n_batch == 1
        y = get_batch()
        assert y.shape == (9, 4)
        assert bool(np.all(np.diff(y, axis=1) == 1))
        assert get_batch.epoch == epoch + 1
        assert get_batch.iters == 0
        assert get_batch.cycle_iters == 0


def zero_batch_test(cls, source):
    try:
        cls(source, block_size=4, batch_size=10, step=2, shuffle=False)
    except ValueError as e:
        assert 'No full batch available' in str(e)
    else:
        raise AssertionError('Expected ValueError for no full batches')


data = np.arange(20)
for shuffle in (False, True):
    set_seed(0)
    get_batch = cf.CutOutBatch(data, (4, 4), 2, 3, shuffle)
    smoke_test(get_batch, 'CutOutBatch')

    set_seed(0)
    get_batch = cf.CutOutBatchIx(len(data), (4, 4), 2, 3, shuffle)
    smoke_test(get_batch, 'CutOutBatchIx')

for cls, source in ((cf.CutOutBatch, data), (cf.CutOutBatchIx, len(data))):
    single_batch_test(cls(source, 4, 9, 2, False), cls.__name__)
    zero_batch_test(cls, source)

print('\nCutOutBatch / CutOutBatchIx smoke test OK')
