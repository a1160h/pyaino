from pyaino.Config import *
from pyaino import common_function as cf

def smoke_test(get_batch, name):
    print(f'\n--- {name} : shuffle={get_batch.shuffle} ---')

    expected_offsets = ((0, 1, 2), (0, 1, 2), (0, 1), (0, 1, 2))

    for epoch, offsets in enumerate(expected_offsets):
        assert get_batch.epoch == epoch
        assert get_batch.n_batch == len(offsets)

        for offset in offsets:
            y = get_batch()
            assert y.shape == (2, 4)
            assert np.all(np.diff(y, axis=1) == 1)
            assert np.all(y[:, 0] % get_batch.step == offset)

            if not get_batch.shuffle:
                print('epoch', epoch, 'offset', offset, '\n', y)


data = np.arange(20)
block_size = (4, 4)
batch_size = 2
step = 3

for shuffle in (False, True):
    set_seed(0)
    get_batch = cf.CutOutBatch(data, block_size, batch_size, step, shuffle)
    smoke_test(get_batch, 'CutOutBatch')

    set_seed(0)
    get_batch = cf.CutOutBatchIx(len(data), block_size, batch_size, step, shuffle)
    smoke_test(get_batch, 'CutOutBatchIx')

print('\nCutOutBatch / CutOutBatchIx smoke test OK')
