from pyaino.Config import *
from pyaino import common_function as cf


def smoke_test(get_batch, name):
    print(f'\n--- {name} : shuffle={get_batch.shuffle} ---')

    for i in range(4):
        get_batch.info()

        assert get_batch.epoch == i
        assert get_batch.offset == i % get_batch.step
        assert np.all(get_batch.start_ix % get_batch.step == get_batch.offset)

        n_batch = get_batch.n_batch
        for j in range(n_batch):
            y = get_batch()
            assert y.shape == (2, 4)
            assert np.all(np.diff(y, axis=1) == 1)

            if not get_batch.shuffle:
                print(y)


data = np.arange(20)
block_size = (4, 4, 3)
batch_size = 2

for shuffle in (False, True):
    set_seed(0)
    get_batch = cf.CutOutBatch(data, block_size, batch_size, shuffle)
    smoke_test(get_batch, 'CutOutBatch')

    set_seed(0)
    get_batch = cf.CutOutBatchIx(len(data), block_size, batch_size, shuffle)
    smoke_test(get_batch, 'CutOutBatchIx')

print('\nCutOutBatch / CutOutBatchIx smoke test OK')
