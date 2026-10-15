from pyaino.Config import *
from pyaino import common_function as cf


def smoke_test(get_batch, name):
    print(f'\n--- {name} : shuffle={get_batch.shuffle} ---')

    # 1 cycle = 1 epoch。全offsetのfull batchをround-robinで使用する。
    expected_offsets = (0, 1, 2, 0, 1, 2, 0, 1)
    expected_starts = (
        (0, 3), (1, 4), (2, 5),
        (6, 9), (7, 10), (8, 11),
        (12, 15), (13, 16),
    )

    assert get_batch.n_batches == [3, 3, 2]
    assert get_batch.n_batch == len(expected_offsets)

    for epoch in range(4):
        assert get_batch.epoch == epoch
        assert get_batch.iters == 0

        for batch, offset in enumerate(expected_offsets):
            assert get_batch.iters == batch
            y = get_batch()
            assert y.shape == (2, 4)
            assert np.all(np.diff(y, axis=1) == 1)
            assert np.all(y[:, 0] % get_batch.step == offset)

            if not get_batch.shuffle:
                assert np.array_equal(y[:, 0], np.array(expected_starts[batch]))
                print('epoch', epoch, 'offset', offset, '\n', y)

        # 最後のbatchを払い出した時点で次のepochへ移る。
        assert get_batch.epoch == epoch + 1
        assert get_batch.iters == 0


def single_batch_test(get_batch, name):
    """片方のoffsetだけがfull batchになる場合の回帰テスト。"""
    print(f'\n--- {name} : single full batch ---')
    assert get_batch.n_batches == [1, 0]
    assert get_batch.n_batch == 1

    for epoch in range(3):
        assert get_batch.epoch == epoch
        y = get_batch()
        assert y.shape == (9, 4)
        assert np.all(np.diff(y, axis=1) == 1)
        assert get_batch.epoch == epoch + 1
        assert get_batch.iters == 0


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

# 以前 IndexError を起こした、n_batches=[1, 0] のケースも確認する。
for cls, source in ((cf.CutOutBatch, data), (cf.CutOutBatchIx, len(data))):
    get_batch = cls(source, block_size=4, batch_size=9, step=2, shuffle=False)
    single_batch_test(get_batch, cls.__name__)

print('\nCutOutBatch / CutOutBatchIx smoke test OK')
