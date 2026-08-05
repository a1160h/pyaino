from pyaino.Config import *
set_higher_derivative(True)
from pyaino import HDFunctions as F


indices_axis = np.array([[3, 1, 3], [0, 2, 2]])
indices_flat = np.array([4, 1, 4, 0])

tests = [
    (
        'TakeAlongAxis HDF test: integer axis',
        F.take_along_axis,
        [[1., 2., 3., 4.], [5., 6., 7., 8.]],
        (indices_axis,),
        {'axis': -1},
        [[4., 2., 4.], [5., 7., 7.]],
        [[0., 4., 0., 16.], [10., 0., 28., 0.]],
        [[0., 2., 0., 4.], [2., 0., 4., 0.]],
    ),
    (
        'ScatterAddAlongAxis HDF test: integer axis',
        F.scatter_add_along_axis,
        [[1., 2., 3.], [4., 5., 6.]],
        (indices_axis, (2, 4)),
        {'axis': -1},
        [[0., 2., 0., 4.], [4., 0., 11., 0.]],
        [[8., 4., 8.], [8., 22., 22.]],
        [[4., 2., 4.], [2., 4., 4.]],
    ),
    (
        'TakeAlongAxis HDF test: axis=None',
        F.take_along_axis,
        [[1., 2., 3.], [4., 5., 6.]],
        (indices_flat,),
        {'axis': None},
        [5., 2., 5., 1.],
        [[2., 4., 0.], [0., 20., 0.]],
        [[2., 2., 0.], [0., 4., 0.]],
    ),
    (
        'ScatterAddAlongAxis HDF test: axis=None',
        F.scatter_add_along_axis,
        [1., 2., 3., 4.],
        (indices_flat, (2, 3)),
        {'axis': None},
        [[4., 2., 0.], [0., 4., 0.]],
        [8., 4., 8., 8.],
        [4., 2., 4., 2.],
    ),
]


for name, func, x_data, args, kwargs, y_ref, gx_ref, ggx_ref in tests:
    print(f'\n##### {name} #####')

    x = np.hdarray(x_data)

    y = func(x, *args, **kwargs)
    print('y =\n', y)
    assert np.allclose(y, y_ref)

    F.sum(F.square(y)).backtrace()
    gx = x.grad
    print('gx =\n', gx)
    assert np.allclose(gx, gx_ref)

    F.sum(gx).backtrace()
    ggx = x.grad
    print('ggx =\n', ggx)
    assert np.allclose(ggx, ggx_ref)


print('\nTakeAlongAxis / ScatterAddAlongAxis HDF test passed')
