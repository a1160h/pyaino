from pyaino.Config import *
set_np('numpy'); np=Config.np
from pyaino import nucleus
from pyaino import Functions as F


class L2Normalize:
    """ L2Normalize 手組み """
    def __init__(self, axis=None):
        self.axis = axis
        self.config = None
        self.root_sum_square = F.RootSumSquare(axis)
    
    def forward(self, x):
        l2n = np.sum(x**2, axis=self.axis, keepdims=True)**0.5
        y = x / l2n
        self.x = x
        self.l2n = l2n
        self.y = y
        return y
   
    def backward(self, gy=1):
        x = self.x
        y = self.y
        l2n = self.l2n
        gx0 = gy / l2n
        gl2n = - gy * x / l2n ** 2
        gl2n = np.sum(gl2n, self.axis, keepdims=True)
        gsqsm = gl2n * 0.5 / (l2n + 1e-12)
        gx1 = 2 * x * gsqsm
        gx = gx0 + gx1
        return gx

class L2Normalize2(nucleus.CompositFunction):
    """ L2Normalize CompositFunction """
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis
        
    def _forward(self, x):
        l2n = F.RootSumSquare(self.axis, keepdims=True)(x)
        y = F.Div()(x, l2n)
        return y

class L2Normalize3:
    """ L2Normalize Funcionsの関数をそのまま使う """
    def __init__(self, axis=None):
        self.axis=axis
        self.root_sum_square = F.RootSumSquare(axis, keepdims=True)
        self.div = F.Div()

    def forward(self, x):
        l2n = self.root_sum_square(x)
        y = self.div(x, l2n)
        return y

    def backward(self, gy=1):
        gx0, gl2n = self.div.backward(gy)
        gx1 = self.root_sum_square.backward(gl2n)
        gx = gx0 + gx1
        return gx


functions = (
             L2Normalize,
             L2Normalize2,
             L2Normalize3,
             )
axis = (None, 0, 1, 2)

x = np.random.rand(24).reshape(4, 3, 2)
print(x)


for func in functions:
    for a in axis:
        print('##### test', func.__name__, 'axis =', a, '#####')
        # テスト対照 
        reference = F.L2Normalize(a)
        model = func(a)
        # 順伝播を実行
        y = model.forward(x)
        # 対照による計算
        y_reference = reference.forward(x)
        # 順伝播の結果が正しいか確認
        assert np.allclose(y, y_reference), \
               'Forward pass incorrect \ny\n{}, \ny_reference\n{}'.format(y, y_reference)
        print('Forward pass  test passed.', y.shape, y_reference.shape)
        #print(y)

        # 逆伝播のためにdyを用意
        dy = np.random.rand(*y.shape)
        # 逆伝播を実行
        dx = model.backward(dy)
        # 対照による計算
        dx_reference = reference.backward(dy)
        # 逆伝播の結果が正しいか確認
        assert np.allclose(dx, dx_reference),\
               'Backward pass incorrect \ndx\n{}, \ndx_reference\n{}'.format(dx, dx_reference)
        print('Backward pass test passed.', dx.shape, dx_reference.shape)
        #print(dx)
