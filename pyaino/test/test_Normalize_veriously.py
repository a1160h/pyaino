from pyaino.Config import *
set_np('numpy'); np=Config.np  
from pyaino import Neuron, HDFunctions as HDF, Functions as F
#set_create_graph('True')

class Normalize: # reference
    """ 対照の基本関数の組み合わせによるNormalize """
    def __init__(self, axis=None):
        self.mean = F.Mean(axis=axis, keepdims=True)
        self.std  = F.Std(axis=axis, keepdims=True)
        self.sub  = F.Sub()
        self.div  = F.Div()
    
    def forward(self, x):
        mu  = self.mean(x)
        std = self.std(x)
        z = self.sub(x, mu)
        y = self.div(z, std)
        return y
   
    def backward(self, dy):
        dz, dstd = self.div.backward(dy)
        dx0, dmu = self.sub.backward(dz)
        dx1 = self.mean.backward(dmu)
        dx2 = self.std.backward(dstd)
        dx = dx0 + dx1 + dx2
        return dx

from pyaino.nucleus import CompositFunction
class Normalize1(CompositFunction):
    """ CompositFunctionによるNormalize """
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis
    
    def _forward(self, x):
        mu  = F.Mean(axis=self.axis, keepdims=True)(x)
        std = F.Std(axis=self.axis, keepdims=True)(x)
        z = F.Sub()(x, mu)
        y = F.Div()(z, std)
        return y

class Normalize2:
    """ batch_normalizationから抜き出し """
    def __init__(self, axis=None):
        self.axis=axis
    
    def forward(self, x):
        self.x  = x
        mu =  np.mean(x, axis=self.axis, keepdims=True)
        std = np.std(x, axis=self.axis, keepdims=True)
        self.mu   = mu
        self.std  = std
        y = (x - mu) / (std + 1e-12)
        self.y = y
        return y
    
    def backward(self, gy=1):
        istd = 1/self.std
        iN = self.mu.size / self.x.size
        xc = self.x - self.mu
        gy_sum = np.sum(gy * xc, axis=self.axis, keepdims=True)
        gz = (gy - (self.y * gy_sum * istd * iN)) * istd
        gz_sum = np.sum(gz, axis=self.axis, keepdims=True)
        gx = gz - (gz_sum * iN)
        return gx

class Normalize3:
    """ 新たに手作り """
    def __init__(self, axis=None):
        self.axis=axis

    def forward(self, x):
        self.x  = x
        mu =  np.mean(x, axis=self.axis, keepdims=True)
        std = np.std(x, axis=self.axis, keepdims=True)
        self.std = std
        self.mu  = mu
        y = (x - mu) / (std + 1e-12)
        self.y = y
        return y

    def backward(self, gy):
        n = self.x.size / self.mu.size # muおよびstdを求める際に畳んだ大きさ
        xc = self.x - self.mu
        # div(xc, std) 
        gxc  = gy / self.std
        gstd = -gy * xc / (self.std ** 2)
        gstd = np.sum(gstd, axis=self.axis, keepdims=True)
        # sub(x, mu)
        gx0 = gxc
        gmu = -gxc
        gmu = np.sum(gmu, axis=self.axis, keepdims=True)
        # mean(x)
        gx1 = (1/n) * gmu
        # std(x)
        gvar = gstd * 0.5 / (self.std + 1e-12)
        dvar_dx = (2/n) * xc
        gx2 = gvar * dvar_dx
        # まとめ 
        gx = gx0 + gx1 + gx2
        return gx


 
functions = (Neuron.Normalization,
             F.Normalize,
             F.NormalizeSimple,
             F.Normalize_bkup,
             HDF.Normalize,
             HDF.Normalize_bkup,
             Normalize1,
             Normalize2,
             Normalize3,
             )
axis = (None, 0, 1, 2, (0,1), (0, -1))

x = np.random.rand(24).reshape(4, 3, 2)*100
print(x)

for func in functions:
    for a in axis:
        print('\n##### test', func.__name__, 'axis =', a, '#####')
        model = func(axis=a)
        y = model.forward(x)
        print(type(y))
        reference = Normalize(axis=a) 
        y_reference = reference.forward(x) 

        assert np.allclose(y, y_reference, rtol=1e-5, atol=1e-5), \
               'Forward pass incorrect \ny\n{}, \ny_reference\n{}'.format(y, y_reference)
        print('Forward pass  test passed.', y.shape, y_reference.shape)

        dy = np.random.rand(*y.shape)*10

        dx = model.backward(dy)
        dx_reference = reference.backward(dy)
        print(type(dx))
        assert np.allclose(dx, dx_reference, rtol=1e-3, atol=1e-3),\
               'Backward pass incorrect \ndx\n{}, \ndx_reference\n{}'.format(dx, dx_reference)
        print('Backward pass test passed.', dx.shape, dx_reference.shape)

