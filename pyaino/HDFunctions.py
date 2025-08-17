# HDFunctions 
# 20250817 A.Inoue

from pyaino.Config import *
from pyaino.nucleus import HDArray, HDFunction
import copy


"""
ここで定義される関数の一群は高階微分をサポートする
このためには、逆伝播の際に行う演算も、HD関数で行ってグラフ生成する必要がある

"""

class Zeros(HDFunction):
    """ 加法の単位元 Derivative Identity """
    def __forward__(self, x):
        y = np.zeros_like(x, dtype=Config.dtype)
        return y

    def __backward__(self, gy):
        x, = self.inputs
        gx = zeros(x)         # 0を微分しても0
        return gx
        
def zeros(x):
    return Zeros()(x)

class Ones(HDFunction):
    """ 乗法の単位元 Derivative Identity """
    def __forward__(self, x):
        y = np.ones_like(x, dtype=Config.dtype)
        return y

    def __backward__(self, gy):
        x, = self.inputs
        gx = zeros(x)         # 1を微分したら0
        return gx
        
def ones(x):
    return Ones()(x)

class Assign(HDFunction):
    """ y=x 但し演算子オーバーロードは効かない """
    def __forward__(self, x):
        return x.copy()

    def __backward__(self, gy):
        x, = self.inputs
        gx = gy * ones(x)     # gx=gy*dydx 
        return gx
   
def assign(x):
    return Assign()(x)

class Neg(HDFunction):
    def __forward__(self, x):
        return -x

    def __backward__(self, gy):
        x, = self.inputs
        gx = - gy * ones(x)
        return gx

def neg(x):
    return Neg()(x)

class Pow(HDFunction):
    def __init__(self, c=1):
        super().__init__()
        self.c = HDArray(c)     # 20241019 nucleusでdtypeを指定しないならば元の型を継承
        #if isinstance(c, int):  # 20241030 不要
        #    self.c = self.c.astype(int)
        #self.c.name = 'exponent' # 数値が出ればそれで良い
    
    def __forward__(self, x):
        y = np.power(x, self.c) # 20241019
        return y

    def __backward__(self, gy):
        x, = self.inputs
        c  = self.c
        gx = gy * c * x ** (c - 1)
        return gx

def pow(x, c):
    return Pow(c)(x)

class Square(HDFunction):
    def __forward__(self, x):
        y = np.square(x)        # 20241019
        return y 

    def __backward__(self, gy):
        x, = self.inputs
        gx = gy * 2 * x
        return gx

def square(x):
    return Square()(x)

class SquareRoot(HDFunction):
    def __forward__(self, x):
        y = np.sqrt(x)
        return y 

    def __backward__(self, gy):
        x, = self.inputs
        gx = gy * 0.5 * x ** -0.5
        return gx

class Sqrt(SquareRoot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
def sqrt(x):
    return SquareRoot()(x)

class Exp(HDFunction):
    """ 指数関数(底を指定可能) """
    def __init__(self, a=None):
        super().__init__()
        if a is None:          # 底がネイピア数eの場合
            log_of_base = 1          
        else:                  # 底が指定された場合　
            log_of_base = np.log(a) # 底がeの対数,これを用いて底の交換
        self.log_of_base = HDArray(log_of_base) # 20241019   
        self.log_of_base.name = None if a is None else 'log'+str(a)  
   
    def __forward__(self, x):
        y = np.exp(self.log_of_base * x)
        return y

    def __backward__(self, gy):
        y = self.get_outputs()   
        gx = gy * self.log_of_base * y
        return gx

def exp(x, a=None):
    return Exp(a)(x)

class Log(HDFunction):
    """ 対数関数(底を指定可能) """
    def __init__(self, a=None):
        super().__init__()
        if a is None:          # 自然対数
            log_of_base = 1
        elif a > 0 and a != 1: # 対数の底が指定された場合
            log_of_base = np.log(a)
        else:
            raise Exception("Bad base is given for log")
        self.log_of_base = HDArray(log_of_base) # 20241019   
        self.log_of_base.name = None if a is None else 'log'+str(a)  
       
    def __forward__(self, x):
        y = np.log(x)/self.log_of_base
        return y

    def __backward__(self, gy):
        x, = self.inputs 
        gx = gy * (x * self.log_of_base) ** -1
        return gx

def log(x, a=None): 
    return Log(a)(x) 


class Abs(HDFunction):
    def __forward__(self, x):
        return np.abs(x)

    def __backward__(self, gy):
        x, = self.inputs
        gx = gy * (step(x) * 2 - 1)
        return gx

def abs(x):
    return Abs()(x)

class Sin(HDFunction):
    def __forward__(self, x):
        y = np.sin(x)
        return y

    def __backward__(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(HDFunction):
    def __forward__(self, x):
        y = np.cos(x)
        return y

    def __backward__(self, gy):
        x, = self.inputs
        gx = - gy * sin(x)
        return gx

def cos(x):
    return Cos()(x)

class SinCos(HDFunction):
    def __forward__(self, x):
        """ 複数出力の関数の確認用 """
        y = np.sin(x), np.cos(x)
        return y

    def __backward__(self, *gy):
        x, = self.inputs
        gx = gy[0] * cos(x) - gy[1] * sin(x)
        return gx

def sin_cos(x):
    return SinCos()(x)

class Add(HDFunction):
    def __forward__(self, x0, x1):
        y = x0 + x1
        return y

    def __backward__(self, gy):
        x0, x1 = self.inputs
        gx0 = sum_to(gy, x0.shape) * ones(x0)
        gx1 = sum_to(gy, x1.shape) * ones(x1)
        return gx0, gx1

def add(x0, x1):
    return Add()(x0, x1)


class Sub(HDFunction):
    def __forward__(self, x0, x1):
        y = x0 - x1
        return y

    def __backward__(self, gy):
        x0, x1 = self.inputs
        gx0 = sum_to(gy, x0.shape) * ones(x0)
        gx1 = - sum_to(gy, x1.shape) * ones(x1)
        return gx0, gx1

def sub(x0, x1):
    return Sub()(x0, x1)

def rsub(x0, x1):
    return Sub()(x1, x0)

class Mul(HDFunction):
    def __forward__(self, x0, x1):
        y = x0 * x1
        return y

    def __backward__(self, gy):
        x0, x1 = self.inputs
        gx0 = sum_to(gy * ones(x0) * x1, x0.shape)
        gx1 = sum_to(gy * x0 * ones(x1), x1.shape)        
        return gx0, gx1

def mul(x0, x1):
    return Mul()(x0, x1)

class Div(HDFunction):
    def __init__(self, epsilon=0.0):
        super().__init__()
        self.epsilon = epsilon
        
    def __forward__(self, x0, x1):
        y = x0 / (x1 + self.epsilon)
        return y

    def __backward__(self, gy):
        x0, x1 = self.inputs
        gx0 = sum_to(gy * ones(x0) / x1, x0.shape)
        gx1 = sum_to(- gy * x0 * x1 ** -2, x1.shape)
        return gx0, gx1
   
def div(x0, x1):
    return Div()(x0, x1)

def rdiv(x0, x1):
    return Div()(x1, x0)
  
class SumTo(HDFunction):
    def __init__(self, shape=()):
        super().__init__()
        self.shape = shape
        self.gy_shape = None

    def __forward__(self, x):
        if x.shape == self.shape:
            self.gy_shape = x.shape             # backwardで必要
            return x

        ope_shape = list(self.shape)            # 操作用にコピー
        target_shape = () 
        for i, sx in enumerate(x.shape):        # 次元数を揃えたターゲット形状を得る
            if   len(ope_shape) > i and ope_shape[i] == sx:
                target_shape += sx,
            elif len(ope_shape) > i and ope_shape[i] == 1:
                target_shape += 1,
            else:                               # ope_shape[i] != sxもこのケース
                target_shape += 1,
                ope_shape = [1] + ope_shape     # ope_shapeを1つずらす
        axis = ()
        for i in range(len(x.shape)):           # 同一次元数での比較して軸を得る　
            if target_shape[i] != x.shape[i]:
                axis += i,

        y = np.sum(x, axis=axis, keepdims=True) # 下記の為に次元保持
        self.gy_shape = y.shape                 # backwardで必要(操作したxの形状に)
        y = np.reshape(y, self.shape)
        return y

    def __backward__(self, gy):
        x, = self.inputs 
        gx = gy.reshape(self.gy_shape)          # 先ずは次元数を合わせる
        gx = broadcast_to(gx, x.shape)          # それから所望のbroadcast
        return gx

def sum_to(x, shape):
    return SumTo(shape)(x)

class BroadcastTo(HDFunction):
    def __init__(self, shape=()):
        super().__init__()
        self.shape = shape

    def __forward__(self, x):
        y = np.broadcast_to(x, self.shape)
        return y

    def __backward__(self, gy):
        x, = self.inputs
        gx = sum_to(gy, x.shape)
        return gx

def broadcast_to(x, shape):
    return BroadcastTo(shape)(x)

class SumTo_bkup(HDFunction):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __forward__(self, x):
        ndim = len(self.shape)
        lead = x.ndim - ndim
        lead_axis = tuple(range(lead))
        axis = tuple([i + lead for i, sx in enumerate(self.shape) if sx==1])
        axis = lead_axis + axis
        y = np.sum(x, axis=axis, keepdims=True)
        if lead > 0:
            y = y.squeeze(lead_axis)
        return y

    def __backward__(self, gy):
        x, = self.inputs
        gx = broadcast_to(gy, x.shape)
        return gx

class BroadcastTo_bkup(HDFunction):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __forward__(self, x):
        y = np.broadcast_to(x, self.shape)
        return y

    def __backward__(self, gy):
        x, = self.inputs
        gx = sum_to(gy, x.shape)
        return gx

class Sum_bkup(HDFunction):
    def __init__(self, axis, keepdims):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        
    def __forward__(self, x):
        y = np.sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def __backward__(self, gy):
        x, = self.inputs
        y_shape, = self.y_shapes 
        if self.axis is not None: # 畳まれる軸を'1'とした形状
            gy_shape = x.shape[:self.axis] + (1,) + x.shape[self.axis+1:]
        else: # y.ndimはkeepdimsに従うからdon't care
            gy_shape = y_shape
            
        gy = gy.reshape(gy_shape)
        gx = broadcast_to(gy, x.shape)
        return gx

class SumMeanVar(HDFunction):
    """ 配列操作を担う共通クラス """
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.keepdims = keepdims
        self.axis = axis
        self.n = None
        self.gy_shape = None
    
    def set_axis_and_shape(self, shape):
        """ 畳まれる軸と残る軸を明らかにする """
        ndim = len(shape)
        axis = self.axis # 指定された軸
        all_axis = list(range(ndim))
        if axis is None:
            axis = all_axis
        if type(axis) not in (tuple, list):
            #print('axisをタプルに')
            axis = axis,
            
        # 畳まれる軸と畳まれない軸    
        axis = [ndim + ax if ax<0 else ax for ax in axis] # 負値に対応

        # 畳まれる軸を1、他はそのままの形状
        self.gy_shape = [1 if ax in axis else s for ax, s in zip(all_axis, shape)]

        n = 1
        for ax in axis:
            n *= shape[ax] # 畳まれる軸の形状の積=大きさ
        #print('指定された軸', self.axis)
        #print('畳まれる軸', axis)
        #print('畳まれる軸を1、他はそのままの形状', self.gy_shape)
        #print('畳まれる軸の形状の積=大きさ', n)
        self.n = HDArray(n) # VCG対応20241018
        self.n.name = 'n'   #  

    def __forward__(self, x):
        last_x = self.inputs[0] if self.inputs else None
        if last_x is not None and last_x.shape == x.shape: # 前回と同じ形状
            return
        self.set_axis_and_shape(x.shape)

    def __backward__(self, gy):
        """ 逆伝播:形状のややこしい操作があるので親クラスのbackwardを上書き """
        x, = self.inputs
        gy = gy.reshape(self.gy_shape)          # gyは次元を合わせる
        gy = broadcast_to(gy, x.shape)          # 畳まれた分をbroadcastして元に戻す
        return gy

class Sum(SumMeanVar):
    """ 和 """
    def __forward__(self, x):
        super().__forward__(x)
        y = np.sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class Mean(SumMeanVar):
    """ 平均 """
    def __forward__(self, x):
        super().__forward__(x)
        y = np.mean(x, axis=self.axis, keepdims=self.keepdims)
        return y
    
    def __backward__(self, gy):
        gy = super().__backward__(gy)
        gx = gy * (1/self.n) 
        return gx

def mean(x, axis=None, keepdims=False):
    return Mean(axis, keepdims)(x)

class Var(SumMeanVar):
    """ 分散 """
    def __forward__(self, x):
        super().__forward__(x)
        y = np.var(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def __backward__(self, gy):
        gy = super().__backward__(gy)
        x, = self.inputs
        mu = x.mean(axis=self.axis, keepdims=True)
        gx = gy * (2/self.n) * (x - mu)
        return gx

def var(x, axis=None, keepdims=False):
    return Var(axis, keepdims)(x)

def var_bkup(x, axis=None, keepdims=False):
    mu = mean(x, axis, keepdims=True)
    sq = square(sub(x, mu)) # (x - mu)**2
    return mean(sq, axis=axis, keepdims=keepdims)

class Std(SumMeanVar):
    """ 標準偏差 """
    def __forward__(self, x):
        super().__forward__(x)
        y = np.std(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def __backward__(self, gy):
        gy = super().__backward__(gy)
        x, = self.inputs
        y = self.get_outputs()
        mu = x.mean(axis=self.axis, keepdims=True)
        yr = y.reshape(self.gy_shape)
        gvar = gy * 0.5 / (yr + 1e-12) # std = sqrt(var) の逆伝播　
        dvar_dx = (2/self.n) * (x - mu)
        gx = gvar * dvar_dx
        return gx

def std(x, axis=None, keepdims=False):
    return Std(axis, keepdims)(x)

class SquareSum(SumMeanVar):
    """ 二乗和 """
    def __forward__(self, x):
        super().__forward__(x)
        y = np.sum(x**2, axis=self.axis, keepdims=self.keepdims)
        return y

    def __backward__(self, gy):
        gy = super().__backward__(gy)
        x, = self.inputs
        gx = gy * 2 * x
        return gx
        
class SquareMean(SumMeanVar):
    """ 二乗平均 """
    def __forward__(self, x):
        super().__forward__(x)
        y = np.mean(x**2, axis=self.axis, keepdims=self.keepdims)
        return y

    def __backward__(self, gy):
        gy = super().__backward__(gy)
        x, = self.inputs
        gx = gy * (1/self.n) * 2 * x
        return gx
    
class RootSumSquare(SumMeanVar):
    """ 二乗和平方根 """
    def __forward__(self, x):
        super().__forward__(x)
        sqsm = np.sum(x**2, axis=self.axis, keepdims=self.keepdims)
        y = np.sqrt(sqsm)
        return y

    def __backward__(self, gy):
        gy = super().__backward__(gy)
        x, = self.inputs
        y = self.get_outputs()
        yr = y.reshape(self.gy_shape) # gyと形状を揃える
        gsqsm = gy * 0.5 / (yr + 1e-12)    # RMS = sqrt(sqmu)の逆伝播
        gx = 2 * x * gsqsm
        return gx

class RootMeanSquare(SumMeanVar):
    """ 二乗平均平方根(RootMeanSquare) """
    def __forward__(self, x):
        super().__forward__(x)
        sqmu = np.mean(x**2, axis=self.axis, keepdims=self.keepdims)
        y = np.sqrt(sqmu)
        return y

    def __backward__(self, gy):
        gy = super().__backward__(gy)
        x, = self.inputs
        y = self.get_outputs()
        yr = y.reshape(self.gy_shape) # gyと形状を揃える
        gsqmu = gy * 0.5 / (yr + 1e-12)    # RMS = sqrt(sqmu)の逆伝播
        gx = (1/self.n) * 2 * x * gsqmu
        return gx

class RMS(RootMeanSquare):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class MaxMin(HDFunction):
    """ MaxとMinの共通ベース """
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def condition(self, x, y, axis):
        """ 抽出されたところを示す配列を作る、併せて逆伝播に必要なものを揃える """
        self.x_shape = x.shape
        self.y_shape = y.shape
        if axis is None:
            axis = range(x.ndim)
        elif isinstance(axis, int):
            axis = axis,                      
        self.z_shape = [1 if i in axis else s for i, s in enumerate(x.shape)] #xと同次元数のyの形
        self.cond = x == np.reshape(y, self.z_shape) # xとyの比較

    def __forward__(self, *args, **kwargs):
        raise NotImplementedError()

    def __backward__(self, gy):
        """ 逆伝播は共通 """
        #gy = gy if isinstance(gy, np.ndarray) else np.array(gy, dtype=Config.dtype) 
        gy = broadcast_to(gy, self.y_shape)   # 先ずはyの形状に合わせる
        gy = gy.reshape(self.z_shape)            # 次にxに次元を揃える(keepdimsの形状)
        gx = gy * self.cond                      # yに抽出されたところにgyを入れる
        return gx

class Max(MaxMin):
    """ 最大値を抽出 """
    def __forward__(self, x):
        y = np.max(x, axis=self.axis, keepdims=self.keepdims)
        self.condition(x, y, self.axis)
        return y

class Min(MaxMin):
    """ 最小値を抽出 """
    def __forward__(self, x):
        y = np.min(x, axis=self.axis, keepdims=self.keepdims)
        self.condition(x, y, self.axis)
        return y

def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)

def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)

class GetItem(HDFunction):
    def __init__(self, slices):
        super().__init__()
        self.slices = slices

    def __forward__(self, x):
        y = x[self.slices]
        return y

    def __backward__(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)

class GetItemGrad(HDFunction):
    def __init__(self, slices, in_shape):
        super().__init__()
        self.slices = slices
        self.in_shape = in_shape
        # 関数の設定 
        try:
            self.func = np.add.at
        except:
            try:
                self.func = np.scatter_add
            except:
                try:
                    self.func = np._cupyx.scatter_add
                except:
                    def f(x, y, z): # xのyの位置にzを加算する
                        for i, idx in enumerate(y):
                            x[idx] += z[i]
                    self.func = f        

    def __forward__(self, gy):
        gx = np.zeros(self.in_shape, dtype=gy.dtype)
        # gyをgxのslices位置に埋める
        self.func(gx, self.slices, gy)

        return gx

    def __backward__(self, ggx):
        return getitem(ggx, self.slices)

def getitem(x, slices):
    f = GetItem(slices)
    return f(x)

class Reshape(HDFunction):
    def __init__(self, *shape):
        super().__init__()
        if len(shape) > 1:
            self.shape = shape
        else: # もともとタプル
            self.shape, = shape 
        
    def __forward__(self, x):
        y = np.reshape(x, self.shape)
        return y

    def __backward__(self, gy):
        x, = self.inputs
        gx = reshape(gy, x.shape)
        return gx

def reshape(x, *shape):
    return Reshape(*shape)(x)

class Transpose(HDFunction):
    def __init__(self, *axes):
        super().__init__()
        if axes is None:
            self.axes  = (1, 0)
        elif len(axes) == 0:
            self.axes  = (1, 0)
        elif len(axes) == 1:
            self.axes, = axes
        else:
            self.axes  = axes
        #if self.axes is None:
        #    self.raxes = None
        #    return
        self.raxes = np.argsort(np.array(self.axes)).tolist() # cupy対応
        
    def __forward__(self, x):
        y = np.transpose(x, self.axes)
        return y

    def __backward__(self, gy):
        gx = transpose(gy, self.raxes)                
        return gx

def transpose(x, *axes):
    return Transpose(*axes)(x)

class Transpose_s(HDFunction):
    """ nucleusと自身でVariable.Tに使う """
    def __forward__(self, x):
        y = np.transpose(x)
        return y

    def __backward__(self, gy):
        gx = transpose_s(gy)
        return gx

def transpose_s(x):
    return Transpose_s()(x)

class Dot(HDFunction):
    def __forward__(self, x, w):
        """
        以下の場合があり、backwardで配慮が必要
        単なるスカラ積(x.ndim==w.ndim==0)->スカラ
        ベクトルのスカラ積(x.ndim==1, w.ndim==0 ないし x.ndim==0, w.ndim==1)->ベクトル
        行列のスカラ積(x.ndim>1, w.ndim==0 ないし x.ndim==0, w.ndim>1)->行列
        ベクトルの内積(x.ndim==w.ndim==1)->スカラ
        行列積(x.ndim>1, w.ndim>1)->行列
        
        """
        y = np.dot(x, w)
        return y

    def __backward__(self, gy):
        x, w = self.inputs
        x_shape = x.shape
        w_shape = w.shape
        if x.ndim>1 and w.ndim>1:
            gx = dot(gy, w.transpose())
            gw = dot(x.transpose(), gy)
            return gx, gw
        if x.ndim>1 and w.ndim==1:
            w  = w.reshape(-1, 1)
            gy = gy.reshape(-1, 1)
            gx = dot(gy, w.transpose())
            gw = dot(x.transpose(), gy)
            gw = gw.reshape(*w_shape)
            return gx, gw
        if x.ndim>1 and w.ndim==0:
            gx = gy * w
            gw = dot(x.reshape(-1), gy.reshape(-1))
            return gx, gw
        if x.ndim==1 and w.ndim>1:
            x  = x.reshape(1, -1)
            gy = gy.reshape(1, -1)
            gx = dot(gy, w.transpose())
            gw = dot(x.transpose(), gy)
            gx = gx.reshape(*x_shape)
            return gx, gw
        if x.ndim==1 and w.ndim==1:
            gx = dot(gy, w)
            gw = dot(x, gy)
            return gx, gw
        if x.ndim==1 and w.ndim==0:
            gx = gy * w
            gw = dot(x, gy)
            return gx, gw
        if x.ndim==0 and w.ndim>1:
            gx = dot(gy.reshape(-1), w.reshape(-1))
            gw = x * gy
            return gx, gw
        if x.ndim==0 and w.ndim==1:
            gx = dot(gy, w)
            gw = x * gy
            return gx, gw
        if x.ndim==0 and w.ndim==0:
            gx = gy * w
            gw = x * gy
            return gx, gw
        raise Exception('Cant handle the case.')

def dot(x, w):
    return Dot()(x, w)

class MatMul(HDFunction):
    def __forward__(self, x0, x1):
        y = np.matmul(x0, x1)
        return y

    def __backward__(self, gy):
        x0, x1 = self.inputs
        #x0T = x0.T if x0.ndim <= 2 else x0.transpose(*range(x0.ndim)[:-2], -1, -2)
        #x1T = x1.T if x1.ndim <= 2 else x1.transpose(*range(x1.ndim)[:-2], -1, -2)
        x0T = x0.T if x0.ndim <= 2 else transpose(x0, (*range(x0.ndim)[:-2], -1, -2))
        x1T = x1.T if x1.ndim <= 2 else transpose(x1, (*range(x1.ndim)[:-2], -1, -2))
        gx0 = matmul(gy, x1T)
        gx1 = matmul(x0T, gy)
        return gx0, gx1
     
def matmul(x0, x1):
    return MatMul()(x0, x1)

class DotLinear(HDFunction):
    def __forward__(self, x, w, b):
        """
        以下の場合があり、backwardで配慮が必要
        単なるスカラ積(x.ndim==w.ndim==0)->スカラ
        ベクトルのスカラ積(x.ndim==1, w.ndim==0 ないし x.ndim==0, w.ndim==1)->ベクトル
        行列のスカラ積(x.ndim>1, w.ndim==0 ないし x.ndim==0, w.ndim>1)->行列
        ベクトルの内積(x.ndim==w.ndim==1)->スカラ
        行列積(x.ndim>1, w.ndim>1)->行列
        
        """
        y = np.dot(x, w)
        self.dot_dim = y.ndim
        y += b
        return y

    def __backward__(self, gy):
        x, w, b = self.inputs
        y = self.get_outputs()
        x_shape = x.shape
        w_shape = w.shape
        gb = gy if y.shape==b.shape else SumTo(b.shape)(gy)

        if x.ndim>1 and w.ndim>1:
            gx = dot(gy, w.transpose())
            gw = dot(x.transpose(), gy)
            return gx, gw, gb
        if x.ndim>1 and w.ndim==1:
            w  = w.reshape(-1, 1)
            gy = gy.reshape(-1, 1)
            gx = dot(gy, w.transpose())
            gw = dot(x.transpose(), gy)
            gw = gw.reshape(*w_shape)
            return gx, gw, gb
        if x.ndim>1 and w.ndim==0:
            gx = gy * w
            gw = dot(x.reshape(-1), gy.reshape(-1))
            return gx, gw, gb
        if x.ndim==1 and w.ndim>1:
            x  = x.reshape(1, -1)
            gy = gy.reshape(1, -1)
            gx = dot(gy, w.transpose())
            gw = dot(x.transpose(), gy)
            gx = gx.reshape(*x_shape)
            return gx, gw, gb
        if x.ndim==1 and w.ndim==1:
            if self.dot_dim < y.ndim and len(y)==1:
                # forwardの際に+bでスカラがarrayになった場合
                gy = gy[0]
            gx = dot(gy, w)
            gw = dot(x, gy)
            return gx, gw, gb
        if x.ndim==1 and w.ndim==0:
            gx = gy * w
            gw = dot(x, gy)
            return gx, gw, gb
        if x.ndim==0 and w.ndim>1:
            gx = dot(gy.reshape(-1), w.reshape(-1))
            gw = x * gy
            return gx, gw, gb
        if x.ndim==0 and w.ndim==1:
            gx = dot(gy, w)
            gw = x * gy
            return gx, gw, gb
        if x.ndim==0 and w.ndim==0:
            gx = gy * w
            gw = x * gy
            return gx, gw, gb
        raise Exception('Cant handle the case.')
        

class DotLinearz(HDFunction):
    def __forward__(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    def __backward__(self, gy):
        x, w, b = self.inputs
        y_shape, = self.y_shapes ()
        x_shape = x.shape
        w_shape = w.shape
        if w.ndim == 1 and x.ndim >= 1:
            w  = w.reshape(-1, 1)
            gy = gy.reshape(-1, 1)
        if x.ndim == 1 and w.ndim >= 1:
            x  = x.reshape(1, -1)
            gy = gy.reshape(1, -1)
        #print('x', x.shape, x_shape, 'w', w.shape, w_shape, 'b', b.shape, 'gy', gy.shape)    
        gx = dot(gy, w.transpose())
        gw = dot(x.transpose(), gy)
        if x.ndim >= 1:
            gx = gx.reshape(*x_shape)
        if w.ndim >= 1:
            gw = gw.reshape(*w_shape)
        elif w.ndim == 0 and gw.ndim > 1: # 仮処置20240417
            gw = gw[0, 0]
        gb = gy if y_shape==b.shape else SumTo(b.shape)(gy)
        return gx, gw, gb

def dot_linear(x, w, b):
    return DotLinear()(x, w, b)

class HadamardLinear(HDFunction):
    def __forward__(self, x, w, b):
        y = x * w + b
        return y

    def __backward__(self, gy):
        x, w, b = self.inputs
        y_shape, = self.y_shapes ()
        gx = gy * w
        gw = gy * x
        gb = gy if y_shape==b.shape else SumTo(b.shape)(gy)
        return gx, gw, gb

def hadamard_linear(x, w, b):
    return HadamardLinear()(x, w, b)
     
class Flatten(HDFunction):
    """ 軸0はバッチとし、それ以下の軸を平坦化 """
    def __forward__(self, x):
        return np.reshape(x, (x.shape[0], -1))

    def __backward__(self, gy):
        x, = self.inputs
        return gy.reshape(x.shape)


def normalize(x, axis=None, eps=1e-12):
    mu = mean(x, axis=axis, keepdims=True)
    sigma = std(x, axis=axis, keepdims=True)
    z = x - mu
    y = z / (sigma + eps)
    mask = sigma < eps
    z.name ='x - mu'
    mu.name = 'mu'
    sigma.name = 'sigma'
    mask.name = 'mask'
    return y * (1 - mask) + x * mask

def normalize_bkup(x, axis=None):
    mu  = mean(x, axis, keepdims=True)
    varx = var(x, axis, keepdims=True)
    std = sqrt(varx)
    x_minus_mu = sub(x, mu)
    div = Div(1e-7)
    y = div(x_minus_mu, std)
    return y 

class Normalize(HDFunction):
    """ 平均0標準偏差1にする標準化(正規化の一種) """
    def __init__(self, axis=None):
        super().__init__()
        self.mean = Mean(axis, keepdims=True)
        self.std  = Std(axis, keepdims=True)
        self.sub  = Sub()
        self.div  = Div()
    
    def __forward__(self, x):
        mu  = self.mean(x)
        std = self.std(x)
        z = self.sub(x, mu)
        y = self.div(z, std)
        return y
   
    def __backward__(self, gy):
        gz, dstd = self.div.backward(gy)
        gx0, dmu = self.sub.backward(gz)
        gx1 = self.mean.backward(dmu)
        gx2 = self.std.backward(dstd)
        gx = gx0 + gx1 + gx2
        return gx

class Standardize(Normalize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class L2Normalize(HDFunction):
    """ L2ノーマライゼーション """
    def __init__(self, axis=None):
        super().__init__()
        self.root_sum_square = RootSumSquare(axis=axis, keepdims=True)
        self.div = Div()
    
    def __forward__(self, x):
        l2n = self.root_sum_square.forward(x)
        y = self.div.forward(x, l2n)
        return y
   
    def __backward__(self, gy):
        gx0, gl2n = self.div.backward(gy)
        gx1 = self.root_sum_square.backward(gl2n)
        gx = gx0 + gx1
        return gx

class Normalize_bkup(HDFunction):
    """ 平均0標準偏差1にする標準化(正規化の一種) """
    def __init__(self, axis=None):
        self.axis = axis
        
    def forward(self, x):
        self.x = x
        mu =  np.mean(x, axis=self.axis, keepdims=True)
        std = np.std(x, axis=self.axis, keepdims=True)
        self.mu   = mu
        self.std  = std
        y = (x - mu) / (std + 1e-12)
        self.y = y
        return y
    
    def backward(self, gy=1):
        istd = 1/self.std
        iN = self.mu.size / self.x.size # muおよびstdを求める際に畳んだ大きさ
        xc = self.x - self.mu
        gy_sum = np.sum(gy * xc, axis=self.axis, keepdims=True)
        gz = (gy - (self.y * gy_sum * istd * iN)) * istd
        gz_sum = np.sum(gz, axis=self.axis, keepdims=True)
        gx = gz - (gz_sum * iN)
        return gx

##########################################################################
# 以下、数学的に不正確で制約があるが、デバグ用に簡略化した定義 : 20230303 A.I.
##########################################################################

class AddSimple(HDFunction):
    """ デバグ用 """
    def __forward__(self, x0, x1):
        y = x0 + x1
        return y

    def __backward__(self, gy):
        self.assign0 = Assign()
        self.assign1 = Assign()
        gx0 = self.assign0(gy)
        gx1 = self.assign1(gy)
        return gx0, gx1

class MulSimple(HDFunction):
    """ デバグ用 """
    def __forward__(self, x0, x1):
        y = x0 * x1
        return y

    def __backward__(self, gy):
        x0, x1 = self.inputs
        gx0 = muls(gy, x1)
        gx1 = muls(gy, x0)
        return gx0, gx1

def muls(x0, x1):
    return MulSimple()(x0, x1)


class PowSimple(HDFunction):
    """ デバグ用 """
    def __init__(self, c):
        super().__init__()
        self.c = c
    
    def __forward__(self, x):
        y = x ** self.c
        return y

    def __backward__(self, gy):
        x, = self.inputs
        c  = self.c
        gx = muls(gy, muls(c, pows(x, c - 1))) # gy * c * x ** (c - 1)
        return gx

def pows(x, c):
    return PowSimple(c)(x)

class SquareSimple(HDFunction):
    """ デバグ用 """
    def __forward__(self, x):
        y = x ** 2
        return y 

    def __backward__(self, gy):
        x, = self.inputs
        gx = muls(gy, muls(2, x)) # gy * 2 * x
        return gx

def squares(x):
    return SquareSimple()(x)


class DivSimple(HDFunction):
    """ デバグ用 """
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon
        
    def __forward__(self, x0, x1):
        y = x0 / (x1 + self.epsilon)
        return y

    def __backward__(self, gy):
        x0, x1 = self.inputs
        gx0 = divs(gy, x1)
        gx1 = negs(divs(muls(gy, x0), squares(x1)))
        return gx0, gx1

def divs(x0, x1):
    return DivSimple()(x0, x1)

class NegSimple(HDFunction):
    """ デバグ用 """
    def __forward__(self, x):
        return -x

    def __backward__(self, gy):
        return negs(gy)

def negs(x):
    return NegSimple()(x)



############################################ 
# 以下、__forward__のみの定義 
############################################

class Step(HDFunction):
    def __init__(self, c=0):
        super().__init__()
        self.c = c
        
    def __forward__(self, x):
        return np.where(x <= self.c, 0, 1)

    def __backward__(self, gy):
        raise NotImplementedError('unable to get differential of '+self.__class__.__name__)

def step(x, c=0):
    return Step(c)(x)

class Equal(HDFunction):
    def __forward__(self, x0, x1):
        return x0 == x1

def equal(x0, x1):
    return Equal()(x0, x1)

class GreaterThan(HDFunction):
    def __forward__(self, x0, x1):
        return x0 > x1

def greater_than(x0, x1):
    return GreaterThan()(x0, x1)

class GreaterThanOrEqual(HDFunction):
    def __forward__(self, x0, x1):
        return x0 >= x1

def greater_than_or_equal(x0, x1):
    return GreaterThanOrEqual()(x0, x1)

class LessThan(HDFunction):
    def __forward__(self, x0, x1):
        return x0 < x1

def less_than(x0, x1):
    return LessThan()(x0, x1)

class LessThanOrEqual(HDFunction):
    def __forward__(self, x0, x1):
        return x0 <= x1

def less_than_or_equal(x0, x1):
    return LessThanOrEqual()(x0, x1)

class Argmax(HDFunction):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def __forward__(self, x):
        return np.argmax(x, axis=self.axis, keepdims=self.keepdims)

class Argmin(Argmax):
    def __forward__(self, x):
        return np.argmin(x, axis=self.axis, keepdims=self.keepdims)

class Argsort(HDFunction):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def __forward__(self, x):
        return np.argsort(x, axis=self.axis)

def argmax(x, axis=None, keepdims=False):
    return Argmax(axis, keepdims)(x)

def argmin(x, axis=None, keepdims=False):
    return Argmin(axis, keepdims)(x)
    
def argsort(x, axis=None):
    return Argsort(axis)(x)

#######################################################
# OperatorOverload
#######################################################
class OverloadContents:
    def save(self):
        HDArray.original_neg  = HDArray.__neg__
        HDArray.original_add  = HDArray.__add__
        HDArray.original_radd = HDArray.__radd__
        HDArray.original_sub  = HDArray.__sub__
        HDArray.original_rsub = HDArray.__rsub__
        HDArray.original_mul  = HDArray.__mul__
        HDArray.original_rmul = HDArray.__rmul__
        HDArray.original_div  = HDArray.__truediv__
        HDArray.original_rdiv = HDArray.__rtruediv__
        HDArray.original_pow  = HDArray.__pow__
        HDArray.original_rpow = HDArray.__rpow__
        HDArray.original_getitem = HDArray.__getitem__
        HDArray.original_equal = HDArray.__eq__
        HDArray.original_gt    = HDArray.__gt__
        HDArray.original_ge    = HDArray.__gt__
        HDArray.original_lt    = HDArray.__lt__
        HDArray.original_le    = HDArray.__le__

        HDArray.original_reshape = HDArray.reshape
        HDArray.original_transpose = HDArray.transpose
        HDArray.original_mean = HDArray.mean

    def overload(self):
        HDArray.__neg__  = neg 
        HDArray.__add__  = add 
        HDArray.__radd__ = add 
        HDArray.__sub__  = sub 
        HDArray.__rsub__ = rsub 
        HDArray.__mul__  = mul
        HDArray.__rmul__ = mul
        HDArray.__truediv__  = div 
        HDArray.__rtruediv__ = rdiv 
        HDArray.__pow__  = pow 
        HDArray.__rpow__ = exp
        HDArray.__getitem__ = getitem 
        HDArray.__eq__ = equal
        HDArray.__gt__ = greater_than
        HDArray.__ge__ = greater_than_or_equal
        HDArray.__lt__ = less_than
        HDArray.__le__ = less_than_or_equal

        HDArray.reshape = reshape
        HDArray.transpose = transpose
        HDArray.mean = mean

    def recover(self):
        HDArray.__neg__  = HDArray.original_neg 
        HDArray.__add__  = HDArray.original_add 
        HDArray.__radd__ = HDArray.original_add
        HDArray.__sub__  = HDArray.original_sub
        HDArray.__rsub__ = HDArray.original_rsub
        HDArray.__mul__  = HDArray.original_mul
        HDArray.__rmul__ = HDArray.original_mul 
        HDArray.__truediv__  = HDArray.original_div
        HDArray.__rtruediv__ = HDArray.original_rdiv
        HDArray.__pow__  = HDArray.original_pow
        HDArray.__rpow__ = HDArray.original_rpow
        HDArray.__getitem__ = HDArray.original_getitem
        HDArray.__eq__  = HDArray.original_equal
        HDArray.__gt__  = HDArray.original_gt
        HDArray.__ge__  = HDArray.original_ge
        HDArray.__lt__  = HDArray.original_lt
        HDArray.__le__  = HDArray.original_le

        HDArray.reshape = HDArray.original_reshape
        HDArray.transpose = HDArray.original_transpose
        HDArray.mean = HDArray.original_mean

### test sum, mean ###
if __name__=='__main__':
    print('\n#### all cast ####')
    import inspect
    import sys
    current_module = sys.modules[__name__]
    classes = map(lambda x:x[0],inspect.getmembers(current_module,inspect.isclass))
    classes = list(classes)
    print(classes)
    
    #
    from pyaino.nucleus import CompositFunction 
    import matplotlib.pyplot as plt
    print('基本関数のテスト')
    #set_create_graph('True')

    set_higher_derivative(True)    

    #"""#
    print('オペランドが１つで１回微分だけの関数')
    functions = Abs, 
    x = HDArray(np.linspace(-4, 4))

    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func(x)
        #gx = func.backward()
        y.backtrace(create_graph=True)
        gx = x.grad
        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        plt.show()

    print('オペランドが１つで高階微分可能な関数')
    functions = Zeros, Ones, Assign, Sin, Cos, Square, Sqrt, Exp, Pow, Log
    x = HDArray(np.linspace(-4, 4))

    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func(x)
        #gx = func.backward()
        y.backtrace(create_graph=True)
        gx = x.grad
        gx.backtrace()
        g2x = x.grad
        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        plt.plot(x.tolist(), g2x.tolist())
        plt.show()

    print('オペランドが１つの関数 拡張')
    functions = Exp, Pow, Log
    
    x = HDArray(np.linspace(-4, 4))
    a = 3

    for f in functions:
        func = f(a)
        print('test ', func.__class__.__name__)
        y = func(x)
        y.backtrace(create_graph=True)
        gx = x.grad
        gx.backtrace()
        g2x = x.grad
        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        plt.plot(x.tolist(), g2x.tolist())
        plt.show()

    print('オペランドが2つの関数')
    functions = Add, Sub, Mul, Div
    x0 = HDArray(np.linspace(-4, 4)) 
    x1 = HDArray(np.linspace(4, -4))
    
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func(x0, x1)

        y.backtrace(create_graph=True)
        gx0, gx1 = x0.grad, x1.grad
        gx0 = x0.grad
        gx1 = x1.grad
        gx0.backtrace(create_graph=True)
        gx0x0 = x0.grad
        gx0x1 = x1.grad
        
        gx1.backtrace(create_graph=True)
        gx1x0 = x0.grad
        gx1x1 = x1.grad
        
        plt.plot(x0.tolist(), label='x0')
        plt.plot(x1.tolist(), label='x1')
        plt.plot(y.tolist(), label='y')
        plt.plot(gx0.tolist(), label='gx0')
        plt.plot(gx1.tolist(), label='gx1')
        plt.plot(gx0x0.tolist(), label='gx0x0')
        plt.plot(gx0x1.tolist(), label='gx0x1')
        plt.plot(gx1x0.tolist(), label='gx1x0')
        plt.plot(gx1x1.tolist(), label='gx1x1')
        plt.legend()
        plt.show()

    set_higher_derivative(False)    
        
       
    #"""#
    print('そのほかの関数のテスト')
    functions = Transpose, Transpose_s, Flatten, Max, Min
    x = np.arange(12).reshape(3,4)
    print(x)
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func(x)
        print(y)
        gx = func.backward()
        print(gx)
    #"""#
        
    print('そのほかの関数のテスト2')
    #set_higher_derivative(True)
    set_create_graph(True)
    a = HDArray(np.arange(12).reshape(3,4))
    print('a =', a)

    print('-- test reshape --')
    #b = a.reshape(4, 3)
    b = reshape(a, (4, 3)) 
    print('b =', b)
    b.backtrace()
    print('gb =', b.grad)
    print('ga =', a.grad)

    print('-- test transpose --')
    c = transpose(a, (1, 0))
    print('c = a.T =', c)
    c.backtrace()
    print('gc =', c.grad)
    print('ga =', a.grad)
    
    print('-- test var --')
    b = var(a, axis=1)
    print('b =', b)
    b.backtrace()
    print('gb =', b.grad)
    print('ga =', a.grad)

    print('-- test normalize --')
    c = normalize(a, axis=0)
    print('c =', c)
    c.backtrace()
    print('gb =', c.grad)
    print('ga =', a.grad)

    #'''#
    a = HDArray(np.arange(12).reshape(3,4))
    print('a =', a)

    cases = (sum, mean, max, min)
    for case in cases:
        print('-- test', case.__name__, '--')
        b = case(a, axis=1)
        print('b =', b)
        b.backtrace()
        print('gb =', b.grad)
        print('ga =', a.grad)


    #"""#
    print('基本関数の組み合わせのテスト')
    set_create_graph(False)
    functions = (Normalize, L2Normalize)
    x = np.random.rand(10)

    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)

        y = func.forward(x)
        print(x.shape, y.shape)
        gy = np.arange(0, y.size) #np.random.rand(y.data.size)
        gy = gy[::-1]
        gx = func.backward(gy)

        print(type(x), type(y), type(gy), type(gx))

        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        plt.plot(x.tolist(), gy.tolist())
        plt.grid()
        plt.show()

    print('基本関数の組み合わせのテスト2 backtrace')
    functions = (Normalize, L2Normalize)
    set_create_graph(True)
    set_derivative(True)

    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)

        y = func.forward(x)
        print(x.shape, y.shape)
        gy = np.arange(0, y.size) #np.random.rand(y.data.size)
        gy = gy[::-1]
        #func.outputs[0].grad = gy # 対象の出力に勾配を設定
        y.backtrace(gy)
        gx = func.inputs[0].grad

        print(type(x), type(y), type(gx))

        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        plt.plot(x.tolist(), gy.tolist())
        plt.grid()
        plt.show()

    set_derivative(False)

    print('基本関数でnormalize')
    set_derivative(True)
    print('test normalize')

    x = HDArray(np.random.rand(10))

    y = normalize(x)
    print(x.shape, y.shape)
    gy = np.arange(0, y.size).astype(Config.dtype) # 20250130AI
    gy = gy[::-1]
    y.backtrace(gy)
    gx = x.grad

    print(type(x), type(y), type(gx))

    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.plot(x.tolist(), gy.tolist())
    plt.grid()
    plt.show()

    set_derivative(False)

    #"""#



    #'''#
    #'''#
    print('合成関数の検証(例としてsigmoid関数を取り上げる)')
    class SigmoidComposit(CompositFunction):
        def _forward(self, x):
            y = div(1, add(1, exp(neg(x))))
            return y

    x = np.linspace(-5, 5, 10)

    sigmoid = SigmoidComposit()
    y = sigmoid.forward(x)
    gx = sigmoid.backward()

    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.show()

    #'''#
    print('合成関数の検証(例としてsigmoid関数を演算子オーバーロードで)')
    #'''#
    #'''#
    class SigmoidComposit(CompositFunction):
        def _forward(self, x):
            y = 1 / (1 + np.e**(-x))            
            return y

    x = np.linspace(-5, 5, 10)

    sigmoid = SigmoidComposit()
    y = sigmoid.forward(x)
    gx = sigmoid.backward(); print(type(gx))

    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.show()

    #'''#
    #'''#
    print('合成関数の検証(例としてtan関数を取り上げる)')
    class TanComposit(CompositFunction):
        def _forward(self, x):
            sinx, cosx = sin_cos(x)
            return div(sinx, cosx) 

    x = np.linspace(-1, 1, 10)

    tan = TanComposit()
    y = tan.forward(x)
    gx = tan.backward()
             
    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.show()

    #'''#
    #'''#
    print('合成関数の検証(例としてnormalize関数を取り上げる)')
    class NormalizeComposit(CompositFunction):
        def _forward(self, x, axis=None, keepdims=False):
            mu  = mean(x, axis, keepdims)
            std = sqrt(var(x, axis, keepdims))
            std = add(std, 1e-12)
            y   = div(sub(x, mu), std)
            return y

    x = np.random.rand(10)

    normalize = NormalizeComposit()
    y = normalize.forward(x)
    gy = np.arange(0, y.size) #np.random.rand(y.data.size)
    gy = gy[::-1]
    gx = normalize.backward(gy)

    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.plot(x.tolist(), gy.tolist())
    plt.grid()
    plt.show()

    #'''#

    print('追加の検証～演算子オーバーロード')

    set_higher_derivative(True)
    Config.enable_debug_print=True

    x = HDArray(np.linspace(-2, 2, 5))
    print(type(x))

    print('うまく行く')
    y = 3*x**2 + x
    gy = np.ones_like(y) # gyを明示的に与える
    y.backtrace(gy)#, create_graph=True)
   
    plt.plot(x.tolist(), y.tolist(), label="y=f(x)")
    plt.plot(x.tolist(), x.grad.tolist(), label="y'")
    plt.legend()#loc='lower right')
    plt.show()

    print('これもうまく行く(Funcionsではうまく行かない)')
    y = 2*x**2 + x
    y.backtrace()#create_graph=True) # gyを指定しない
    
    plt.plot(x.tolist(), y.tolist(), label="y=f(x)")
    plt.plot(x.tolist(), x.grad.tolist(), label="y'")
    plt.legend()#loc='lower right')
    plt.show()

    print('テンソル操作の関数のテスト')
    x = np.arange(24).reshape(2,3,4)
    a = (4,2,3)
    func = Reshape(a)
    print('test ', func.__class__.__name__, x.shape, '->', a)
    y = func(x)  
    gx = func.backward()
    
    print(x)
    print(y)
    print(gx)
    
    func = Reshape(4,2,3)
    print('test ', func.__class__.__name__, x.shape, '->', a)
    y = func(x)  
    gx = func.backward()
    
    print(x)
    print(y)
    print(gx)

    a = (2,0,1)
    func = Transpose(a)
    print('test ', func.__class__.__name__, x.shape, ':', a)
    y = func(x)  
    gx = func.backward()
    
    print(x)
    print(y)
    print(gx)

    func = Transpose(2,0,1)
    print('test ', func.__class__.__name__, x.shape, ':', a)
    y = func(x)  
    gx = func.backward()
    
    print(x)
    print(y)
    print(gx)

    x = x.reshape(6,4) 
    func = Transpose()
    print('test ', func.__class__.__name__, x.shape, ':', a)
    y = func(x)  
    gx = func.backward()
    
    print(x)
    print(y)
    print(gx)

    x0 = np.arange(12).reshape(3, 4)
    x1 = np.arange(12).reshape(4, 3)
    func = Dot()
    print('test ', func.__class__.__name__, x0.shape, x1.shape)
    y = func(x0, x1)
    gx0, gx1 = func.backward()
    
    print(x0)
    print(x1)
    print(y)
    print(gx0)
    print(gx1)
    
    x0 = np.arange(24).reshape(2, 3, 4)
    x1 = np.arange(24).reshape(2, 4, 3)
    func = MatMul()
    print('test ', func.__class__.__name__, x0.shape, x1.shape)
    y = func(x0, x1)
    gx0, gx1 = func.backward()
    
    print(x0)
    print(x1)
    print(y)
    print(gx0)
    print(gx1)
    
    
