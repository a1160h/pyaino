"""
pyaino: Define-by-run style automatic differentiation framework

"""

__version__ = "0.1.0"

# 基本モジュールの読み込み × numpy/cupy確定前に読み込むとエラーの原因になる
#from pyaino import Config
#from pyaino import nucleus
#from pyaino import Functions
#from pyaino import HDFunctions
#from pyaino import Neuron
#from pyaino import Activators
#from pyaino import Optimizers
#from pyaino import LossFunctions
#from pyaino import common_function
#from pyaino import CBOW
#from pyaino import NN_CNN
#from pyaino import RNN
#from pyaino import seq2seq
#from pyaino import VAE
#from pyaino import GAN

#from pyaino import MatMath
#from pyaino import Markov
#from pyaino import ELCA
#from pyaino import LCBF

# よく使うクラス・関数をトップレベルに公開
#from pyaino.nucleus import HDArray, Function, CompositFunction, asndarray
#from pyaino.Functions import Add, Mul, Exp, MatMul, Mean, Sum, Reshape
#from pyaino.Config import set_dtype, set_seed, set_np

# 公開APIを制限（必要に応じて）
'''
__all__ = [
    "Variable", "Function", "as_array",
    "Add", "Mul", "Exp", "MatMul", "Mean", "Sum", "Reshape",
    "set_backend", "get_backend", "backend_name"
]
'''
