from pyaino.Config import *
set_seed(0); np=Config.np
from pyaino import Initializer

class BaseLayer:
    def __init__(self, m, n, activator='Mish', width=None, debug_mode=False, bias=True):
        self.size  = m, n
        self.width = width
        self.activator = activator
        self.bias = bias
        self.debug_mode = debug_mode

    def init_parameter(self):#, m, n):
        """ Neuron.BaseLayerから """
        m, n = self.get_parameter_size() # m:入力幅、n:ニューロン数
        if m is None or n is None:
            raise Exception('Configuration is not fixed.', self.__class__.__name__)

        self.w = Initializer.init_weight((m, n),
                                         width=self.width,
                                         activator=self.activator,
                                         debug_mode=self.debug_mode)        
      
        if self.bias:
            self.b = np.zeros(n, dtype=Config.dtype)                   

    def get_parameter_size(self):
        return self.size
        

init_w = Initializer.WeightInitializer(method='lecun')
w = init_w((3, 4))
print(w)

model = BaseLayer(3, 4)#, debug_mode=True)
model.init_parameter()
print('model.w\n', model.w)
print('model.b\n', model.b)
print('mean =', np.mean(model.w), 'std =', np.std(model.w))


