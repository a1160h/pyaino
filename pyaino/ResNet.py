from pyaino.Config import *
set_derivative(True)
from pyaino import Neuron 
from pyaino import stems_blocks_heads as sbh
from pyaino import common_function as cf


class ResNetStage:
    def __init__(self, n_layer=3, base_ch=16, stride=1,
                 residual=True, pre_activation=False, **kwargs):
        
        activate  = kwargs.pop('activate',  'ReLU')
        optimize  = kwargs.pop('optimize', 'AdamT')
        w_decay   = kwargs.pop('w_decay',     0.01)
        batchnorm = kwargs.pop('batchnorm',   True)

        options = {'residual':residual,
                   'activate':activate,
                   'optimize':optimize,
                   'w_decay' :w_decay,
                   'batchnorm':batchnorm}
        
        strides = [stride if i == 0 else 1 for i in range(n_layer)]
        chanels = base_ch if stride == 1 else base_ch * 2
        if stride not in (1, 2):
            raise ValueError('Invalid stride specified.')

        self.blocks = Neuron.Sequential(
            *[sbh.ConvBlock(chanels, strides[i], **options)
              for i in range(n_layer)]
            )
        
    def forward(self, x, train=True):
        return self.blocks.forward(x, train=train)

    def update(self, **kwargs):
        self.blocks.update(**kwargs)


class CifarResNet:
    def __init__(self, classes=10, n_stage=3, base_ch=16, stride=2, 
                 residual=True, preactivation=False, **kwargs):

        self.stem = Neuron.Conv2dLayer(base_ch, 3, 1, **kwargs)

        chs_and_strides = [(base_ch, 1) if i==0 else (base_ch * i, stride)
                           for i in range(n_stage)] 
        
        self.stages = Neuron.Sequential(
            *[ResNetStage(3, *chs_and_strides[i], **kwargs) 
              for i in range(n_stage)]
            )

        self.head = sbh.ClassificationHead(classes=classes, **kwargs)

    def forward(self, x, t=None, train=True, dropout=0.0):
        y = self.stem.forward(x, train=train)
        y = self.stages.forward(y, train=train)
        y = self.head.forward(y, t=t, train=train, dropout=dropout)
        return y

    def update(self, **kwargs):
        self.stem.update(**kwargs)
        self.stages.update(**kwargs)
        self.head.update(**kwargs)

if __name__=='__main__':
    model = CifarResNet(n_stage=3, base_ch=16)

    cf.get_obj_info(model)

    B,C,H,W = 1,3,32,32
    x = np.arange(B*C*H*W).reshape(B,C,H,W)
    x = np.hdarray(x)
    t = np.array(3).reshape(1,-1)

    y, l = model.forward(x, t, train=True, dropout=0.5)

    l.backtrace()

    model.update()

    print(x.shape, y.shape, l, x.grad.shape)
