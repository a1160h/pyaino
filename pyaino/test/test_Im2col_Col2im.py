from pyaino.Config import *
from pyaino import Neuron

#                         B, C, Ih,Iw
x = np.arange(64).reshape(1, 1, 8, 8).astype(Config.dtype)
print('img\n', x)

#                      C, Ih,Iw,M, Fh,Fw,Sh,Sw
modelc = Neuron.Im2col(1, 8, 8, 1, 2, 2, 2, 2)

# Oh, Ow = 4, 4  

# (B, C, Ih,Iw) -> (B*Oh*Ow, C*Fh*Fw)
# (1, 1, 8, 8 )    (1*4 *4 , 1*2 *2 ) = (16, 4)   
y = modelc(x)                          
print('col\n', y)

modeld = Neuron.Col2im(1, 4, 4, 1, 2, 2, 2, 2)

# (B*Ih*Iw, M*Fh*Fw) -> (B, M, Oh, Ow)

z = modeld(y)
print('img\n', z)

assert (x==z).all()
