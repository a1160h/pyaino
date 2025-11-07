from pyaino.Config import *
from pyaino import Neuron


#                        B, C, Iw
x = np.arange(8).reshape(1, 1, 8).astype(Config.dtype)
print('img\n', x)

#                C, Iw,M, Fw,S
modelc = Neuron.Vec2col(1, 8, 1, 2, 2)

# Ow = 4  

# (B, C, Iw) -> (B*Ow, C*Fw)
# (1, 1, 8 )    (1*4 , 1*2 ) = (4, 2)   
y = modelc(x)                          
print('col\n', y)

#                C, Iw,M, Fw,S 
modeld = Neuron.Col2vec(1, 4, 1, 2, 2)

# (B*Iw, M*Fw) -> (B, M, Ow)

z = modeld(y)
print('img\n', z)

assert (x==z).all()
