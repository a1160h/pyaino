import numpy as np


class VarianceFunction:
    def __init__(self):
        self.x = None
        self.N = None
        self.mean = None
    
    def forward(self, x):
        self.x = x
        y = np.var(x)
        return y
    
    def backward(self, dy):
        N = self.x.shape[0]
        mean = np.mean(self.x)
        dx = (2 / N) * (self.x - mean)
        return dx * dy



x = np.array([1, 2, 3, 4, 5])
print(x)
variance_function = VarianceFunction()
y = variance_function.forward(x)
print("Forward pass result (variance):", y)

dy = 1 # 逆伝播の入力として、通常は損失関数の勾配が与えられる
dx = variance_function.backward(dy)
print("Backward pass result (gradient of input x):", dx)

