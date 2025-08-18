from pyaino.Config import *
set_np('numpy'); np=Config.np  


class Normalize:
    def forward(self, x):
        self.x  = x
        mu =  np.mean(x, axis=0)
        std = np.std(x, axis=0, ddof=0)
        self.std  = std
        self.mu   = mu
        y = (x - mu) / (std + 1e-12)
        self.y = y
        return y
    
    def backward(self, gy):
        istd = 1/self.std
        iN = 1/len(gy)
        xc = self.x - self.mu
        gy_sum = np.sum(dy * xc, axis=0, keepdims=True)
        gz   = (gy - (self.y * gy_sum * istd * iN)) * istd
        gz_sum = np.sum(gz, axis=0, keepdims=True)
        gx   = gz - (gz_sum * iN)
        return gx

x = np.array([1, 2, 3, 4, 5])
print(x)
func = Normalize()
y = func.forward(x)
print("Forward y =", y)
dy =  np.ones_like(y)#y #1 # 逆伝播の入力として、通常は損失関数の勾配が与えられる
dx = func.backward(dy)
print("Backward gx =", dx)

    
