# Diffuser
# 20251224 A.Inoue

from pyaino.Config import *
from pyaino import Functions as F

class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = np.zeros(num_timesteps+1, dtype=Config.dtype)
        self.betas[1:] = np.linspace(beta_start, beta_end, num_timesteps,
                                     dtype=Config.dtype)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas, axis=0)

    def fix_t(self, t, min_t=0, ndim=None):  # ndimはブロードキャストが必要な場合のみ
        T = self.num_timesteps
        t_arr = np.asarray(t, dtype=np.int64)
        assert (t_arr >= min_t).all() and (t_arr <= T).all()
        if ndim is None:
            return t
        return t_arr.reshape((-1,) + (1,) * (ndim - 1))

    def add_noise(self, x_0, t):
        t = self.fix_t(t, 0, x_0.ndim) # x_0 に次元を合わせる
        alpha_bar = self.alpha_bars[t]
        noise = np.random.randn(*x_0.shape).astype(x_0.dtype)
        x_t = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t):
        """ 基本のDDPM """
        t = self.fix_t(t, 1)
        t_prev = self.fix_t(t-1, 0)

        alpha     = self.alphas[t]      
        alpha_bar = self.alpha_bars[t]  
        alpha_bar_prev = self.alpha_bars[t_prev]  

        eps = model(x, t)  

        noise = np.random.randn(*x.shape).astype(x.dtype)
        noise[t==1] = 0 # t=1 の場合は全部0でノイズ無し

        mu  = (x - ((1 - alpha) / np.sqrt(1 - alpha_bar)) * eps) / np.sqrt(alpha)
        std = np.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        return mu + noise * std

    def denoise_ddpm(self, model, x, t, eta=1.0):
        """ DDPM posterior mean/var """
        # eta: noise scale. eta=1.0 -> standard DDPM,
        #                   eta=0.0 -> "mean-only" (tends to average/whiten)
        #print('eta =', eta)
        # サンプリングでは t はスカラ（または全要素同一の (N,)）とする
        t = self.fix_t(t, 1)
        
        # 係数はスカラのまま（broadcast は演算時に自動で効く）
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t - 1] if t >= 1 else 1.0

        eps = model(x, t)
        #print('eps_hat.std', np.std(eps))

        # x0_pre（clip前）
        x0_pre = (x - np.sqrt(1 - alpha_bar) * eps) / np.sqrt(alpha_bar)
        oor = (((x0_pre < -1.0) | (x0_pre > 1.0))).astype(np.float32).mean()

        # x0_hat（clip後）
        x0_hat = np.clip(x0_pre, -1.0, 1.0)

        delta = (np.abs(x0_hat - x0_pre)).astype(np.float32).mean()
        #print("t", int(t), "mean|clip_delta|", delta)

        # posterior mean / std
        coef1 = (np.sqrt(alpha_bar_prev) * (1 - alpha)) / (1 - alpha_bar)
        coef2 = (np.sqrt(alpha) * (1 - alpha_bar_prev)) / (1 - alpha_bar)
        mu = coef1 * x0_hat + coef2 * x
        std = np.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))

        # ログ（std 定義後に出す）
        #print("t", int(t), "x0_pre min/max", x0_pre.min(), x0_pre.max(), "oor", oor)
        #print("t", int(t), "x0_hat min/max", x0_hat.min(), x0_hat.max())
        #print("t", int(t), "std mean/max", std.mean(), std.max())

        noise = np.random.randn(*x.shape).astype(x.dtype)
        if t == 1:
            noise[:] = 0  # t=1 は deterministic

        return mu + eta * std * noise

    def _make_ddim_schedule(self, steps: int):
        """ T= self.num_timesteps を steps 個に間引いた t 列と t_prev 列を返す """
        T = self.num_timesteps
        steps = int(steps)
        if steps < 2:
            raise ValueError("steps must be >= 2")

        ts = np.linspace(1, T, steps, dtype=np.int64)
        ts = np.unique(ts)          # 重複除去（stepsが大きいと重複し得る）
        ts = ts[::-1]               # 降順: T -> ... -> 1

        t_prev = np.concatenate([ts[1:], np.array([0], dtype=np.int64)])
        return ts, t_prev

    def denoise_ddim(self, model, x, t, t_prev, eta=0.0, clip_denoised=False):
        """ DDIM """
        t = self.fix_t(t, 1)
        t_prev = self.fix_t(t_prev, 0)

        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev] if t_prev > 0 else 1.0

        eps = model(x, t)

        # x0 推定
        x0_hat = (x - np.sqrt(1.0 - alpha_bar) * eps) / np.sqrt(alpha_bar)
        if clip_denoised:
            x0_hat = np.clip(x0_hat, -1.0, 1.0)

        # DDIM の係数
        sigma = (eta * np.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
                     * np.sqrt(1.0 - alpha_bar / alpha_bar_prev))
        dir_coef = np.sqrt(np.maximum(1.0 - alpha_bar_prev - sigma * sigma, 0.0))

        noise = np.random.randn(*x.shape).astype(x.dtype)
        if t_prev == 0:
            noise[:] = 0

        x_prev = np.sqrt(alpha_bar_prev) * x0_hat + dir_coef * eps + sigma * noise
        return x_prev


    def sample_simple(self, model, x_shape=(20, 1, 28, 28)):
        batch_size = x_shape[0]
        x = np.random.randn(*x_shape)

        for t in range(self.num_timesteps, 0, -1):
            x = self.denoise(model, x, t)
            print(f'{t:4d} / {self.num_timesteps}'
                  + f'{np.min(x):7.3f} {np.max(x):7.3f}')
        return x    

    def sample(self, model, x_shape=(20, 1, 28, 28),
               sampler="ddpm", eta=1.0, steps=None, clip_denoised=False):
        batch_size = x_shape[0]
        x = np.random.randn(*x_shape).astype(Config.dtype)

        if sampler == "ddpm":
            for t in range(self.num_timesteps, 0, -1):
                x = self.denoise_ddpm(model, x, t, eta=eta)
            return x

        elif sampler == "ddim":
            if steps is None: # 間引き無し
                for t in range(self.num_timesteps, 0, -1):
                    x = self.denoise_ddim(model, x, t=t, t_prev=t-1,
                                          eta=eta, clip_denoised=clip_denoised)
                return x

            # 間引き
            ts, t_prev_list = self._make_ddim_schedule(steps=steps)
            for t, t_prev in zip(ts, t_prev_list):
                x = self.denoise_ddim(model, x, t=int(t), t_prev=int(t_prev),
                                          eta=eta, clip_denoised=clip_denoised)
            return x

        else:
            raise ValueError(f"Unknown sampler: {sampler}")

    def reverse_to_img(self, x):
        import numpy
        x = (x + 1) / 2 * 255
        x = np.clip(x, 0, 255)
        if not isinstance(x, numpy.ndarray): 
            x = numpy.asarray(x.get())  # cupyもnumpyに揃える
        x = x.astype(numpy.uint8).transpose(1,2,0)
        return x

    def loss(self, eps_pred, eps, weighting=False, t=None, gamma=1.0):
        """ 時刻に応じた重み付け可能な平均2乗誤差 """
        # gammaが小さい：減衰はゆるい。高 SNR もそこそこ学習させたいとき。
        # gammaが大きい：高 SNR の損失が一気に軽くなる。終盤の復元（低 SNR）を重視
        l = (eps_pred - eps)**2
        if weighting:
            t = self.fix_t(t, 0, eps.ndim)
            alpha_bar = self.alpha_bars[t]
            snr = alpha_bar / (1 - alpha_bar) # 信号雑音比
            w = (snr + 1)**(-gamma)
            l = l * w
            #print(l.shape, w.shape)
        return F.Mean()(l)



