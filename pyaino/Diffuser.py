# Diffuser
# 20260720 A.Inoue
#
# Base version
# - Experimental options removed
# - Standard DDPM/DDIM implementation
# - Preserved:
#     eta
#     gamma
#     clip_denoised
#     preserve_mean_beta
#     sample_state

from pyaino.Config import *
from pyaino import Functions as F
from pyaino import LossFunctions as lf
from pyaino import safe_np as snp
from pathlib import Path
import matplotlib.pyplot as plt
import copy
import re

"""
Diffusionの時刻体系:

    x_0 : clean image
    x_t : diffusion state, t = 1, ..., T

num_timesteps = T とし、係数配列はすべて長さ T+1 とする。

    betas[0]      = 0
    alphas[0]     = 1
    alpha_bars[0] = 1

    betas[1:T+1]      = beta_1, ..., beta_T
    alphas[1:T+1]     = alpha_1, ..., alpha_T
    alpha_bars[1:T+1] = alpha_bar_1, ..., alpha_bar_T
"""

class BetaSchedule:
    def __init__(self, num_timesteps=1000, schedule_type=None,
                 beta_start=0.0001, beta_end=0.02,
                 s=0.008, eps_final=1e-4):
        self.num_timesteps = int(num_timesteps)
        self.schedule_type = schedule_type
        if schedule_type == "cosine":
            self.schedule = self.cosine_schedule
        elif schedule_type == "linear":
            self.schedule = self.linear_schedule
        else:
            self.schedule = self.legacy_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.s = s
        self.eps_final = eps_final
        if self.num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")

    def __call__(self):
        return self.schedule()

    def legacy_schedule(self):
        """ DDPMの論文記載の古典的方法(β線形だがnum_timestepsが小さい場合に不完全) """
        betas = np.zeros(self.num_timesteps+1, dtype=Config.dtype)
        betas[1:] = np.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        alphas = 1 - betas
        alpha_bars = np.cumprod(alphas, axis=0)
        return betas, alphas, alpha_bars

    def cosine_schedule(self):
        """alpha_barをcosineに形状設計し、それを離散alphaに変換"""
        u = np.linspace(0.0, 1.0, self.num_timesteps + 1, dtype=Config.dtype)
        alpha_bars = np.cos((u + self.s) / (1.0 + self.s) * np.pi / 2.0) ** 2
        alpha_bars /= alpha_bars[0]
        betas = np.zeros_like(alpha_bars)
        betas[1:] = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas[1:] = np.clip(betas[1:], 1e-8, 0.999)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        return betas, alphas, alpha_bars

    def linear_schedule(self):
        """ betaを線形に設定して、その線形形状を保ちながら全体をスケーリング """
        T = self.num_timesteps
        beta_shape = np.linspace(self.beta_start, self.beta_end, T, dtype=Config.dtype)
        sum_beta_shape = np.sum(beta_shape)
        k_approx = -np.log(self.eps_final) / sum_beta_shape
        k_max = (1.0 / (np.max(beta_shape) + 1e-12))
        k = min(k_approx, k_max * 0.999)
        beta_steps = k * beta_shape
        beta_steps = np.clip(beta_steps, 1e-8, 0.999)
        betas = np.zeros(self.num_timesteps+1, dtype=Config.dtype)
        betas[1:] = beta_steps
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas, axis=0)
        return betas, alphas, alpha_bars

    def show_schedule(self):
        betas, alphas, alpha_bars = self.schedule()
        plt.figure(figsize=(8, 4))
        plt.plot(betas.tolist(), label="beta")
        plt.plot(alphas.tolist(), label="alpha")
        plt.plot(alpha_bars.tolist(), label="alpha_bar")
        plt.title(f"Schedule type: {self.schedule_type}")
        plt.xlabel("t  (0: clean, 1...T: diffusion)")
        plt.ylabel("coefficient")
        plt.grid(True)
        plt.legend()
        plt.show()

    def show_snr_and_w(self, gamma=1):
        betas, alphas, alpha_bars = self.schedule()
        snrs = alpha_bars / (1 - alpha_bars)
        w = (snrs + 1)**(-gamma)
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
        ax1.plot(snrs.tolist(), 'C0', label='snr')
        ax2.plot(w.tolist(), 'C1', label='w')
        ax1.set_ylabel('snr')
        ax2.set_ylabel('w')
        ax1.set_title(f"snr and w: {self.schedule_type} gamma={gamma}")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='center right')
        ax1.set_xlabel("t  (0: clean → T: noise)")
        ax1.grid()
        plt.show()


class Diffuser:
    def __init__(self, num_timesteps=1000, beta_schedule=None,
                 step_log=False, weighting=False):
        self.beta_schedule = BetaSchedule(num_timesteps, beta_schedule)
        self.betas, self.alphas, self.alpha_bars = self.beta_schedule()
        self.num_timesteps = num_timesteps
        self.step_log = step_log
        self.weighting = weighting
        if step_log: # 時刻毎のエラー記録
            self.stat_sum = np.zeros(num_timesteps+1, dtype=float)
            self.stat_cnt = np.zeros(num_timesteps+1, dtype=np.int32)
        self.kwargs = None # denoise時のオプションを覚えておく
        self.analizer = Analizer(self)

    def fix_t(self, t, min_t=0, ndim=None):  # ndimはブロードキャストが必要な場合のみ
        """
        時刻 t を整数配列に整形、必要に応じブロードキャスト可能な形 (B,1,...,1) に変換
        """
        T = self.num_timesteps
        t_arr = np.asarray(t, dtype=np.int32)
        assert (t_arr >= min_t).all() and (t_arr <= T).all()
        if ndim is None:
            return t_arr
        return t_arr.reshape((-1,) + (1,) * (ndim - 1))

    def schedule_time_steps(self, steps=None):
        """
        ddim逆拡散反復に用いる時刻遷移列(ts, t_prevs)

        完全ステップ　: T -> T-1 -> ... -> 1 -> 0
        間引きステップ: Tから1までをほぼ等間隔に選び、最後に0へ遷移する
        """
        T = self.num_timesteps
        steps = None if steps is None else int(steps)
        if steps is None or steps >= T:
            ts = np.arange(T, 0, -1, dtype=np.int32)
        else:
            assert steps > 0
            ts = np.rint(np.linspace(T, 1, steps)).astype(np.int32)
            # 丸めで重複した時刻を除去する。
            ts = np.unique(ts)[::-1] # np.uniqueは昇順に並べるため降順へ戻す
        t_prevs = np.concatenate([ts[1:], np.array([0], dtype=np.int32)])
        return ts, t_prevs

    def predict(self, model, x, t, labels=None, gamma=None):
        """モデル出力を取得し、必要に応じてclassifier-free guidanceを適用する。"""
        if gamma is None:
            return model(x, t, labels)
        y = model(x, t, labels)
        y_uncond = model(x, t)
        return y_uncond + gamma * (y - y_uncond)

    def add_noise(self, x0, t, noise=None, dc_removal=False):
        t = self.fix_t(t, 0, x0.ndim) # x0 に次元を合わせる
        alpha_bar = self.alpha_bars[t]
        if noise is None:
            noise = np.random.randn(*x0.shape).astype(x0.dtype)
        if dc_removal:    
            noise = noise - noise.mean(axis=(2,3), keepdims=True) # DC抑止 20260131AI　
        x_t = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t, t_prev, labels=None, **kwargs):
        """ eps予測モデルを用いる基本のDDPM """
        eta           = kwargs.pop('eta', 1.0) # eta=1が基本　
        gamma         = kwargs.pop('gamma', None)
        
        t = self.fix_t(t, 1)
        t_prev = self.fix_t(t_prev, 0)

        alpha     = self.alphas[t]      
        alpha_bar = self.alpha_bars[t]  
        alpha_bar_prev = self.alpha_bars[t_prev] #if t >=1 else 1.0 

        eps = self.predict(model, x, t, labels, gamma)

        std = eta * np.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        coef1 = 1 / np.sqrt(alpha)
        coef2 = - (1 - alpha) / np.sqrt(alpha * (1 - alpha_bar))
        mu = coef1 * x + coef2 * eps
         
        if int(t_prev) == 0:
            x_prev = mu
        else:    
            noise = np.random.randn(*x.shape).astype(x.dtype)
            x_prev = mu + std * noise
        return x_prev


    def denoise_x0(self, model, x, t, t_prev, labels=None, **kwargs):
        """x0 予測モデルを用いる基本のDDPM """
        eta   = kwargs.pop('eta', 1.0)   # eta=1 が基本
        gamma = kwargs.pop('gamma', None)

        t      = self.fix_t(t, 1)
        t_prev = self.fix_t(t_prev, 0)

        alpha          = self.alphas[t]
        alpha_bar      = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev]

        x0_hat = self.predict(model, x, t, labels, gamma)

        # q(x_{t-1} | x_t, x0) の平均とposterior variance
        std = eta * np.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        coef1 = (np.sqrt(alpha) * (1 - alpha_bar_prev)) / (1 - alpha_bar)
        coef2 = (np.sqrt(alpha_bar_prev) * (1 - alpha)) / (1 - alpha_bar)
        mu = coef1 * x + coef2 * x0_hat 

        if int(t_prev) == 0:
            x_prev = mu
        else:
            noise = np.random.randn(*x.shape).astype(x.dtype)
            x_prev = mu + std * noise
        return x_prev


    def dynamic_thresholding(self, x0, p=0.995, clip_val=1.0, 
                             preserve_mean=True, lam=1.0,
                             axis=(2,3),
                             soft_clip=False):
        """Quantile-based adaptive clipping."""
        if soft_clip:
            clip = lambda x, s : s * np.tanh(x / s)
        else:    
            clip = lambda x, s : np.clip(x, -s, s)
        
        # sは小さい順に並べたときのpで指定した割合の位置にある値(外れ値とみなす値)
        s = np.quantile(np.abs(x0), p, axis=axis, keepdims=True)
        s = np.maximum(s, clip_val)
        x0_clip = clip(x0, s)
        clip_rate = np.mean(x0!=x0_clip, axis=axis)
        if preserve_mean: # 元の平均を温存する
            x0_clip += lam * (x0.mean(axis=axis, keepdims=True)
                              - x0_clip.mean(axis=axis, keepdims=True))
        return x0_clip, s.reshape(-1), clip_rate

    def regularize_eps(self, eps, beta=0.0, axis=(2,3)):
        mean = snp.mean(eps, axis=axis, keepdims=True)
        var  = snp.var(eps, axis=axis, keepdims=True)
        std  = np.sqrt(var + 1e-8)
        eps *= (1.0 - beta) + beta / std
        eps -= beta * mean / std
        return eps

    def pred_x0_from_eps(self, x, alpha_bar, eps):
        return (x - np.sqrt(1.0 - alpha_bar) * eps) / np.sqrt(alpha_bar)

    def pred_eps_from_x0(self, x, alpha_bar, x0):
        eps = (x - np.sqrt(alpha_bar) * x0) / np.sqrt(np.maximum(1.0 - alpha_bar, 1e-8))
        return eps

    def sample_step(self, x, mu, std, t_prev, preserve_mean_beta=0.0, dc_axis=(0, 2, 3)):
        """平均muと標準偏差stdから次状態x_prevを生成する。"""
        if int(t_prev) == 0:
            x_prev = mu
            noise_term = np.zeros_like(x_prev, dtype=Config.dtype)
        else:
            noise = np.random.randn(*x.shape).astype(x.dtype)
            noise_term = std * noise
            x_prev = mu + noise_term
        if preserve_mean_beta > 0:
            x_prev += (preserve_mean_beta
                       * ((x.mean(axis=dc_axis, keepdims=True)
                           - x_prev.mean(axis=dc_axis, keepdims=True))))
        return x_prev, noise_term

    def denoise_ddpm(self, model, x, t, t_prev, labels=None, debug=False, **kwargs):
        """ DDPM posterior mean/var """
        self.kwargs    = kwargs.copy() # 何を指定したかを覚えておく
        eta            = kwargs.pop('eta', 1.0) # サンプリング時のノイズスケール
        #   eta = 1.0 -> DDPM と同じノイズ量
        #   eta = 0.0 -> 事後平均のみを辿る（ノイズなしの deterministic ステップ）
        clip_denoised  = kwargs.pop('clip_denoised', False)
        dt_p           = kwargs.pop('dt_p', 0.995) # clipオプション
        preserve_mean_beta = kwargs.pop('preserve_mean_beta', 0.0)
        dc_axis        = kwargs.pop('dc_axis', (0, 2, 3)) # 補正軸
        gamma          = kwargs.pop('gamma', None) # 誘導オプション
        mu_mode        = kwargs.pop('mu_mode', 'eps') # μ計算オプション

        sample_state   = kwargs.pop('sample_state', None)  # API互換のため保持（現在は未使用）

        t = self.fix_t(t, 1)
        t_prev = self.fix_t(t_prev, 0)
        
        alpha          = self.alphas[t]
        alpha_bar      = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev]

        """ step1 : model -> eps -> x0_hat """
        # モデルによるeps予測、x0_hat推定
        eps = self.predict(model, x, t, labels, gamma)
        x0_hat = self.pred_x0_from_eps(x, alpha_bar, eps)
        
        """ step2 : clip等 """
        # x0_hatをクリップするとともにepsをそれに合わせて再計算
        if clip_denoised:
            x0_hat, s, clip_rate \
                = self.dynamic_thresholding(x0_hat, p=dt_p, clip_val=1.0)
            eps = self.pred_eps_from_x0(x, alpha_bar, x0_hat) # clipしたx0_hatに合わせる
        else:
            s =0
            clip_rate = np.zeros(x0_hat.shape[:2], dtype=Config.dtype) # ログ用

        """ step3 : eps, x0_hat -> std, mu """
        # q(x_prev | x_t, x0) のガウス事後分布のパラメータ(平均と分散)
        std = eta * np.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        coef1 = (np.sqrt(alpha) * (1 - alpha_bar_prev)) / (1 - alpha_bar)
        coef2 = (np.sqrt(alpha_bar_prev) * (1 - alpha)) / (1 - alpha_bar)
        coef3 = 1 / np.sqrt(alpha)
        coef4 = - (1 - alpha) / np.sqrt(alpha * (1 - alpha_bar))

        if   mu_mode == 'x0':       # x0とxから算出
            mu = coef1 * x + coef2 * x0_hat 
        elif mu_mode == 'eps':      # x0_hatと整合するepsを使って算出
            mu = coef3 * x + coef4 * eps
        else:
            raise ValueError(f"Unknown mu_mode: {mu_mode}")

        """ step4 : mu, std -> x_prev """
        x_prev, noise_term = self.sample_step(
            x, mu, std, t_prev, preserve_mean_beta, dc_axis)

        """ logging """
        if debug:
            self.analizer.append_log(
                t, x, x_prev, x0_hat, eps, mu, noise_term, clip_rate
                )
        return x_prev


    def denoise_ddim(self, model, x, t, t_prev, labels=None, debug=False, **kwargs):
        """ DDIM (Denoising Diffusion Implicit Models) """
        self.kwargs    = kwargs.copy() # 何を指定したかを覚えておく
        eta            = kwargs.pop('eta', 0.0) # サンプリング時のノイズスケール
        #   eta = 1.0 -> DDPM と同じノイズ量
        #   eta = 0.0 -> 事後平均のみを辿る（ノイズなしの deterministic ステップ）
        clip_denoised  = kwargs.pop('clip_denoised', False)
        dt_p           = kwargs.pop('dt_p', 0.995) # clipオプション
        preserve_mean_beta = kwargs.pop('preserve_mean_beta', 0.0)
        dc_axis        = kwargs.pop('dc_axis', (0, 2, 3)) # 補正軸
        gamma          = kwargs.pop('gamma', None) # 誘導オプション

        sample_state   = kwargs.pop('sample_state', None)  # API互換のため保持（現在は未使用）

        t = self.fix_t(t, 1)
        t_prev = self.fix_t(t_prev, 0)
        
        alpha_bar      = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev]

        """ step1 : model -> eps -> x0_hat """
        eps = self.predict(model, x, t, labels, gamma)
        x0_hat = self.pred_x0_from_eps(x, alpha_bar, eps)

        """ step2 : clip等 """
        if clip_denoised:
            x0_hat, s, clip_rate \
                = self.dynamic_thresholding(x0_hat, p=dt_p, clip_val=1.0)
            eps = self.pred_eps_from_x0(x, alpha_bar, x0_hat)
        else:
            s = 0
            clip_rate = np.zeros(x0_hat.shape[:2], dtype=Config.dtype)

        """ step3 : x0_hat, eps -> std, mu """
        if eta == 0.0:
            std = np.zeros_like(alpha_bar, dtype=Config.dtype)
            dir_coef = np.sqrt(np.maximum(1.0 - alpha_bar_prev, 0.0))
        else:
            std = (np.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
                 * np.sqrt(1.0 - alpha_bar / alpha_bar_prev))
            std *= eta # noise_std
            dir_coef = np.sqrt(np.maximum(1.0 - alpha_bar_prev - std * std, 0.0))

        mu = np.sqrt(alpha_bar_prev) * x0_hat + dir_coef * eps

        """ step4 : mu, std -> x_prev """
        x_prev, noise_term = self.sample_step(
            x, mu, std, t_prev, preserve_mean_beta, dc_axis)
    
        """ logging """
        if debug:
            self.analizer.append_log(
                t, x, x_prev, x0_hat, eps, mu, noise_term, clip_rate
            )
        return x_prev

    def denoise_ddpm_x0(self, model, x, t, t_prev, labels=None, debug=False, **kwargs):
        """ x0 予測モデルを用いるDDPM posterior mean/var """
        self.kwargs    = kwargs.copy() # 何を指定したかを覚えておく
        eta            = kwargs.pop('eta', 1.0) # サンプリング時のノイズスケール
        #   eta = 1.0 -> DDPM と同じノイズ量
        #   eta = 0.0 -> 事後平均のみを辿る（ノイズなしの deterministic ステップ）
        clip_denoised  = kwargs.pop('clip_denoised', False)
        dt_p           = kwargs.pop('dt_p', 0.995) # clipオプション
        preserve_mean_beta = kwargs.pop('preserve_mean_beta', 0.0)
        eps_stdz_beta  = kwargs.pop('eps_stdz_beta', 0.0)
        dc_axis        = kwargs.pop('dc_axis', (0, 2, 3)) # 補正軸
        gamma          = kwargs.pop('gamma', None) # 誘導オプション

        sample_state   = kwargs.pop('sample_state', None)  # API互換のため保持（現在は未使用）

        t      = self.fix_t(t, 1)
        t_prev = self.fix_t(t_prev, 0)

        alpha          = self.alphas[t]
        alpha_bar      = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev]

        """ step1 : model -> x0 -> eps """
        # モデルによるx0予測、eps推定
        x0_hat = self.predict(model, x, t, labels, gamma)
        eps = self.pred_eps_from_x0(x, alpha_bar, x0_hat)

        """ step2 : 正則化、clip等 """
        if eps_stdz_beta > 0:
            eps = self.regularize_eps(eps, eps_stdz_beta, dc_axis)
            x0_hat = self.pred_x0_from_eps(x, alpha_bar, eps)
            
        if clip_denoised:
            x0_hat, s, clip_rate \
                = self.dynamic_thresholding(x0_hat, p=dt_p, clip_val=1.0)
            eps = self.pred_eps_from_x0(x, alpha_bar, x0_hat)
        else:
            s =0
            clip_rate = np.zeros(x0_hat.shape[:2], dtype=Config.dtype) # ログ用

        """ step3 : x0_hat -> std, mu """
        std = eta * np.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        coef1 = (np.sqrt(alpha) * (1 - alpha_bar_prev)) / (1 - alpha_bar)
        coef2 = (np.sqrt(alpha_bar_prev) * (1 - alpha)) / (1 - alpha_bar)
        mu = coef1 * x + coef2 * x0_hat 

        """ step4 : mu, std -> x_prev """
        x_prev, noise_term = self.sample_step(
            x, mu, std, t_prev, preserve_mean_beta, dc_axis)

        """ logging """
        if debug:
            self.analizer.append_log(
                t, x, x_prev, x0_hat, eps, mu, noise_term, clip_rate
                )
        return x_prev

    def sample(self, model, x_shape=(20, 1, 28, 28), x=None, labels=None,
               sampler='legacy', steps=None, start=None, halt=None, debug=False,
               batch_size=10, **kwargs):

        if start is not None and x is None:
            raise ValueError("x must be supplied when start is specified")

        if x is None:
            x = np.random.randn(*x_shape).astype(Config.dtype)

        ts, t_prevs = self.schedule_time_steps(steps=steps)

        if sampler != "ddim" and steps is not None:
            raise ValueError("steps can be specified only when sampler='ddim'")

        if sampler == "legacy":
            denoise_fn = self.denoise
        elif sampler == "legacy_x0":
            denoise_fn = self.denoise_x0
        elif sampler == "ddpm":
            denoise_fn = self.denoise_ddpm
        elif sampler == "ddim":
            denoise_fn = self.denoise_ddim
        elif sampler == "ddpm_x0":
            denoise_fn = self.denoise_ddpm_x0
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        kwargs.setdefault("sample_state", {})

        if labels is not None:
            labels = np.array(labels)
            if len(x)!=len(labels):
                raise ValueError(f'Wrong size of labels: len(x)={len(x)}, len(labels)={len(labels)}')
        for b in range(0, len(x), batch_size):
            xb = x[b:b+batch_size]
            lb = None if labels is None else labels[b:b+batch_size]
          
            for i, (t, t_prev) in enumerate(zip(ts, t_prevs)):
                if start is not None and t > start:
                    continue
                
                xb = denoise_fn(model, xb, t, t_prev, labels=lb, debug=debug, **kwargs)

                if halt is not None and t<=(self.num_timesteps-halt):
                    print(f'halt={halt} t={t}')
                    break
            x[b:b+batch_size] = xb     
        return x    


    def reverse_to_img(self, x):
        import numpy
        x = (x + 1) / 2 * 255
        x = np.clip(x, 0, 255)
        if not isinstance(x, numpy.ndarray): 
            x = numpy.asarray(x.get())  # cupyもnumpyに揃える
        x = x.astype(numpy.uint8).transpose(1,2,0)
        return x

    def loss(self, eps_hat, eps, t=None, gamma=1.0, dc_reg=False, lam=1e-3):
        """ 与えたノイズと予測したノイズの隔たりで、
          　時刻に応じたエラー集計と時刻に応じた重み付け可能な平均2乗誤差 """
        l = lf.MeanSquaredError(reduction=None)(eps_hat, eps) 
        l = l.mean(axis=(1,2,3)) 
        # 時刻tはstep_log,weightingの両方に使う
        if t is not None:
            t = self.fix_t(t, 1, None)
        if self.step_log and t is not None: # 時刻毎のエラー集計
            #print(t, l)
            np.add.at(self.stat_sum, t, l) 
            np.add.at(self.stat_cnt, t, 1)
        if self.weighting and t is not None:     # 時刻に応じた重付け
            # gammaが小さい：減衰はゆるい。高 SNR もそこそこ学習させたいとき。
            # gammaが大きい：高 SNR の損失が一気に軽くなる。終盤の復元（低 SNR）を重視
            alpha_bar = self.alpha_bars[t]
            snr = alpha_bar / (1 - alpha_bar) # 信号雑音比
            w = (snr + 1)**(-gamma)
            l = l * w
        loss = l.mean()    
        if dc_reg:    
            dc = F.Mean(axis=(2,3), keepdims=True)(eps_hat) # B,Cごと
            dc_l = F.Mean()(dc**2)  
            loss += lam * dc_l
        return loss 

    def loss_x0(self, x0_hat, x0, t=None):
        """予測画像 x0_hat と正解画像 x0 の単純な平均2乗誤差。

        純粋な x0 予測を観察するため、SNR重み付けなどは行わない。
        step_log=True の場合のみ、時刻別MSEを記録する。
        """
        l = lf.MeanSquaredError(reduction=None)(x0_hat, x0)
        l = l.mean(axis=(1,2,3)) 
        if self.step_log and t is not None:
            t = self.fix_t(t, 1, None)
            np.add.at(self.stat_sum, t, l)
            np.add.at(self.stat_cnt, t, 1)
        return l.mean()


    def ddim_inversion(self, model, x0,
                       tend=500,
                       steps=None,
                       use_true_x0_for_test=False, 
                       ):
        """DDIM inversion: x0 (t=0) から x_tend を生成（eta=0前提）"""

        # schedule_time_steps は降順の遷移列 (t -> t_prev) を返すので、
        # inversion ではこれを反転して昇順の遷移列 (t_from -> t_to) として用いる
        ts, t_prevs = self.schedule_time_steps(steps=steps)
        t_froms = t_prevs[::-1] # 逆順
        t_tos   = ts[::-1]      # 逆順

        x = x0
        for t_from, t_to in zip(t_froms, t_tos):
            if int(t_to) > int(tend):
                break

            alpha_bar_from = self.alpha_bars[t_from]
            alpha_bar_to   = self.alpha_bars[t_to]

            # eps 予測（x_{t_from}）
            eps_hat = model(x, t_from)

            # x0 推定
            # inversion の出発点は x0 であるが、
            # 反復の各時刻で直接使えるのは現在状態 x_{t_prev} のみである。
            # DDIM の決定的更新は x_{t_prev}, eps_hat, x0_hat の関係で定まるため、
            # 各ステップでは eps_hat から x0_hat を再推定し、
            # それを用いて次の x_t を構成する。
            x0_hat = (x - np.sqrt(1.0 - alpha_bar_from) * eps_hat) \
                     / np.sqrt(alpha_bar_from)
            x0_used = x0 if use_true_x0_for_test else x0_hat # 検証用: 真の x0 を固定

            # 決定的 DDIM forward: x_{t_to}
            x = np.sqrt(alpha_bar_to) * x0_used + np.sqrt(1.0 - alpha_bar_to) * eps_hat
        return x

class Analizer:
    """ Diffuserのdenoiseのlogの収集、解析、グラフ描画を担う """
    def __init__(self, diffuser):
        self.diffuser = diffuser
        self.log = []             
    
    def append_log(self, t, x, x_prev, x0_hat, eps, mu, noise_term, clip_rate,
                   axis=(2,3)):
        """ logを収集する """
        if np.may_share_memory(x, x_prev):
            print("WARN share_memory: t=", int(t))

        self.log.append({
            't'             : t,
            'mu_xt'         : x.mean(axis=axis),
            'std_xt'        : x.std(axis=axis),
            'mu_x_prev'     : x_prev.mean(axis=axis),
            'std_x_prev'    : x_prev.std(axis=axis),
            'mu_x0_hat'     : x0_hat.mean(axis=axis),
            'std_x0_hat'    : x0_hat.std(axis=axis),
            'mu_eps'        : eps.mean(axis=axis),
            'std_eps'       : eps.std(axis=axis),
            'mu_mu'         : mu.mean(axis=axis),
            'mu_noise_term' : noise_term.mean(axis=axis),
            'clip_rate'     : clip_rate,
            })

    def get_stem(self, epoch):
        stem = f"epoch{epoch:03d}"
        if self.diffuser.kwargs is not None:
            stem += '_'.join([f"{k}_{v}" for k, v in sorted(self.diffuser.kwargs.items())])
        stem = re.sub(r'[^A-Za-z0-9\_]', '', stem)
        return stem
        
    def save_log(self, epoch, out_dir: Path, flush=True):
        if not self.log:
            raise RuntimeError(
                "diffuser.log is empty. Did you run sample(debug=True)?"
            )

        stem = self.get_stem(epoch)
        file = str(out_dir / f"log_summary_{stem}.npz")
        keys = self.log[0].keys()
        data = {
            key: np.stack([row[key] for row in self.log], axis=0)
            for key in keys
        }
        np.savez(file, **data)

        if flush:
            self.log.clear()

        return file

    def rgb_plot(self, xs, t, title, out_png,
                 series_names=None,
                 channel_names=("R", "G", "B"),
                 ylim=None):
        if not isinstance(xs, (list, tuple)):
            xs = (xs,)

        if series_names is None:
            series_names = tuple(f"s{i}" for i in range(len(xs)))
        elif isinstance(series_names, str):
            series_names = (series_names,)

        plt.figure()
        for series_name, values in zip(series_names, xs):
            for c, channel_name in enumerate(channel_names):
                plt.plot(
                    t.tolist(),
                    values[:, c].tolist(),
                    label=f"{series_name}_{channel_name}",
                )

        plt.gca().invert_xaxis()
        plt.ticklabel_format(useOffset=False)
        plt.title(title)
        plt.xlabel("t")
        plt.grid(True)
        plt.legend(fontsize=9)

        if ylim is not None:
            plt.ylim(*ylim)

        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        
    def analize_and_draw(self, epoch, out_dir: Path, suffix=None):
        stem = self.get_stem(epoch)
        file = str(out_dir / f"log_summary_{stem}.npz")
        data = np.load(file)
        t = data['t']

        plots = (
            ('xt', data['mu_xt'], data['std_xt']),
            ('x_prev', data['mu_x_prev'], data['std_x_prev']),
            ('x0_hat', data['mu_x0_hat'], data['std_x0_hat']),
            ('eps', data['mu_eps'], data['std_eps']),
        )

        for name, mean, std in plots:
            mean = mean.mean(axis=1)
            std = std.mean(axis=1)
            filename = f"{name}_{stem}"
            if suffix is not None:
                filename += f"_{suffix}"
            filename += ".png"

            self.rgb_plot(
                (mean, std),
                t,
                title=f"mean and std: {name}",
                out_png=str(out_dir / filename),
                series_names=(f"mu_{name}", f"std_{name}"),
            )


    def get_log(self, key=None):
        if key is None:
            return self.log
        return [row[key] for row in self.log]



