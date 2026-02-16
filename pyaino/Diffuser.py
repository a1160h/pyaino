# Diffuser
# 20260215 A.Inoue

from pyaino.Config import *
from pyaino import Functions as F
from pyaino import LossFunctions as lf

class BetaSchedule:
    def __init__(self, num_timesteps=1000, schedule_type=None,
                 beta_start=0.0001, beta_end=0.02,
                 s=0.008, eps_final=1e-4):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        if schedule_type == 'cosine':
            self.schedule = self.cosine_schedule
        elif schedule_type == 'linear':
            self.schedule = self.linear_schedule
        else:
            self.schedule = self.legacy_schedule
        self.beta_start = beta_start # linear/legacy 線形形状の下限
        self.beta_end = beta_end     # linear/legacy 線形形状の上限
        self.s = s                   # cosine 端点調整パラメータ（通常 0.008）
        self.eps_final = eps_final   # linear 最終ステップでの alpha_bar_T の目標値

    def __call__(self):
        return self.schedule() 
       
    def legacy_schedule(self):
        """ DDPMの論文記載の古典的方法(β線形だがnum_timestepsが小さい場合に不完全) """
        betas = np.zeros(self.num_timesteps, dtype=Config.dtype)
        betas[1:] = np.linspace(self.beta_start, self.beta_end, self.num_timesteps-1)
        alphas = 1 - betas
        alpha_bars = np.cumprod(alphas, axis=0)
        return betas, alphas, alpha_bars
    
    def cosine_schedule(self):
        """ alpha_barを余弦スケジュールに基づきを計算 """
        
        t = np.linspace(0, self.num_timesteps, self.num_timesteps+1) \
            / self.num_timesteps  # 0〜1 の連続時間に正規化
        alpha_bars = np.cos((t + self.s) / (1 + self.s) * np.pi / 2) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]  
        alphas = alpha_bars[1:] / alpha_bars[:-1]
        betas = 1.0 - alphas
        return betas, alphas, alpha_bars

    def linear_schedule(self):
        """ βの線形を保ちつつスケール調整しalpha_bar_T≈eps_finalを作る　"""
        # 1. まず「形」だけ線形に作る（0〜1の区間での線形）
        t = np.arange(self.num_timesteps, dtype=np.float64)
        beta_shape = (self.beta_start + (self.beta_end - self.beta_start) * t
                   / max(self.num_timesteps - 1, 1))
        # 2. スケール係数 k を決める
        #    log alpha_bar_num_timesteps ≈ -k * sum(beta_shape) を使った近似
        sum_beta_shape = np.sum(beta_shape)
        # eps_final = exp(-k * sum_beta_shape) → k ≈ -log(eps_final) / sum_beta_shape
        k_approx = -np.log(self.eps_final) / (sum_beta_shape + 1e-12)

        # 3. β_t = k * beta_shape としたときに、β_t < 1 を保証するためのクリップ
        k_max = 1.0 / (np.max(beta_shape) + 1e-12)
        k = min(k_approx, k_max * 0.999)  # ちょっとだけ余裕を持たせる

        betas = k * beta_shape
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        return betas, alphas, alpha_bars

    def show_schedule(self):
        import matplotlib.pyplot as plt
        betas, alphas, alpha_bars = self.schedule()
        plt.figure(figsize=(8, 4))
        plt.plot(betas.tolist(), label='beta')
        plt.plot(alphas.tolist(), label='alpha')
        plt.plot(alpha_bars.tolist(), label='alpha_bar')
        plt.title(f"Schedule type: {self.schedule_type} alpha_bar(t)")
        plt.xlabel("t")
        plt.ylabel("alpha_bar")
        plt.grid(True)
        plt.legend()
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
            self.stat_sum = np.zeros(num_timesteps, dtype=float)
            self.stat_cnt = np.zeros(num_timesteps, dtype=np.int32)
        self.log = []          # sampleとdenoiseのログ
        self.debug_info = None # sampleとdenoiseのデバグ情報のやりとり用

    def fix_t(self, t, min_t=0, ndim=None):  # ndimはブロードキャストが必要な場合のみ
        T = self.num_timesteps
        t_arr = np.asarray(t, dtype=np.int32)
        assert (t_arr >= min_t).all() and (t_arr < T).all()
        if ndim is None:
            return t_arr
        return t_arr.reshape((-1,) + (1,) * (ndim - 1))

    def schedule_time_steps(self, steps=None):
        """ T= self.num_timesteps を steps 個に間引いた t 列と t_prev 列を返す """
        T = self.num_timesteps
        if steps is None:
            steps = T
        steps = int(steps)
        if steps < 2:
            raise ValueError("steps must be >= 2")

        ts = np.linspace(1, T-1, steps, dtype=np.int32)
        ts = np.unique(ts)          # 重複除去（stepsが大きいと重複し得る）
        ts = ts[::-1]               # 降順: T -> ... -> 1

        t_prev = np.concatenate([ts[1:], np.array([0], dtype=np.int32)])
        return ts, t_prev

    def schedule_time_steps2(self, steps=None):
        T = self.num_timesteps
        if steps is None or int(steps) == T:
            ts = np.arange(T-1, 0, -1, dtype=np.int32)        # 999,998,...,1
            t_prev = np.arange(T-2, -1, -1, dtype=np.int32)   # 998,...,0
            return ts, t_prev
        steps = int(steps)
        if steps < 2:
            raise ValueError("steps must be >= 2")

        ts = np.linspace(1, T-1, steps, dtype=np.int32)
        ts = np.unique(ts)[::-1]
        t_prev = np.concatenate([ts[1:], np.array([0], dtype=np.int32)])
        return ts, t_prev

    def add_noise(self, x_0, t, noise=None, dc_removal=False):
        t = self.fix_t(t, 0, x_0.ndim) # x_0 に次元を合わせる
        alpha_bar = self.alpha_bars[t]
        if noise is None:
            noise = np.random.randn(*x_0.shape).astype(x_0.dtype)
        if dc_removal:    
            noise = noise - noise.mean(axis=(2,3), keepdims=True) # DC抑止 20260131AI　
        x_t = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t, t_prev, **kwargs):
        """ 基本のDDPM """
        eta           = kwargs.pop('eta', 1.0) # eta=1が基本　
        labels        = kwargs.pop('labels', None)  
        gamma         = kwargs.pop('gamma', None)
        debug         = kwargs.pop('debug', False)
        
        t = self.fix_t(t, 1)
        t_prev = self.fix_t(t_prev, 0)

        alpha     = self.alphas[t]      
        alpha_bar = self.alpha_bars[t]  
        alpha_bar_prev = self.alpha_bars[t_prev] #if t >=1 else 1.0 

        if gamma is not None:
            eps = model(x, t, labels)
            eps_uncond = model(x, t)
            eps = eps_uncond + gamma * (eps - eps_uncond)
        else:
            eps = model(x, t, labels)

        mu  = (x - ((1 - alpha) / np.sqrt(1 - alpha_bar)) * eps) / np.sqrt(alpha)
        std = np.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
         
        if int(t_prev) == 0:
            x_prev = mu
        else:    
            noise = np.random.randn(*x.shape).astype(x.dtype)
            x_prev = mu + eta * noise * std
        return x_prev

    def dynamic_thresholding_bkup(self, x0, p=0.995, clip_val=1.0, eps=1e-6,
        preserve_mean=True, preserve_mean_iter=1, return_stats=False):
        """
        Dynamic thresholding (per-sample scaling + clip) with optional mean-preserving correction.

        Parameters
        ----------
        x0 : ndarray (B,C,H,W)
            Predicted x0 before clipping.
        p : float
            Quantile for dynamic thresholding (e.g. 0.995).
        clip_val : float
            Clip range becomes [-clip_val, +clip_val].
        eps : float
            Numerical stability.
        preserve_mean : bool
            If True, apply a post-clip mean correction per-channel so that
            mean(x0_after) ≈ mean(x0_before) (measured after scaling).
        preserve_mean_iter : int
            Number of correction iterations (1 is usually enough).
        return_stats : bool
            If True, returns a dict of stats for debugging.

        Returns
        -------
        x0_clip : ndarray (B,C,H,W)
        s : ndarray (B,)
            Per-sample scale (>= clip_val).
        stats : dict (optional)
            Contains mu_pre_RGB, mu_post_RGB, std_pre_RGB, clip_rate_RGB, etc.
        """
        # x0: (B,C,H,W)
        ax = np.abs(x0)

        # --- dynamic thresholding scale (per-sample) ---
        # Bごとにしきい値（高パーセンタイル）を取る
        s = np.quantile(ax.reshape(ax.shape[0], -1), p, axis=1)
        s = np.maximum(s, clip_val)  # 少なくとも clip_val は確保
        s = s.reshape(-1, 1, 1, 1)

        # scale to [-clip_val, +clip_val] domain
        x0_scaled = x0 / (s + eps) * clip_val

        # pre-stats (after scaling, before clipping)
        mu_pre = x0_scaled.mean(axis=(0, 2, 3))
        std_pre = x0_scaled.std(axis=(0, 2, 3))
        clip_rate = (np.abs(x0_scaled) > clip_val).mean(axis=(0, 2, 3))

        # clip
        x0_clip = np.clip(x0_scaled, -clip_val, clip_val)

        # optional: mean-preserving correction (per-channel, global over BHW)
        if preserve_mean:
            #print('preserve_mean')
            # target mean is mu_pre (after scaling)
            target = mu_pre
            y = x0_clip
            it = int(max(1, preserve_mean_iter))
            for _ in range(it):
                mu_post_now = y.mean(axis=(0, 2, 3))
                delta = (target - mu_post_now).reshape(1, -1, 1, 1)
                y = np.clip(y + delta, -clip_val, clip_val)
            x0_clip = y

        mu_post = x0_clip.mean(axis=(0, 2, 3))

        if return_stats:
            stats = {
                "mu_pre_RGB": mu_pre,      # shape (C,)
                "mu_post_RGB": mu_post,    # shape (C,)
                "std_pre_RGB": std_pre,    # shape (C,)
                "clip_rate_RGB": clip_rate # shape (C,)
            }
            return x0_clip, s.reshape(-1), stats

        return x0_clip, s.reshape(-1)


    def dynamic_thresholding_bkup2(self, x0, p=0.995, clip_val=1.0, eps=1e-6,
        preserve_mean=True, preserve_mean_iter=1):
        """
        Dynamic thresholding (per-sample scaling + clip) with optional mean-preserving correction.

        Parameters
        ----------
        x0 : ndarray (B,C,H,W)
            Predicted x0 before clipping.
        p : float
            Quantile for dynamic thresholding (e.g. 0.995).
        clip_val : float
            Clip range becomes [-clip_val, +clip_val].
        eps : float
            Numerical stability.
        preserve_mean : bool
            If True, apply a post-clip mean correction per-channel so that
            mean(x0_after) ≈ mean(x0_before) (measured after scaling).
        preserve_mean_iter : int
            Number of correction iterations (1 is usually enough).
        return_stats : bool
            If True, returns a dict of stats for debugging.

        Returns
        -------
        x0_clip : ndarray (B,C,H,W)
        s : ndarray (B,)
            Per-sample scale (>= clip_val).
        stats : dict (optional)
            Contains mu_pre_RGB, mu_post_RGB, std_pre_RGB, clip_rate_RGB, etc.
        """
        # x0: (B,C,H,W)
        ax = np.abs(x0)

        # --- dynamic thresholding scale (per-sample) ---
        # Bごとにしきい値（高パーセンタイル）を取る
        s = np.quantile(ax.reshape(ax.shape[0], -1), p, axis=1)
        s = np.maximum(s, clip_val)  # 少なくとも clip_val は確保
        s = s.reshape(-1, 1, 1, 1)

        # scale to [-clip_val, +clip_val] domain
        x0_scaled = x0 / (s + eps) * clip_val

        # pre-stats (after scaling, before clipping)
        x0_clip = np.clip(x0_scaled, -clip_val, clip_val)

        # optional: mean-preserving correction (per-channel, global over BHW)
        if preserve_mean:
            mu_pre  = x0.mean(axis=(0, 2, 3), keepdims=True)
            delta = mu_pre - x0_clip.mean(axis=(0, 2, 3), keepdims=True)
            x0_clip = np.clip(x0_scaled + delta, -clip_val, clip_val)

        return x0_clip, s.reshape(-1)


    def dynamic_thresholding(self, x0, p=0.995, clip_val=1.0, eps=1e-6,
        preserve_mean=True, preserve_mean_iter=1, per_sample_mean=True):

        B, C, H, W = x0.shape
        axis = (2,3) if per_sample_mean else (0,2,3) # 維持するのはバッチ内か個々のサンプルか
        x0_mean = x0.mean(axis=axis, keepdims=True)  # 元の平均(これに合わせに行く)

        # 小さい順に並べたときのpで指定した割合の位置にある値(外れ値とみなす値)
        s = np.quantile(np.abs(x0).reshape(B, -1), p, axis=1)
        s = np.maximum(s, clip_val).reshape(B, 1, 1, 1)

        # 元空間で clip
        x0_clip = np.clip(x0, -s, s) # 外れ値をクリップ

        if preserve_mean:
            it = int(max(1, preserve_mean_iter))
            target = x0.mean(axis=axis, keepdims=True)   
            for _ in range(it):
                mu = x0_clip.mean(axis=axis, keepdims=True) # clip後の平均 
                x0_clip = np.clip(x0_clip + (x0_mean - mu), -s, s)
        # この段階でx0_clip.mean(axis=axis)≒x0.mean(axis=axis)

        # 正規化空間へスケーリング
        x0_clip *= (clip_val / (s + eps))
        return x0_clip, s.reshape(-1)

    def remove_dc(self, x, strength=1.0, axis=None):
        """ 
        DC成分の除去（サンプル時の安定化用）

        axis:
          None    "Batch_luma"   : 全体の明るさ補正 
          (1,2,3) "luma"         : 各サンプル明るさ補正、色相は崩れにくい
          (2,3)   "per_channel"  : 各サンプルの色成分毎の補正
          (0,2,3) "batch_channel": バッチ色かぶり補正
        """
        x -= strength * x.mean(axis=axis, keepdims=True)
        return x

    def remove_dc_luma_chroma(self, x, dc_kappa=1.0,
                              k_luma=None, k_chroma=0.0,
                              by_alpha_bar=True, alpha_bar=None):
        """
        明暗（luma）と色（chroma）の両方を抑える
        x: (B,C,H,W)
        luma  = mean over (B,C,H,W)  -> (1,1,1,1)  (全チャネル共通オフセット)
        chroma= mean over (B,H,W) per channel - luma -> (1,C,1,1) (チャネル間の偏り)
        """
        if k_luma is None:
            k_luma = dc_kappa
        kL = float(k_luma)
        kC = float(k_chroma)

        if by_alpha_bar and kC != 0.0:
            kC = float(kC * alpha_bar) # 低t(終盤)ほど効く

        #mu_ch = x.mean(axis=(2,3), keepdims=True)  # 色成分毎のサンプル毎平均
        mu_ch = x.mean(axis=(0,2,3), keepdims=True) # 色成分毎のバッチ内平均 
        luma  = mu_ch.mean(axis=1, keepdims=True)   # 色成分全平均=明るさ
        chroma = mu_ch - luma                       # (1,C,1,1)
        if kL != 0.0:
            x = x - kL * luma
        if kC != 0.0:
            x = x - kC * chroma
        return x, luma, chroma


    def target_guide(self, x0_hat, alpha_bar, x0_target=None, guide=0.0):
        """
        x0_target ガイド（morphing 用）
        x0_hat を “狙った中間顔” に少し引っ張る

        """
        if x0_target is None or guide is None or guide <= 0.0:
            return x0_hat
        # x0_target を x0_hat と同shapeに揃える（(C,H,W) -> (B,C,H,W) など）
        tgt = x0_target
        if tgt.ndim == x0_hat.ndim - 1:
            tgt = tgt.reshape((1,) + tgt.shape)
        if tgt.shape[0] == 1 and x0_hat.shape[0] > 1:
            tgt = np.repeat(tgt, x0_hat.shape[0], axis=0)
        # 線形ブレンド（guide=0.05〜0.2 推奨）
        g = guide * alpha_bar
        #g = guide if int(t) <= 20 else 0.0

        x0_hat = (1.0 - g) * x0_hat + g * tgt
        return x0_hat

    def denoise_ddpm(self, model, x, t, t_prev, **kwargs):
        """ DDPM posterior mean/var """
        eta           = kwargs.pop('eta', 1.0) # eta=1が基本　
        labels        = kwargs.pop('labels', None)  
        gamma         = kwargs.pop('gamma', None)
        clip_denoised = kwargs.pop('clip_denoised' , False)
        dc_removal    = kwargs.pop('dc_removal' , False)
        dt_p          = kwargs.pop('dt_p' , 0.995)
        denom_floor   = kwargs.pop('denom_floor' , 1e-4)
        x0_target     = kwargs.pop('x0_target' , None)
        guide         = kwargs.pop('guide' , 0.0)
        debug         = kwargs.pop('debug', False)

        # eta: noise scale. eta=1.0 -> standard DDPM,
        #                   eta=0.0 -> "mean-only" (tends to average/whiten)
        #print('eta =', eta)
        # サンプリングでは t はスカラ（または全要素同一の (N,)）とする
        t = self.fix_t(t, 1)
        t_prev = self.fix_t(t_prev, 0)
        
        # 係数はスカラのまま（broadcast は演算時に自動で効く）
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev]

        if gamma is not None:
            eps = model(x, t, labels)
            eps_uncond = model(x, t)
            eps = eps_uncond + gamma * (eps - eps_uncond)
        else:
            eps = model(x, t, labels)

        #print('eps_hat.std', np.std(eps))
        if dc_removal:
            eps = self.remove_dc(eps, dc_removal)

        # x0 推定
        x0_hat = (x - np.sqrt(1 - alpha_bar) * eps) / np.sqrt(alpha_bar)
        
        if debug:
            x0_pre = x0_hat.copy() # 差分ログ用
            
        # x0_hatをクリップするとともにepsをそれに合わせて再計算
        if clip_denoised:
            x0_hat, s = self.dynamic_thresholding(x0_hat, p=dt_p, clip_val=1.0)

        if x0_target is not None and guide is not None and guide > 0.0:
            x0_hat = self.target_guide(x0_hat, alpha_bar, x0_target, guide)
            # ガイド後は eps と整合させ直すと安定

        denom = np.sqrt(np.maximum(1.0 - alpha_bar, denom_floor))
        eps = (x - np.sqrt(alpha_bar) * x0_hat) / denom

        # posterior mean / std
        coef1 = (np.sqrt(alpha_bar_prev) * (1 - alpha)) / (1 - alpha_bar)
        coef2 = (np.sqrt(alpha) * (1 - alpha_bar_prev)) / (1 - alpha_bar)
        mu = coef1 * x0_hat + coef2 * x
        std = np.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
       
        if int(t_prev) == 0:
            x_prev = mu # 終端はノイズを入れない　
        else:
            noise = np.random.randn(*x.shape).astype(x.dtype)
            x_prev = mu + eta * std * noise

        # ログ情報収集
        if debug:
            self.append_log(t, x, x_prev, x0_pre, x0_hat, eps)
        return x_prev

    def denoise_ddim(self, model, x, t, t_prev, **kwargs):
        """ ddim(Denoising Diffusion Implicit Model) """
        eta           = kwargs.pop('eta', 0.0) # eta=0が基本
        labels        = kwargs.pop('labels', None)  
        gamma         = kwargs.pop('gamma', None)
        clip_denoised = kwargs.pop('clip_denoised' , False)
        dc_removal    = kwargs.pop('dc_removal' , False)
        dt_p          = kwargs.pop('dt_p' , 0.995)
        denom_floor   = kwargs.pop('denom_floor' , 1e-4)
        debug         = kwargs.pop('debug', False)

        t = self.fix_t(t, 1)
        t_prev = self.fix_t(t_prev, 0)

        alpha_bar  = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev]
        
        if gamma is not None:
            eps = model(x, t, labels)
            eps_uncond = model(x, t)
            eps = eps_uncond + gamma * (eps - eps_uncond)
        else:
            eps = model(x, t, labels)

        #eps -= model.out.parameters.b.reshape(1,3,1,1)

        if dc_removal:
            eps = self.remove_dc(eps, dc_removal)

        # x0 推定
        x0_hat = (x - np.sqrt(1.0 - alpha_bar) * eps) / np.sqrt(alpha_bar)

        if debug:
            x0_pre = x0_hat.copy() # 差分ログ用

        # x0_hatをクリップするとともにepsをそれに合わせて再計算
        if clip_denoised: 
            x0_hat, s = self.dynamic_thresholding(x0_hat, p=dt_p, clip_val=1.0)
            denom = np.sqrt(max(1.0 - alpha_bar, denom_floor))
            eps = (x - np.sqrt(alpha_bar) * x0_hat) / denom

        # eta=0（決定的）なら sigma=0 でOK
        if eta == 0.0:
            sigma = 0.0
            dir_coef = np.sqrt(max(1.0 - alpha_bar_prev, 0.0))  # この形が一番安定
        else:
            sigma = (eta * np.sqrt((1.0 - alpha_bar_prev) / max(1.0 - alpha_bar, 1e-8))
                        * np.sqrt(max(1.0 - alpha_bar / alpha_bar_prev, 0.0)))
            dir_coef = np.sqrt(max(1.0 - alpha_bar_prev - sigma * sigma, 0.0))

        # ログ情報収集
        if debug:
            self.debug_info = {'x0_pre' : x0_pre, 'x0_hat' : x0_hat, 'eps' : eps} 

        # 返り値
        if int(t_prev) == 0:
            x_prev = x0_hat # 終端は x0 を返すのが安定（eta=0なら特に）
        else:
            noise = np.random.randn(*x.shape).astype(x.dtype)
            x_prev = np.sqrt(alpha_bar_prev) * x0_hat + dir_coef * eps + sigma * noise

        # ログ情報収集
        if debug:
            self.append_log(t, x, x_prev, x0_pre, x0_hat, eps)
        return x_prev

    def sample(self, model, x_shape=(20, 1, 28, 28), x=None, sampler=None,
               steps=None, start=None, halt=None, debug=False, **kwargs):

        if x is None:
            x = np.random.randn(*x_shape).astype(Config.dtype)

        ts, t_prev_list = self.schedule_time_steps(steps=steps)

        if sampler is None:
            denoise_fn = self.denoise
        elif sampler == "ddpm":
            denoise_fn = self.denoise_ddpm
        elif sampler == "ddim":
            denoise_fn = self.denoise_ddim
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        self.log = [] # ログ情報を空に　
        for i, (t, t_prev) in enumerate(zip(ts, t_prev_list)):
            if start is not None and t > start:
                continue
            
            x = denoise_fn(model, x, t, t_prev, debug=debug, **kwargs)

            if halt is not None and t<=(self.num_timesteps-halt):
                break
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
        l = lf.MeanSquaredError()(eps_hat, eps) 
        #(eps_hat - eps)**2
        # 時刻tはstep_log,weightingの両方に使う
        if self.step_log and t is not None: # 時刻毎のエラー集計
            t = self.fix_t(t, 0, None)
            np.add.at(self.stat_sum, t, ((eps_hat-eps)**2).mean(axis=(1,2,3))) 
            np.add.at(self.stat_cnt, t, 1)
        if self.weighting and t is not None:     # 時刻に応じた重付け
            # gammaが小さい：減衰はゆるい。高 SNR もそこそこ学習させたいとき。
            # gammaが大きい：高 SNR の損失が一気に軽くなる。終盤の復元（低 SNR）を重視
            t = self.fix_t(t, 0, eps.ndim)
            alpha_bar = self.alpha_bars[t]
            snr = alpha_bar / (1 - alpha_bar) # 信号雑音比
            w = (snr + 1)**(-gamma)
            l = l * w
            #print(l.shape, w.shape)
        if dc_reg:    
            dc = F.Mean(axis=(2,3), keepdims=True)(eps_hat) # B,Cごと
            dc_l = F.Mean()(dc**2)  
            l = l + lam * dc_l
        return l

    def ddim_inversion(self, model, x0,
                       tend=500,
                       steps=None,
                       clip_denoised=False,
                       dynamic_thresholding=False,
                       dt_p=0.995,
                       ):
        """DDIM inversion: x0(=t=0) -> x_tend を生成（eta=0前提）"""

        # schedule_time_steps は降順 (大きいt -> 小さいt)
        ts_desc, tprev_desc = self.schedule_time_steps(steps=steps)
        ts_asc = list(reversed(ts_desc))
        tprev_asc = list(reversed(tprev_desc))

        x = x0
        for t_prev, t in zip(tprev_asc, ts_asc):
            if int(t) > int(tend):
                break

            t_prev_i = self.fix_t(t_prev, 0, None)
            t_i      = self.fix_t(t,      0, None)

            alpha_bar_prev = self.alpha_bars[t_prev_i]
            alpha_bar_t    = self.alpha_bars[t_i]

            # eps 予測（x_{t_prev}）
            eps_hat = model(x, t_prev_i)

            # x0 推定
            x0_hat = (x - np.sqrt(1.0 - alpha_bar_prev) * eps_hat) / np.sqrt(alpha_bar_prev)
            if clip_denoised:
                x0_hat = self.clip_denoised(x0_hat,
                                            dynamic_thresholding=dynamic_thresholding,
                                            p=dt_p)

            # 決定的 DDIM forward: x_t
            x = np.sqrt(alpha_bar_t) * x0_hat + np.sqrt(1.0 - alpha_bar_t) * eps_hat

        return x

    def append_log(self, t, x, x_prev, x0_pre, x0_hat, eps, 
                   spatial_axes=(2,3), channel_axes=(0,2,3)):
        import numpy
        mu_xt       = x.mean(axis=spatial_axes)
        mu_x_prev   = x_prev.mean(axis=spatial_axes)
        mu_x0_pre   = x0_pre.mean(axis=spatial_axes)
        mu_x0_hat   = x0_hat.mean(axis=spatial_axes)
        std_xt      = x.std(axis=spatial_axes)
        std_eps     = eps.std(axis=spatial_axes)

        mu_x_prev_c = x_prev.mean(axis=channel_axes) 
        mu_pre_c    = x0_pre.mean(axis=channel_axes)
        mu_post_c   = x0_hat.mean(axis=channel_axes)
        std_pre_c   = x0_pre.std(axis=channel_axes)
        std_post_c  = x0_hat.std(axis=channel_axes)

        delta_mu_c  = mu_post_c - mu_pre_c
        clip_rate_c = (np.abs(x0_pre) > 1.0).mean(axis=channel_axes)

        mu_eps = eps.mean(axis=spatial_axes)

        self.log.append({
            't'          : int(t),
            'mu_xt'      : mu_xt,
            'mu_x_prev'  : mu_x_prev,
            'mu_x0_pre'  : mu_x0_pre,
            'mu_x0_hat'  : mu_x0_hat,
            'std_xt'     : std_xt,
            'std_eps'    : std_eps,
            'mu_pre_c'   : mu_pre_c,
            'mu_post_c'  : mu_post_c,
            'std_pre_c'  : std_pre_c,
            'std_post_c' : std_post_c,
            'delta_mu_c' : delta_mu_c,
            'clip_rate_c': clip_rate_c,
            'mu_eps'     : mu_eps,
        })
