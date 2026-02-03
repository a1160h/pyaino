# Diffuser
# 20260131 A.Inoue

from pyaino.Config import *
from pyaino import Functions as F
from pyaino import LossFunctions as lf

class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 step_log=False, weighting=False):
        betas = np.zeros(num_timesteps, dtype=Config.dtype)
        betas[1:] = np.linspace(beta_start, beta_end, num_timesteps-1, dtype=Config.dtype)
        self.alphas = 1 - betas
        self.alpha_bars = np.cumprod(self.alphas, axis=0)
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
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

        noise = np.random.randn(*x.shape).astype(x.dtype)
        if int(t_prev) == 0:
            noise[:] = 0

        mu  = (x - ((1 - alpha) / np.sqrt(1 - alpha_bar)) * eps) / np.sqrt(alpha)
        std = np.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        return mu + eta * noise * std

    def dynamic_thresholding(self, x0, p=0.995, clip_val=1.0, eps=1e-6):
        # x0: (B,C,H,W)
        ax = np.abs(x0)
        # Bごとにしきい値（高パーセンタイル）を取る
        s = np.quantile(ax.reshape(ax.shape[0], -1), p, axis=1).reshape(-1,1,1,1)
        s = np.maximum(s, clip_val)          # 1未満にならないように
        x0 = x0 / (s + eps) * clip_val       # スケールしてから
        x0 = np.clip(x0, -clip_val, clip_val)
        return x0, s

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
            eps -= eps.mean(axis=(-2, -1), keepdims=True) # DC除去

        # x0 推定
        x0_hat = (x - np.sqrt(1 - alpha_bar) * eps) / np.sqrt(alpha_bar)
        
        if debug:
            x0_pre = x0_hat.copy() # 差分ログ用
            
        # x0_hatをクリップするとともにepsをそれに合わせて再計算
        if clip_denoised:
            x0_hat, s = self.dynamic_thresholding(x0_hat, p=dt_p, clip_val=1.0)
            denom = np.sqrt(max(1.0 - alpha_bar, denom_floor))
            eps = (x - np.sqrt(alpha_bar) * x0_hat) / denom

        # 以下、仮実装20260116AI
        # ---------------------------------------
        # x0_target ガイド（morphing 用）
        # x0_hat を “狙った中間顔” に少し引っ張る
        # ---------------------------------------
        if x0_target is not None and guide is not None and guide > 0.0:
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

            # ガイド後は eps と整合させ直すと安定
            denom = np.sqrt(np.maximum(1.0 - alpha_bar, denom_floor))
            eps = (x - np.sqrt(alpha_bar) * x0_hat) / denom
        # ここまで

        # posterior mean / std
        coef1 = (np.sqrt(alpha_bar_prev) * (1 - alpha)) / (1 - alpha_bar)
        coef2 = (np.sqrt(alpha) * (1 - alpha_bar_prev)) / (1 - alpha_bar)
        mu = coef1 * x0_hat + coef2 * x
        std = np.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))

        # ログ情報収集
        if debug:
            self.debug_info = {'x0_pre' : x0_pre, 'x0_hat' : x0_hat, 'eps' : eps} 

        noise = np.random.randn(*x.shape).astype(x.dtype)
        if int(t_prev) == 0:
            noise[:] = 0

        return mu + eta * std * noise

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
            eps -= eps.mean(axis=(-2, -1), keepdims=True) # DC除去

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

        # 終端は x0 を返すのが安定（eta=0なら特に）
        if int(t_prev) == 0:
            return x0_hat

        noise = np.random.randn(*x.shape).astype(x.dtype)
        x_prev = np.sqrt(alpha_bar_prev) * x0_hat + dir_coef * eps + sigma * noise
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

        if debug:
            self.log = []
            self.debug_info = None

        for i, (t, t_prev) in enumerate(zip(ts, t_prev_list)):
            if start is not None and t > start:
                continue
            
            x = denoise_fn(model, x, t, t_prev, debug=debug, **kwargs)

            if debug: # ログの収集
                mu_xt = x.mean(axis=(2,3), keepdims=True)
                mu_x0_pre = self.debug_info['x0_pre'].mean(axis=(2,3), keepdims=True)
                mu_x0_hat = self.debug_info['x0_hat'].mean(axis=(2,3), keepdims=True)
                std_xt = x.std(axis=(2,3), keepdims=True)
                std_eps = self.debug_info['eps'].std(axis=(2,3), keepdims=True)
                self.log.append({'t': t, 't_prev' : t_prev, 'mu_xt': mu_xt,
                                 'mu_x0_pre': mu_x0_pre, 'mu_x0_hat': mu_x0_hat,
                                 'std_xt' : std_xt, 'std_eps' : std_eps
                                 })
                
            if halt is not None and t<=(self.num_timesteps-halt):
                break
        return x    
 
    def logs_garbage(): # デバグ用のゴミを集めたもの（後日削除）　 
        oor = (((x0_hat < -1.0) | (x0_hat > 1.0))).astype(np.float32).mean()

        g_t = np.sqrt((1-alpha_bar)/alpha_bar)

        log['mu_x0_pre'] = x0_pre.mean()
        log['mu_x0_hat'] = x0_hat.mean()
        log['std_eps'] = eps.std()  
        print(f't = {t}',
              f'mu_xt = {x.mean():7.4f}',
              f'g_t * mu_eps = {g_t * eps.mean(axis=(2,3)).mean():7.4f}',
              f'mu_x0 = {x0_hat.mean():7.4f}')

        
        print("t", int(t), "x0_pre min/max", x0_pre.min(), x0_pre.max(), "oor", oor)
        print("t", int(t), "x0_hat min/max", x0_hat.min(), x0_hat.max())
        print("t", int(t), "std mean/max", std.mean(), std.max())

        print(t,
              "eps_hat batch-std(mean)=", float(np.std(eps, axis=0).mean()),
              "x batch-std(mean)=", float(np.std(x, axis=0).mean()))

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

