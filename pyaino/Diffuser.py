# Diffuser
# 20260109 A.Inoue

from pyaino.Config import *
from pyaino import Functions as F

class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 step_log=False, weighting=False):
        self.num_timesteps = num_timesteps
        #self.betas = np.zeros(num_timesteps+1, dtype=Config.dtype)
        #self.betas[1:] = np.linspace(beta_start, beta_end, num_timesteps,
        #                             dtype=Config.dtype)
        self.betas = np.zeros(num_timesteps, dtype=Config.dtype)
        self.betas[1:] = np.linspace(beta_start, beta_end, num_timesteps-1,
                                     dtype=Config.dtype)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas, axis=0)
        self.step_log = step_log
        self.weighting = weighting
        if step_log: # 時刻毎のエラー記録
            self.stat_sum = np.zeros(self.num_timesteps, dtype=float)
            self.stat_cnt = np.zeros(self.num_timesteps, dtype=np.int32)

    def fix_t(self, t, min_t=0, ndim=None):  # ndimはブロードキャストが必要な場合のみ
        T = self.num_timesteps
        t_arr = np.asarray(t, dtype=np.int64)
        assert (t_arr >= min_t).all() and (t_arr < T).all()
        if ndim is None:
            return t_arr
        return t_arr.reshape((-1,) + (1,) * (ndim - 1))

    def add_noise(self, x_0, t):
        t = self.fix_t(t, 0, x_0.ndim) # x_0 に次元を合わせる
        alpha_bar = self.alpha_bars[t]
        noise = np.random.randn(*x_0.shape).astype(x_0.dtype)
        x_t = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t):
        """ 基本のDDPM """
        t = self.fix_t(t, 0)

        alpha     = self.alphas[t]      
        alpha_bar = self.alpha_bars[t]  
        alpha_bar_prev = self.alpha_bars[t - 1] if t >=1 else 1.0 

        eps = model(x, t)  

        noise = np.random.randn(*x.shape).astype(x.dtype)
        noise[t==0] = 0 # t=0 の場合は全部0でノイズ無し

        mu  = (x - ((1 - alpha) / np.sqrt(1 - alpha_bar)) * eps) / np.sqrt(alpha)
        std = np.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        return mu + noise * std

    def dynamic_thresholding(self, x0, p=0.995, clip_val=1.0, eps=1e-6):
        # x0: (B,C,H,W)
        ax = np.abs(x0)
        # Bごとにしきい値（高パーセンタイル）を取る
        s = np.quantile(ax.reshape(ax.shape[0], -1), p, axis=1).reshape(-1,1,1,1)
        s = np.maximum(s, clip_val)          # 1未満にならないように
        x0 = x0 / (s + eps) * clip_val       # スケールしてから
        x0 = np.clip(x0, -clip_val, clip_val)
        return x0, s

    def denoise_ddpm(self, model, x, t, eta=1.0, clip_denoised=False,
                     dt_p=0.995, denom_floor=1e-4):
        """ DDPM posterior mean/var """
        # eta: noise scale. eta=1.0 -> standard DDPM,
        #                   eta=0.0 -> "mean-only" (tends to average/whiten)
        #print('eta =', eta)
        # サンプリングでは t はスカラ（または全要素同一の (N,)）とする
        t = self.fix_t(t, 0)
        
        # 係数はスカラのまま（broadcast は演算時に自動で効く）
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t - 1] if t >= 1 else 1.0

        eps = model(x, t)
        #print('eps_hat.std', np.std(eps))

        #print(t,
        #      "eps_hat batch-std(mean)=", float(np.std(eps, axis=0).mean()),
        #      "x batch-std(mean)=", float(np.std(x, axis=0).mean()))

        # x0_pre（clip前）
        x0_pre = (x - np.sqrt(1 - alpha_bar) * eps) / np.sqrt(alpha_bar)
        oor = (((x0_pre < -1.0) | (x0_pre > 1.0))).astype(np.float32).mean()

        # x0_hat（clip後）
        if clip_denoised:
            x0_hat, s = self.dynamic_thresholding(x0_pre, p=dt_p, clip_val=1.0)
            denom = np.sqrt(max(1.0 - alpha_bar, denom_floor))
            eps = (x - np.sqrt(alpha_bar) * x0_hat) / denom
        else:
            x0_hat = x0_pre

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
        if t == 0:
            noise[:] = 0  # t=0 は deterministic

        return mu + eta * std * noise

    def _make_ddim_schedule(self, steps: int):
        """ T= self.num_timesteps を steps 個に間引いた t 列と t_prev 列を返す """
        T = self.num_timesteps
        steps = int(steps)
        if steps < 2:
            raise ValueError("steps must be >= 2")

        ts = np.linspace(1, T-1, steps, dtype=np.int64)
        ts = np.unique(ts)          # 重複除去（stepsが大きいと重複し得る）
        ts = ts[::-1]               # 降順: T -> ... -> 1

        t_prev = np.concatenate([ts[1:], np.array([0], dtype=np.int64)])
        return ts, t_prev


    def denoise_ddim(self, model, x, t, t_prev, eta=0.0, clip_denoised=False,
                    dt_p=0.995, denom_floor=1e-4):
        t = int(self.fix_t(t, 0))
        t_prev = int(self.fix_t(t_prev, 0))

        a  = float(self.alpha_bars[t])
        ap = float(self.alpha_bars[t_prev])  # t_prev=0でも alpha_bars[0]=1 なのでこれでOK

        eps = model(x, t)

        # x0 推定
        x0_hat = (x - np.sqrt(1.0 - a) * eps) / np.sqrt(a)

        if clip_denoised:
            # hard clip ではなく dynamic thresholding 推奨
            x0_hat, s = self.dynamic_thresholding(x0_hat, p=dt_p, clip_val=1.0)

            # ★整合 eps 再計算（分母が小さい領域の数値事故を避ける）
            denom = np.sqrt(max(1.0 - a, denom_floor))
            eps = (x - np.sqrt(a) * x0_hat) / denom

            # ログ！
            #print(t, "s med/mean/max",
            #      float(np.median(s)), float(np.mean(s)), float(np.max(s)),
            #      "max|x0_hat|", float(np.max(np.abs(x0_hat))),
            #      "max|x|", float(np.max(np.abs(x))))

        # eta=0（決定的）なら sigma=0 でOK
        if eta == 0.0:
            sigma = 0.0
            dir_coef = np.sqrt(max(1.0 - ap, 0.0))  # この形が一番安定
        else:
            sigma = (eta * np.sqrt((1.0 - ap) / max(1.0 - a, 1e-8))
                        * np.sqrt(max(1.0 - a / ap, 0.0)))
            dir_coef = np.sqrt(max(1.0 - ap - sigma * sigma, 0.0))

        # 終端は x0 を返すのが安定（eta=0なら特に）
        if t_prev == 0:
            return x0_hat

        noise = np.random.randn(*x.shape).astype(x.dtype)
        x_prev = np.sqrt(ap) * x0_hat + dir_coef * eps + sigma * noise
        return x_prev

    def sample(self, model, x_shape=(20, 1, 28, 28),
               sampler=None, eta=1.0, steps=None, clip_denoised=False,
               halt=None):
        batch_size = x_shape[0]
        x = np.random.randn(*x_shape).astype(Config.dtype)

        if sampler is None:
            for t in range(self.num_timesteps-1, 0, -1):
                x = self.denoise(model, x, t)
                #print(f'{t:4d} / {self.num_timesteps}'
                #      + f'{np.min(x):7.3f} {np.max(x):7.3f}')
                if halt is not None and t==(self.num_timesteps-halt):
                    break
            return x    

        if sampler == "ddpm":
            for t in range(self.num_timesteps-1, 0, -1):
                x = self.denoise_ddpm(model, x, t,
                                      eta=eta, clip_denoised=clip_denoised)
                if halt is not None and t==(self.num_timesteps-halt):
                    break
            return x

        elif sampler == "ddim":
            if steps is None: # 間引き無し
                for t in range(self.num_timesteps-1, 0, -1):
                    x = self.denoise_ddim(model, x, t=t, t_prev=t-1,
                                          eta=eta, clip_denoised=clip_denoised)
                    if halt is not None and t==(self.num_timesteps-halt):
                        break
                return x

            # 間引き
            ts, t_prev_list = self._make_ddim_schedule(steps=steps)

            metrics_log = []
            for i, (t, t_prev) in enumerate(zip(ts, t_prev_list)):
                x = self.denoise_ddim(model, x, t=int(t), t_prev=int(t_prev),
                                          eta=eta, clip_denoised=clip_denoised)
                # ここ！
                #mean_std, max_abs = batch_metrics(x)
                #metrics_log.append((i, t, mean_std, max_abs))
                # 必要なら間引き表示（例：10ステップごと）
                #if (i % 10) == 0 or (t_prev == 0):
                #    print(f"[i={i:4d}] t={t:4d}  mean_std_over_batch={mean_std:.6f}  max|x|={max_abs:.6f}")
                if halt is not None and t<=(self.num_timesteps-halt):
                    print(__class__.__name__, 'halt at', t)
                    break
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

    def loss(self, eps_pred, eps, t=None, gamma=1.0):
        """ 時刻に応じたエラー集計と時刻に応じた重み付け可能な平均2乗誤差 """
        l = (eps_pred - eps)**2
        # 時刻tはstep_log,weightingの両方に使う
        if self.step_log and t is not None: # 時刻毎のエラー集計
            t = self.fix_t(t, 0, None)
            np.add.at(self.stat_sum, t, l.mean(axis=(1,2,3)))
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
        return F.Mean()(l)

    def denoise_ddim_bkup(self, model, x, t, t_prev, eta=0.0, clip_denoised=False):
        """ DDIM """
        t = self.fix_t(t, 0)
        t_prev = self.fix_t(t_prev, 0)

        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev] if t_prev > 0 else 1.0

        eps = model(x, t)

        # x0 推定
        x0_hat = (x - np.sqrt(1.0 - alpha_bar) * eps) / np.sqrt(alpha_bar)
        if clip_denoised:
            x0_hat = np.clip(x0_hat, -1.0, 1.0)

        if int(t_prev) == 0:
            print('###quit', t_prev)
            return x0_hat
    
        # DDIM の係数
        sigma = (eta * np.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
                     * np.sqrt(1.0 - alpha_bar / alpha_bar_prev))
        dir_coef = np.sqrt(np.maximum(1.0 - alpha_bar_prev - sigma * sigma, 0.0))

        noise = np.random.randn(*x.shape).astype(x.dtype)
        if t == 0:
            noise[:] = 0

        x_prev = np.sqrt(alpha_bar_prev) * x0_hat + dir_coef * eps + sigma * noise
        return x_prev


    def denoise_ddim_bkup(self, model, x, t, t_prev, eta=0.0,
                     clip_denoised=True, clip_only_when_alpha_bar_gt=0.1):
        t = int(t)
        t_prev = int(t_prev)

        a  = float(self.alpha_bars[t])
        ap = float(self.alpha_bars[t_prev]) if t_prev > 0 else 1.0

        eps = model(x, t)

        # x0推定
        x0 = (x - np.sqrt(1.0 - a) * eps) / np.sqrt(a)

        if clip_denoised:
        #if clip_denoised and (a > clip_only_when_alpha_bar_gt):    
            # hard clip ではなく dynamic thresholding
            x0, s = self.dynamic_thresholding(x0, p=0.995, clip_val=1.0)

            # 整合 eps 再計算（分母が小さい領域は事故るので太らせる）
            denom = np.sqrt(max(1.0 - a, 1e-4))   # ★ここが効きます（1e-5〜1e-3で調整）
            eps = (x - np.sqrt(a) * x0) / denom

            print(t, "s med/mean/max",
                  float(np.median(s)), float(np.mean(s)), float(np.max(s)),
                  "max|x0|", float(np.max(np.abs(x0))),
                  "max|x|", float(np.max(np.abs(x))))
       
        # DDIM（eta=0ならsigma=0で決定的）
        if eta == 0.0:
            sigma = 0.0
        else:
            sigma = (eta * np.sqrt((1.0 - ap) / max(1.0 - a, 1e-8))
                         * np.sqrt(max(1.0 - a / ap, 0.0)))

        dir_coef = np.sqrt(max(1.0 - ap - sigma * sigma, 0.0))

        # ここで終端は x0 を返すのが一番安定
        if t_prev == 0:
            return x0

        noise = np.random.randn(*x.shape).astype(x.dtype)
        x_prev = np.sqrt(ap) * x0 + dir_coef * eps + sigma * noise
        return x_prev


'''
def batch_metrics(x: np.ndarray) -> tuple[float, float]:
    """
    x: (B,C,H,W)
    returns:
      mean_std_over_batch: float  # 画素ごとにバッチ軸のstdを取り、それを平均
      max_abs_x: float            # |x| の最大値
    """
    mean_std_over_batch = float(np.std(x, axis=0).mean())
    max_abs_x = float(np.max(np.abs(x)))
    return mean_std_over_batch, max_abs_x
'''
