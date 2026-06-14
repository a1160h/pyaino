# Diffuser
# 20260614 A.Inoue

from pyaino.Config import *
from pyaino import Functions as F
from pyaino import LossFunctions as lf
from pathlib import Path
import matplotlib.pyplot as plt
import copy
import re

from scipy.fftpack import dct, idct


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
        t = np.linspace(0, self.num_timesteps, self.num_timesteps+1, dtype=Config.dtype) \
            / self.num_timesteps  # 0～Tを0〜1 の連続時間に正規化
        alpha_bars = np.cos((t + self.s) / (1 + self.s) * np.pi / 2) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]  
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = np.clip(betas, 1e-8, 0.999) # 数値安定化
        alphas = 1.0 - betas                # β→α再計算
        alpha_bars = np.cumprod(alphas)     # 一貫性を保証するため再計算
        return betas, alphas, alpha_bars

    def linear_schedule(self):
        """ βの線形を保ちつつスケール調整しalpha_bar_T≈eps_finalを作る　"""
        # 1. まず「形」だけ線形に作る（0〜1の区間での線形）
        t = np.arange(self.num_timesteps, dtype=Config.dtype)
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
        self.kwargs = None # denoise時のオプションを覚えておく
        self.analizer = Analizer(self)
        self.jacobian_analizer = JacobianAnalizer(self)

    def fix_t(self, t, min_t=0, ndim=None):  # ndimはブロードキャストが必要な場合のみ
        """
        時刻 t を整数配列に整形、必要に応じブロードキャスト可能な形 (B,1,...,1) に変換
        """
        T = self.num_timesteps
        t_arr = np.asarray(t, dtype=np.int32)
        assert (t_arr >= min_t).all() and (t_arr < T).all()
        if ndim is None:
            return t_arr
        return t_arr.reshape((-1,) + (1,) * (ndim - 1))

    def schedule_time_steps(self, steps=None):
        """
        逆拡散反復に用いる時刻遷移列 (ts, t_prevs) を返す

        ts, t_prevs はともに長さ steps の整数配列であり、
        時刻インデックス区間 [T-1 → 0] を整数上で「ほぼ」等間隔に分割した遷移列
 
        """
        T = self.num_timesteps
        if steps is None or steps >= (T - 1):
            ts = np.arange(T - 1, 0, -1, dtype=np.int32)  # T-1, ..., 1
        else:
            idx = np.linspace(1, T - 1, int(steps))
            ts = np.rint(idx).astype(np.int32)
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

    def denoise(self, model, x, t, t_prev, labels=None, **kwargs):
        """ 基本のDDPM """
        eta           = kwargs.pop('eta', 1.0) # eta=1が基本　
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


    def dynamic_thresholding(self, x0, p=0.995, clip_val=1.0, eps=1e-6,
                             preserve_mean=True, lam=1.0,
                             axis=(2,3),
                             soft_clip=False):
        if soft_clip:
            clip = lambda x, s : s * np.tanh(x / s)
        else:    
            clip = lambda x, s : np.clip(x, -s, s)
        B, C, H, W = x0.shape

        # sは小さい順に並べたときのpで指定した割合の位置にある値(外れ値とみなす値)
        s = np.quantile(np.abs(x0), p, axis=axis, keepdims=True)
        s = np.maximum(s, clip_val)
        x0_clip = clip(x0, s)
        clip_rate = np.mean(x0!=x0_clip, axis=axis)
        if preserve_mean: # 元の平均を温存する
            x0_clip += lam * (x0.mean(axis=axis, keepdims=True)
                              - x0_clip.mean(axis=axis, keepdims=True))

        return x0_clip, s.reshape(-1), clip_rate

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

    def denoise_ddpm(self, model, x, t, t_prev, labels=None, debug=False, **kwargs):
        """ DDPM posterior mean/var """
        self.kwargs    = kwargs.copy() # 何を指定したかを覚えておく

        # 標準オプション standard options
        eta            = kwargs.pop('eta', 1.0)
        clip_denoised  = kwargs.pop('clip_denoised', False)
        dt_p           = kwargs.pop('dt_p', 0.995)
        denom_floor    = kwargs.pop('denom_floor', 1e-4)
        preserve_mean_beta = kwargs.pop('preserve_mean_beta', 0.0)
        sample_state   = kwargs['sample_state'] # 常に受け取るが使わない(DDIM用)

        # 誘導オプション guidance options
        gamma          = kwargs.pop('gamma', None)
        x0_target      = kwargs.pop('x0_target', None)
        guide          = kwargs.pop('guide', 0.0)

        # μ計算オプション mu options
        mu_mode        = kwargs.pop('mu_mode', 'eps')
        mu_blend       = kwargs.pop('mu_blend', 0.5)

        # 必要に応じて行う簡単な補正 optional simple corrections
        dc_axis        = kwargs.pop('dc_axis', (0, 2, 3))
        eps_dc_beta    = kwargs.pop('eps_dc_beta', 0.0)
        eps_recon_dc_beta  = kwargs.pop('eps_recon_dc_beta', 0.0)
        mu_dc_beta     = kwargs.pop('mu_dc_beta', 0.0)
        x0_dc_beta     = kwargs.pop('x0_dc_beta', 0.0)
        x_prev_dc_beta = kwargs.pop('x_prev_dc_beta', 0.0)

        # 一旦外す候補
        x_dc_beta      = kwargs.pop('x_dc_beta', 0.0)

        # eta: サンプリング時に加えるノイズのスケール
        #   eta = 1.0 -> DDPM と同じノイズ量
        #   eta = 0.0 -> 事後平均のみを辿る（ノイズなしの deterministic ステップ）

        # サンプリングでは t はスカラ（または全要素同一の (N,)）とする
        t = self.fix_t(t, 1)
        t_prev = self.fix_t(t_prev, 0)
        
        # 係数はスカラのまま（broadcast は演算時に自動で効く）
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev]

        if x_dc_beta > 0:
            x -= x_dc_beta * x.mean(axis=dc_axis, keepdims=True)

        """ step1 """
        # モデルによるeps予測
        if gamma is not None:
            eps = model(x, t, labels)
            eps_uncond = model(x, t)
            eps = eps_uncond + gamma * (eps - eps_uncond)
        else:
            eps = model(x, t, labels)

        # モデル出力のDC除去
        if eps_dc_beta > 0:
            eps -= eps_dc_beta * eps.mean(axis=dc_axis, keepdims=True)

        """ step2 """
        # x0 推定
        x0_hat = (x - np.sqrt(1 - alpha_bar) * eps) / np.sqrt(alpha_bar)
        
        if debug:
            x0_pre = x0_hat.copy() # 差分ログ用

        # x0_hatをクリップするとともにepsをそれに合わせて再計算
        if clip_denoised:
            x0_hat, s, clip_rate = self.dynamic_thresholding(x0_hat, p=dt_p, clip_val=1.0)
        else:
            s =0
            clip_rate = np.zeros(x0_hat.shape[:2], dtype=Config.dtype) # ログ用

        if x0_target is not None and guide is not None and guide > 0.0:
            x0_hat = self.target_guide(x0_hat, alpha_bar, x0_target, guide)
            # ガイド後は eps と整合させ直すと安定

        """ step3 """
        # q(x_prev | x_t, x_0) のガウス事後分布のパラメータ(平均と分散)
        # std : q(x_prev | x_t, x_0) の標準偏差（モデルには依存しない）
        # coef1, coef2 : 事後平均muを x0_hat, x から計算するための係数
        # mu = E[x_prev | x_t, x_0]
        std = np.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        coef1 = (np.sqrt(alpha_bar_prev) * (1 - alpha)) / (1 - alpha_bar)
        coef2 = (np.sqrt(alpha) * (1 - alpha_bar_prev)) / (1 - alpha_bar)

        if x0_dc_beta > 0:
            beta = x0_dc_beta * np.sqrt(1 - alpha_bar_prev)
            x0_hat -= beta * x0_hat.mean(axis=dc_axis, keepdims=True)

        denom = np.sqrt(np.maximum(1.0 - alpha_bar, denom_floor))
        eps_recon = (x - np.sqrt(alpha_bar) * x0_hat) / denom

        if eps_recon_dc_beta > 0:
            eps_recon -= eps_recon_dc_beta * eps_recon.mean(axis=dc_axis, keepdims=True)

        """ step4 """
        # mu 算出
        if mu_mode == 'x0':         # x0とxから算出
            mu = coef1 * x0_hat + coef2 * x
        elif mu_mode == 'eps':      # x0_hatと整合する eps_recon を使ってを算出
            mu = (x - ((1 - alpha) / denom) * eps_recon) / np.sqrt(alpha)
        elif mu_mode == 'eps_raw':  # モデル生の eps 出力から直接 mu を算出（x0_hat とは整合させない）
            mu = (x - ((1 - alpha) / denom) * eps) / np.sqrt(alpha)
        elif mu_mode == 'blend':    # 両者を混ぜる（安全に様子見したいとき)
            mu_x0 = coef1 * x0_hat + coef2 * x
            mu_eps = (x - ((1 - alpha) / denom) * eps_recon) / np.sqrt(alpha)
            mu = (1 - mu_blend) * mu_x0 + mu_blend * mu_eps
        elif mu_mode == 'blend_raw':# 両者を混ぜるがeps_rawを使う
            mu_x0 = coef1 * x0_hat + coef2 * x
            mu_raw = (x - ((1 - alpha) / denom) * eps) / np.sqrt(alpha) # eps_raw式
            mu = (1 - mu_blend) * mu_x0 + mu_blend * mu_raw
        else:
            raise ValueError(f"Unknown mu_mode: {mu_mode}")

        if mu_dc_beta > 0:
            mu -= mu_dc_beta * mu.mean(axis=dc_axis, keepdims=True)

        """ step5 """
        # muとstdからx_prev 算出        　
        if int(t_prev) == 0:
            # 最終ステップではノイズを加えず、事後平均 mu をそのまま x_0 とみなす
            x_prev = mu 
            stdz = np.zeros_like(x_prev, dtype=Config.dtype) # ログに使うのみ 
        else:
            noise = np.random.randn(*x.shape).astype(x.dtype)
            stdz = std * noise # サンプリング項
            x_prev = mu + eta * stdz

        if x_prev_dc_beta > 0:
            #beta_t = x_prev_dc_beta * np.sqrt(1 - alpha_bar_prev)  # t大では小、t小で効く
            x_prev -= x_prev_dc_beta * x_prev.mean(axis=dc_axis, keepdims=True)

        if preserve_mean_beta > 0:
            x_prev += (preserve_mean_beta
                       * ((x.mean(axis=dc_axis, keepdims=True)
                           - x_prev.mean(axis=dc_axis, keepdims=True))))

        """ step6 """
        # ログ情報収集
        if debug:
            self.analizer.append_log(t, x, x_prev, x0_pre, x0_hat, eps, eps_recon, mu, stdz, clip_rate)
        return x_prev


    def denoise_ddim(self, model, x, t, t_prev, labels=None, debug=False, **kwargs):
        """ DDIM (Denoising Diffusion Implicit Models) """
        self.kwargs    = kwargs.copy()

        # 標準オプション standard options
        eta            = kwargs.pop('eta', 0.0)   # DDIMの基本は eta=0
        clip_denoised  = kwargs.pop('clip_denoised', False)
        dt_p           = kwargs.pop('dt_p', 0.995)
        denom_floor    = kwargs.pop('denom_floor', 1e-4)
        preserve_mean_beta = kwargs.pop('preserve_mean_beta', 0.0)
        sample_state   = kwargs['sample_state'] # 常に受け取る

        # 誘導オプション
        gamma          = kwargs.pop('gamma', None)
        x0_target      = kwargs.pop('x0_target', None)
        guide          = kwargs.pop('guide', 0.0)

        # 必要に応じて行う簡単なオプション　optional simple corrections
        dc_axis        = kwargs.pop('dc_axis', (0, 2, 3))
        x0_plane_beta  = kwargs.pop('x0_plane_beta', 0.0)
        x0_dc_beta     = kwargs.pop('x0_dc_beta', 0.0)
        x0_rowcol_beta = kwargs.pop('x0_rowcol_beta', 0.0)
        eps_recon_dc_beta  = kwargs.pop('eps_recon_dc_beta', 0.0)
        x_prev_dc_beta  = kwargs.pop('x_prev_dc_beta', 0.0)

        # 実験段階のオプション experimental corrections
        eps_mix_beta   = kwargs.pop('eps_mix_beta',0.0)
        eps_mix_remove_dc = kwargs.pop('eps_mix_remove_dc', False)
        eps_mix_axis   = kwargs.pop('eps_mix_axis', (2,3))
        preserve_lowfreq_beta = kwargs.pop('preserve_lowfreq_beta', 0.0)
        preserve_lowband_beta    = kwargs.pop('preserve_lowband_beta', 0.0)
        preserve_lowband_k       = kwargs.pop('preserve_lowband_k', 4)
        preserve_lowband_keep_dc = kwargs.pop('preserve_lowband_keep_dc', True)
        preserve_x0_hat_energy_beta = kwargs.pop('preserve_x0_hat_energy_beta', 0.0)
        x0_hat_energy_until_t  = kwargs.pop('x0_hat_energy_until_t', 500)

        # 一旦外す候補
        x_dc_beta      = kwargs.pop('x_dc_beta', 0.0)
        eps_dc_beta    = kwargs.pop('eps_dc_beta', 0.0)

        # サンプリングでは t はスカラ（または全要素同一の (N,)）とする
        t = self.fix_t(t, 1)
        t_prev = self.fix_t(t_prev, 0)

        alpha_bar      = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev]

        if x_dc_beta > 0:
            x -= x_dc_beta * x.mean(axis=dc_axis, keepdims=True)

        """ step1: eps prediction """
        if gamma is not None:
            eps = model(x, t, labels)
            eps_uncond = model(x, t)
            eps = eps_uncond + gamma * (eps - eps_uncond)
        else:
            eps = model(x, t, labels)

        if eps_dc_beta > 0:
            eps -= eps_dc_beta * eps.mean(axis=dc_axis, keepdims=True)

        """ step2: x0 reconstruction """
        #x0_hat = (x - np.sqrt(1.0 - alpha_bar) * eps) / np.sqrt(alpha_bar)
        alpha_bar_safe = np.maximum(alpha_bar, 1e-4)
        x0_hat = (x - np.sqrt(1.0 - alpha_bar) * eps) / np.sqrt(alpha_bar_safe)

        if preserve_x0_hat_energy_beta > 0:
            # x0_hat の低周波エネルギーが前ステップより低下した場合のみ持ち上げる
            x0_hat_energy = self.analizer.lowfreq_energy(x0_hat)
            prev_energy = sample_state.get("x0_hat_energy", None)

            if prev_energy is not None and t < x0_hat_energy_until_t:
                rate = prev_energy / (x0_hat_energy + 1e-12)
                scale = np.where(rate > 1.0,
                                 1.0 + preserve_x0_hat_energy_beta * (rate - 1.0),
                                 1.0)
                scale = scale[..., np.newaxis, np.newaxis]
                x0_hat = x0_hat * scale
            # 次回に備えて補正後のx0_hatの値を保存
            sample_state["x0_hat_energy"] = self.analizer.lowfreq_energy(x0_hat)

        if x0_plane_beta > 0:
            x0_hat -= x0_plane_beta * self._plane(x0_hat) 

        if x0_rowcol_beta > 0:
            x0_hat -= x0_rowcol_beta * self.rowcol_bias(x0_hat) 

        if debug:
            x0_pre = x0_hat.copy()

        if clip_denoised:
            x0_hat, s, clip_rate \
                = self.dynamic_thresholding(x0_hat, p=dt_p, clip_val=1.0)
        else:
            s = 0
            clip_rate = np.zeros(x0_hat.shape[:2], dtype=Config.dtype)

        if x0_target is not None and guide is not None and guide > 0.0:
            x0_hat = self.target_guide(x0_hat, alpha_bar, x0_target, guide)

        if x0_dc_beta > 0:
            beta = x0_dc_beta #* np.sqrt(1.0 - alpha_bar_prev)
            x0_hat -= beta * x0_hat.mean(axis=dc_axis, keepdims=True)

        """ step3: eps consistency """
        denom = np.sqrt(np.maximum(1.0 - alpha_bar, denom_floor))
        eps_recon = (x - np.sqrt(alpha_bar) * x0_hat) / denom

        if eps_recon_dc_beta > 0:
            eps_recon -= eps_recon_dc_beta * eps_recon.mean(axis=dc_axis, keepdims=True)

        if eps_mix_beta > 0:  
            eps_recon_used = self.mix_eps_with_white(
                eps_recon,
                beta=eps_mix_beta,
                remove_dc=eps_mix_remove_dc,
                axis=eps_mix_axis,
            )
        else:
            eps_recon_used = eps_recon

        """ step4: DDIM update """
        if eta == 0.0:
            sigma = np.zeros_like(alpha_bar, dtype=Config.dtype)
            dir_coef = np.sqrt(np.maximum(1.0 - alpha_bar_prev, 0.0))
        else:
            sigma = (
                eta
                * np.sqrt(np.maximum((1.0 - alpha_bar_prev) / np.maximum(1.0 - alpha_bar, denom_floor), 0.0))
                * np.sqrt(np.maximum(1.0 - (alpha_bar / np.maximum(alpha_bar_prev, denom_floor)), 0.0))
            )
            dir_coef = np.sqrt(np.maximum(1.0 - alpha_bar_prev - sigma * sigma, 0.0))

        if int(t_prev) == 0:
            x_prev = x0_hat
            mu = x_prev
            stdz = np.zeros_like(x_prev, dtype=Config.dtype)
        else:
            noise = np.random.randn(*x.shape).astype(x.dtype)
            #noise = make_lowfreq_noise(
            #        x_shape=x.shape,
            #        dtype=x.dtype,
            #        cutoff_ratio=0.125,
            #        normalize_std=True,
            #        per_channel=True,
            #    )
            
            stdz = sigma * noise
            mu = np.sqrt(alpha_bar_prev) * x0_hat + dir_coef * eps_recon_used
            #mu = np.sqrt(alpha_bar_prev) * x0_hat + dir_coef * eps
            x_prev = mu + stdz

        if x_prev_dc_beta > 0:
            x_prev -= x_prev_dc_beta * x_prev.mean(axis=dc_axis, keepdims=True)

        if preserve_mean_beta > 0:
            x_prev += preserve_mean_beta * (
                x.mean(axis=dc_axis, keepdims=True)
                - x_prev.mean(axis=dc_axis, keepdims=True)
            )

        if preserve_lowfreq_beta > 0:
            low_t    = self.lowfreq_component(x, k=4)
            low_prev = self.lowfreq_component(x_prev, k=4)
            x_prev += preserve_lowfreq_beta * (low_t - low_prev)

        if preserve_lowband_beta > 0:
            x_prev = self.preserve_lowfreq_band(x_prev, x, 
                beta=preserve_lowband_beta,
                k=preserve_lowband_k,
                keep_dc=preserve_lowband_keep_dc,)
            
        """ step5: logging """
        if debug:
            self.analizer.append_log(
                t, x, x_prev, x0_pre, x0_hat, eps, eps_recon, mu, stdz, clip_rate
            )
 
        return x_prev

    def sample(self, model, x_shape=(20, 1, 28, 28), x=None, labels=None,
               sampler=None, steps=None, start=None, halt=None, debug=False,
               jacobian_debug=False, batch_size=10, **kwargs):

        if x is None:
            x = np.random.randn(*x_shape).astype(Config.dtype)

        ts, t_prevs = self.schedule_time_steps(steps=steps)

        if sampler in (None, 'default'):
            denoise_fn = self.denoise
        elif sampler == "ddpm":
            denoise_fn = self.denoise_ddpm
        elif sampler == "ddim":
            denoise_fn = self.denoise_ddim
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        kwargs.setdefault("sample_state", {"x0_hat_energy": None})

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
                
                if jacobian_debug and self.jacobian_analizer is not None:
                    self.jacobian_analizer.measure_step(model, denoise_fn, xb, t, t_prev)

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
        l = lf.MeanSquaredError(reduction='sample')(eps_hat, eps) # 20260426AI
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
                       use_true_x0_for_test=False, 
                       dt_p=0.995):
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
            # inversion の出発点は x0 であるが、反復の各時刻で直接使えるのは現在状態 x_{t_prev} のみである。
            # DDIM の決定的更新は x_{t_prev}, eps_hat, x0_hat の関係で定まるため、
            # 各ステップでは eps_hat から x0_hat を再推定し、それを用いて次の x_t を構成する。
            x0_hat = (x - np.sqrt(1.0 - alpha_bar_from) * eps_hat) / np.sqrt(alpha_bar_from)
            if clip_denoised:
                x0_hat = self.dynamic_thresholding(x0_hat, p=dt_p, clip_val=1.0)

            x0_used = x0 if use_true_x0_for_test else x0_hat # 検証用: 真の x0 を固定

            # 決定的 DDIM forward: x_{t_to}
            x = np.sqrt(alpha_bar_to) * x0_used + np.sqrt(1.0 - alpha_bar_to) * eps_hat
        return x

    def _plane(self, x):

        B,C,H,W = x.shape

        yy = np.linspace(-1,1,H).reshape(H,1)
        xx = np.linspace(-1,1,W).reshape(1,W)

        X0 = np.ones((H,W))
        X1 = np.broadcast_to(xx,(H,W))
        X2 = np.broadcast_to(yy,(H,W))

        A = np.stack([X0,X1,X2],axis=0).reshape(3,-1)
        A = A.T

        A_pinv = np.linalg.pinv(A)

        xf = x.reshape(B,C,-1)

        coeff = xf @ A_pinv.T

        plane = coeff @ A.T
        plane = plane.reshape(B,C,H,W)

        return plane
    
    def rowcol_bias(self, x):
        # x: BCHW
        m = x.mean(axis=(2,3), keepdims=True)      # B,C,1,1
        r = x.mean(axis=3, keepdims=True)          # B,C,H,1
        c = x.mean(axis=2, keepdims=True)          # B,C,1,W
        return r + c - m                           # B,C,H,W

    def lowpass2d(self, x, k=7):
        """
        簡易 low-pass filter
        mean pooling による低周波抽出
        k : kernel size (odd 推奨)
        """
        np = Config.np

        pad = k // 2

        # reflect padding
        x_pad = np.pad(
            x,
            ((0,0),(0,0),(pad,pad),(pad,pad)),
            mode='reflect'
        )

        out = np.zeros_like(x)

        for i in range(k):
            for j in range(k):
                out += x_pad[:, :, i:i+x.shape[2], j:j+x.shape[3]]

        out /= (k * k)

        return out

    def mix_eps_with_white(self, eps, beta=0.0, remove_dc=False, axis=(2,3)):
        """
        eps に白色ノイズを混合して、空間構造を少し崩す。
        beta=0 ならそのまま返す。

        eps_used = (1-beta)*eps + beta*noise
        """
        if beta <= 0:
            return eps

        np = Config.np
        noise = np.random.randn(*eps.shape).astype(eps.dtype)

        if remove_dc:
            noise = noise - noise.mean(axis=axis, keepdims=True)

        eps_used = (1.0 - beta) * eps + beta * noise
        return eps_used


    def lowfreq_component(self, x, k=4):
        import numpy
        if np.__name__=='cupy':
            x = np.asnumpy(x)

        B,C,H,W = x.shape
        low = numpy.zeros_like(x, dtype=Config.dtype)

        for b in range(B):
            for c in range(C):

                img = x[b,c]

                d = dct(dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

                mask = numpy.zeros_like(d)
                mask[:k,:k] = 1
                #mask[0,0] = 0

                d_low = d * mask

                low[b,c] = idct(idct(d_low, axis=0, norm='ortho'), axis=1, norm='ortho')

        return np.asarray(low)


    def preserve_lowfreq_modes(self, x_prev, x, beta=0.1, modes=((0,1), (1,0), (1,1))):
        """
        x_prev, x : (B,C,H,W)

        DCT空間で selected modes を x 側に寄せる。
        DC(0,0)は触らない。
        """
        import numpy
        if np.__name__ == 'cupy':
            x = np.asnumpy(x)
            x_prev = np.asnumpy(x_prev)

        B, C, H, W = x.shape
        y = x_prev.copy()

        for b in range(B):
            for c in range(C):
                img_prev = x_prev[b, c]
                img_t    = x[b, c]

                d_prev = dct(dct(img_prev, axis=0, norm='ortho'), axis=1, norm='ortho')
                d_t    = dct(dct(img_t,    axis=0, norm='ortho'), axis=1, norm='ortho')

                # 指定モードだけ補償
                for i, j in modes:
                    d_prev[i, j] = (1.0 - beta) * d_prev[i, j] + beta * d_t[i, j]

                # 画像空間へ戻す
                y[b, c] = idct(idct(d_prev, axis=0, norm='ortho'), axis=1, norm='ortho')

        print('### preserve_lowfreq_modes ###') 
        return np.asarray(y)

    def preserve_lowfreq_band(self, x_prev, x, beta=0.1, k=4, keep_dc=False):
        """
        x_prev, x : (B,C,H,W)

        DCT空間で k×k low-band を x 側に寄せる。
        keep_dc=False のときは DC(0,0) は触らない。
        """
        import numpy
        if np.__name__ == 'cupy':
            x = np.asnumpy(x)
            x_prev = np.asnumpy(x_prev)

        B, C, H, W = x.shape
        y = x_prev.copy()

        for b in range(B):
            for c in range(C):
                img_prev = x_prev[b, c]
                img_t    = x[b, c]

                d_prev = dct(dct(img_prev, axis=0, norm='ortho'), axis=1, norm='ortho')
                d_t    = dct(dct(img_t,    axis=0, norm='ortho'), axis=1, norm='ortho')

                # k×k low-band 全体を補償
                for i in range(k):
                    for j in range(k):
                        if (not keep_dc) and i == 0 and j == 0:
                            continue
                        d_prev[i, j] = (1.0 - beta) * d_prev[i, j] + beta * d_t[i, j]

                # 画像空間へ戻す
                y[b, c] = idct(idct(d_prev, axis=0, norm='ortho'), axis=1, norm='ortho')

        print('### preserve_lowfreq_band ###')
        return np.asarray(y)

    def preserve_x0_hat_lowfreq_floor(self, x0_hat, x, beta=0.1, k=4, eps=1e-8):
        """
        x0_hat, x : (B,C,H,W)

        x0_hat の k×k low-band (DC除く) の energy が、
        x の corresponding low-band energy より小さいときだけ、
        non-DC low-band 全体を一様増幅して下支えする。

        beta=0   : 補償なし
        beta=1   : energy不足分をフル補償
        """
        import numpy
        if np.__name__ == 'cupy':
            x = np.asnumpy(x)
            x0_hat = np.asnumpy(x0_hat)

        B, C, H, W = x.shape
        y = x0_hat.copy()

        count = 0 # 保証を行った回数記録
        for b in range(B):
            for c in range(C):
                img_hat = x0_hat[b, c]
                img_t   = x[b, c]

                d_hat = dct(dct(img_hat, axis=0, norm='ortho'), axis=1, norm='ortho')
                d_t   = dct(dct(img_t,   axis=0, norm='ortho'), axis=1, norm='ortho')

                low_hat = d_hat[:k, :k].copy()
                low_t   = d_t[:k, :k].copy()

                # non-DC energy
                low_hat[0, 0] = 0
                low_t[0, 0]   = 0

                e_hat = numpy.sqrt((low_hat**2).sum())
                e_t   = numpy.sqrt((low_t**2).sum())

                # x0_hat の lowfreq が不足しているときだけ床上げ
                if e_hat < e_t:
                    gain = e_t / (e_hat + eps)
                    gain = 1.0 + beta * (gain - 1.0)

                    for i in range(k):
                        for j in range(k):
                            if i == 0 and j == 0:
                                continue
                            d_hat[i, j] *= gain

                    count += 1        

                y[b, c] = idct(idct(d_hat, axis=0, norm='ortho'), axis=1, norm='ortho')

        print(f'### preserve_x0_hat_lowfreq_floor. count={count} ###')
        return np.asarray(y)

class Analizer:
    """ Diffuserのdenoiseのlogの収集、解析、グラフ描画を担う """
    def __init__(self, diffuser):
        self.diffuser = diffuser
        self.log = []             
    
    def append_log(self, t, x, x_prev, x0_pre, x0_hat, eps, eps_recon, mu, stdz, clip_rate,
                   axis=(2,3)):
        """ logを収集する """
        if np.may_share_memory(x, x_prev):
            print("WARN share_memory: t=", int(t))


        e_no_dc_xt, e_all_xt, dc_xt = self.lowfreq_energy(x, k=4, return_stats=True)
        e_no_dc_x_prev, e_all_x_prev, dc_x_prev = self.lowfreq_energy(x_prev, k=4, return_stats=True)

        self.log.append({
            't'             : t,
            'mu_xt'         : x.mean(axis=axis),
            'std_xt'        : x.std(axis=axis),
            'mu_x_prev'     : x_prev.mean(axis=axis),
            'std_x_prev'    : x_prev.std(axis=axis),
            'mu_x0_pre'     : x0_pre.mean(axis=axis),
            'std_x0_pre'    : x0_pre.std(axis=axis),
            'mu_x0_hat'     : x0_hat.mean(axis=axis),
            'std_x0_hat'    : x0_hat.std(axis=axis),
            'mu_eps'        : eps.mean(axis=axis),
            'std_eps'       : eps.std(axis=axis),
            'mu_eps_recon'  : eps_recon.mean(axis=axis),
            'std_eps_recon' : eps_recon.std(axis=axis),
            'mu_mu'         : mu.mean(axis=axis),
            'mu_stdz'       : stdz.mean(axis=axis),
            'clip_rate'     : clip_rate,

            # ---- plane stats を追加 ----
            **self._plane_stats_dict('xt', x),
            **self._plane_stats_dict('x0_pre', x0_pre),
            **self._plane_stats_dict('x0_hat', x0_hat),

            'lowfreq_delta' : self.lowfreq_energy(x_prev - x),
            'lowfreq_energy_x0_hat' : self.lowfreq_energy(x0_hat),
            **self.lowfreq_axis_stats('x0_hat', x0_hat),
            **self.lowfreq_delta_energy_gain(x, x_prev), 
            **self.lowfreq_cosine_theta(x, x_prev),
            
            'e_no_dc_xt'    : e_no_dc_xt,
            'e_all_xt'      : e_all_xt,
            'dc_xt'         : dc_xt,
            'e_no_dc_x_prev': e_no_dc_x_prev,
            'e_all_x_prev'  : e_all_x_prev,
            'dc_x_prev'     : dc_x_prev,

            })

    def get_stem(self, epoch):
        stem = f"epoch{epoch:03d}"
        if self.diffuser.kwargs is not None:
            stem += '_'.join([f"{k}_{v}" for k, v in sorted(self.diffuser.kwargs.items())])
        stem = re.sub(r'[^A-Za-z0-9\_]', '', stem)
        return stem
        
    def save_log(self, epoch, out_dir: Path, flush=True):
        """ 収集したlogをファイルに格納する """
        log = self.log
        stem = self.get_stem(epoch)
        self.file = str(out_dir / f"log_summary_{stem}.npz")
        print(self.file)
        
        if len(log) == 0:
            raise RuntimeError("diffuser.log is empty. Did you run sample(debug=True)?")

        _stack_log = lambda log, key : np.stack([l[key] for l in log], axis=0)


        lowfreq_gain = _stack_log(log, 'lowfreq_gain')
        gain = np.maximum(lowfreq_gain, 1e-12)   # 0割やlog対策
        log_gain = np.log(gain)
        log_cum_gain = np.cumsum(log_gain[::-1], axis=0)[::-1]
        cum_gain = np.exp(log_cum_gain)

        # logから読出して保存
        np.savez(self.file,  
                 t             = _stack_log(log, 't'),
                 mu_xt         = _stack_log(log, 'mu_xt'),
                 std_xt        = _stack_log(log, 'std_xt'),
                 mu_x_prev     = _stack_log(log, 'mu_x_prev'),
                 std_x_prev    = _stack_log(log, 'std_x_prev'),
                 mu_x0_pre     = _stack_log(log, 'mu_x0_pre'),
                 std_x0_pre    = _stack_log(log, 'std_x0_pre'),
                 mu_x0_hat     = _stack_log(log, 'mu_x0_hat'),
                 std_x0_hat    = _stack_log(log, 'std_x0_hat'),
                 mu_eps        = _stack_log(log, 'mu_eps'),
                 std_eps       = _stack_log(log, 'std_eps'),
                 mu_eps_recon  = _stack_log(log, 'mu_eps_recon'),
                 std_eps_recon = _stack_log(log, 'std_eps_recon'),
                 mu_mu         = _stack_log(log, 'mu_mu'),
                 mu_stdz       = _stack_log(log, 'mu_stdz'),
                 clip_rate     = _stack_log(log, 'clip_rate'),
                 
                 # 以下追加分20260309
                 a_xt          = _stack_log(log, 'a_xt'),
                 b_xt          = _stack_log(log, 'b_xt'),
                 c_xt          = _stack_log(log, 'c_xt'),
                 plane_rms_xt  = _stack_log(log, 'plane_rms_xt'),
                 total_rms_xt  = _stack_log(log, 'total_rms_xt'),
                 plane_ratio_xt= _stack_log(log, 'plane_ratio_xt'),

                 a_x0_pre          = _stack_log(log, 'a_x0_pre'),
                 b_x0_pre          = _stack_log(log, 'b_x0_pre'),
                 c_x0_pre          = _stack_log(log, 'c_x0_pre'),
                 plane_rms_x0_pre  = _stack_log(log, 'plane_rms_x0_pre'),
                 total_rms_x0_pre  = _stack_log(log, 'total_rms_x0_pre'),
                 plane_ratio_x0_pre= _stack_log(log, 'plane_ratio_x0_pre'),

                 a_x0_hat          = _stack_log(log, 'a_x0_hat'),
                 b_x0_hat          = _stack_log(log, 'b_x0_hat'),
                 c_x0_hat          = _stack_log(log, 'c_x0_hat'),
                 plane_rms_x0_hat  = _stack_log(log, 'plane_rms_x0_hat'),
                 total_rms_x0_hat  = _stack_log(log, 'total_rms_x0_hat'),
                 plane_ratio_x0_hat= _stack_log(log, 'plane_ratio_x0_hat'),

                 lowfreq_delta     = _stack_log(log, 'lowfreq_delta'),

                 x0_hat_row_std_dc = _stack_log(log, 'x0_hat_row_std_dc'),
                 x0_hat_col_std_dc = _stack_log(log, 'x0_hat_col_std_dc'),

                 lowfreq_energy_xt = _stack_log(log, 'lowfreq_energy_xt'),
                 lowfreq_energy_x_prev = _stack_log(log, 'lowfreq_energy_x_prev'),
                 lowfreq_energy_x0_hat = _stack_log(log, 'lowfreq_energy_x0_hat'),
                 lowfreq_delta_energy = _stack_log(log, 'lowfreq_delta_energy'),
                 lowfreq_gain         = _stack_log(log, 'lowfreq_gain'),
                 lowfreq_cos       = _stack_log(log, 'lowfreq_cos'),
                 cum_gain          = cum_gain,

                 e_no_dc_xt        = _stack_log(log, 'e_no_dc_xt'),
                 e_all_xt          = _stack_log(log, 'e_all_xt'),
                 dc_xt             = _stack_log(log, 'dc_xt'),
                 e_no_dc_x_prev    = _stack_log(log, 'e_no_dc_x_prev'),
                 e_all_x_prev      = _stack_log(log, 'e_all_x_prev'),
                 dc_x_prev         = _stack_log(log, 'dc_x_prev'),
                 
                 )


        if flush:
            self.log.clear()

    def rgb_plot(self, x_t3s, t, title, out_png,
                      series_names=None,
                      channel_names=("R", "G", "B"),
                      ylim=None):
        """
        x_t3s: (T,3) もしくは iterable of (T,3)
        series_names: 各系列の名前。例 ("mu_xt", "mu_x_prev")
        """
        plt.figure()
        colors = {"R": ["lightcoral", "indianred", "red", "darkred"],
                  "G": ["palegreen", "mediumseagreen", "green", "darkgreen"],
                  "B": ["lightskyblue", "cornflowerblue", "blue", "darkblue"]}

        # 単一入力をタプル化
        if isinstance(x_t3s, (list, tuple)):
            xs = x_t3s
        else:
            xs = (x_t3s,)

        # 系列名が無い場合は自動付与
        if series_names is None:
            if len(xs) == 1:
                series_names = ("x",)
            else:
                series_names = tuple(f"s{i}" for i in range(len(xs)))
        elif isinstance(series_names, str):
            series_names = (series_names,)
        assert len(series_names) == len(xs), \
                            "len(series_names) must match number of series"

        # plot
        k = 1 - (len(xs) - 1)//2 # グラフの色
        for i, (sname, x) in enumerate(zip(series_names, xs)):
            for c, cname in enumerate(channel_names):
                plt.plot(t.tolist(), x[:, c].tolist(),
                         label=f"{sname}_{cname.upper()}", color=colors[cname][i+k])

        plt.gca().invert_xaxis()
        plt.ticklabel_format(useOffset=False)
        plt.title(title)
        plt.xlabel("t")
        plt.grid(True)
        plt.legend(ncol=len(xs), fontsize=9)  # 凡例が長いので2列推奨
        if ylim is not None:
            plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(out_png) # 出力先の.pngファイル名
        plt.close()

    def rgb_scatter(self, xs, ys, title, out_png,
                      series_name=None,
                      channel_names=("R", "G", "B"),
                      ylim=None):
        """
        series_name: 系列の名前。例 ("mu_xt", "mu_x_prev")
        """
        plt.figure()
        colors = {"R": "indianred",
                  "G": "mediumseagreen",
                  "B": "cornflowerblue"}

        for c, cname in enumerate(channel_names):
            plt.scatter(xs[:, c].tolist(), ys[:, c].tolist(),
                        label=f"{series_name}_{cname.upper()}",
                        marker='+', color=colors[cname])

        plt.title(title)
        plt.grid(True)
        plt.legend(ncol=1, fontsize=9)  # 凡例が長いので2列推奨
        if ylim is not None:
            plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(out_png) # 出力先の.pngファイル名
        plt.close()

    def analize_and_draw(self, epoch, out_dir: Path, sufix=None,
                           rgb_plots=('mu_xt_c_and_mu_x_prev_c',
                                      'mu_eps_c',
                                      'a','b','r','mse',
                                      'diff_x_prev_xt'),
                           rgb_scatters=()
                           ):
        """ 各種解析用のグラフを作って格納する """

        stem = self.get_stem(epoch)
        print(stem)
        
        file = str(out_dir / f"log_summary_{stem}.npz")
        print(file)

        # ファイルからlogを読出す
        data = np.load(file)

        t              = data['t']
        mu_xt          = data['mu_xt']
        std_xt         = data['std_xt']
        mu_x_prev      = data['mu_x_prev']
        std_x_prev     = data['std_x_prev']
        mu_x0_pre      = data['mu_x0_pre']
        std_x0_pre     = data['std_x0_pre']
        mu_x0_hat      = data['mu_x0_hat']
        std_x0_hat     = data['std_x0_hat']
        mu_eps         = data['mu_eps']
        std_eps        = data['std_eps']
        mu_eps_recon   = data['mu_eps_recon']
        std_eps_recon  = data['std_eps_recon']
        mu_mu          = data['mu_mu']
        mu_stdz        = data['mu_stdz']
        clip_rate      = data['clip_rate']

        # 必要な値を算出
        mu_x0_pre_c    = mu_x0_pre.mean(axis=1)
        mu_x0_hat_c    = mu_x0_hat.mean(axis=1)
        delta_mu_c     = mu_x0_hat_c - mu_x0_pre_c
        mu_xt_c        = mu_xt.mean(axis=1)
        mu_x_prev_c    = mu_x_prev.mean(axis=1)
        mu_mu_c        = mu_mu.mean(axis=1)
        mu_stdz_c      = mu_stdz.mean(axis=1)
        mu_eps_c       = mu_eps.mean(axis=1)            
        mu_eps_recon_c = mu_eps_recon.mean(axis=1)

        # 以下20260309
        mu_xt_c, std_xt_c = self._global_mean_std_from_spatial(mu_xt, std_xt)
        mu_x0_hat_c, std_x0_hat_c = self._global_mean_std_from_spatial(mu_x0_hat, std_x0_hat)
        mu_x0_pre_c, std_x0_pre_c = self._global_mean_std_from_spatial(mu_x0_pre, std_x0_pre)
        mu_eps_c, std_eps_c = self._global_mean_std_from_spatial(mu_eps, std_eps)   
        mu_eps_recon_c, std_eps_recon_c = self._global_mean_std_from_spatial(mu_eps_recon, std_eps_recon)   
        # 以上 

        luma_xt,     chroma_xt     = self._luma_chroma(mu_xt_c)
        luma_x_prev, chroma_x_prev = self._luma_chroma(mu_x_prev_c)

        a, b, r, mse = self._fit_affine_per_channel(mu_xt, mu_x_prev)

        # グラフ描画
        if 'delta_mu_c' in rgb_plots:
            self.rgb_plot(delta_mu_c, t,
                     title="delta_mu_c (mu_post - mu_pre)",
                     out_png=str(out_dir / f"delta_mu_{stem}_{sufix}.png"),
                     series_names=("delta_mu",),
                     ylim=None)

        if 'mu_pre vs mu_post' in rgb_plots:
            self.rgb_plot((mu_x0_pre_c, mu_x0_hat_c), t,
                     title="mu_pre vs mu_post (channel mean over BHW)",
                     out_png=str(out_dir / f"mu_pre_post_{stem}_{sufix}.png"),
                     series_names=("mu_pre", "mu_post"),
                     ylim=None)

        if 'mu_xt_c_and_mu_x_prev_c' in rgb_plots:
            self.rgb_plot((mu_xt_c, mu_x_prev_c), t,
                     title="mu_xt_c and mu_x_prev_c (mean of x and x_prev over BHW)",
                     out_png=str(out_dir / f"mu_xt_x_prev_{stem}_{sufix}.png"),
                     series_names=("mu_xt", "mu_x_prev"),
                     ylim=None)

        if 'mu_mu_c_mu_stdz_c' in rgb_plots:
            self.rgb_plot((mu_mu_c, mu_stdz_c), t,
                     title="mu_mu_c mu_stdz_c (mean of mu and stdz over BHW)",
                     out_png=str(out_dir / f"mu_mu_stdz_{stem}_{sufix}.png"),
                     series_names=("mu", "stdz"),
                     ylim=None)

        if 'mu_eps_c' in rgb_plots:
            self.rgb_plot((mu_eps_c, mu_eps_recon_c), t,
                     title="mu_eps_c (mean over batch of mu_eps)",
                     out_png=str(out_dir / f"mu_eps_c_{stem}_{sufix}.png"),
                     series_names=("mu_eps_c", "mu_eps_recon_c",),
                     ylim=None)

        if 'chroma_xt_and_chroma_x_prev' in rgb_plots:
            self.rgb_plot((chroma_xt, chroma_x_prev), t,
                     title="chroma_xt and chroma_x_prev (RGB-mean subtracted)",
                     out_png=str(out_dir / f"chroma_xt_x_prev_{stem}_{sufix}.png"),
                     series_names=("chroma_xt", "chroma_x_prev"),
                     ylim=None)

        if 'a' in rgb_plots:
            self.rgb_plot(a, t,
                     title="a",
                     out_png=str(out_dir / f"a_xt_x_prev_{stem}_{sufix}.png"),
                     series_names=("a",),
                     ylim=None)

        if 'b' in rgb_plots:
            self.rgb_plot(b, t,
                     title="b",
                     out_png=str(out_dir / f"b_xt_x_prev_{stem}_{sufix}.png"),
                     series_names=("b",),
                     ylim=None)

        if 'r' in rgb_plots:
            self.rgb_plot(r, t,
                     title="r",
                     out_png=str(out_dir / f"r_xt_x_prev_{stem}_{sufix}.png"),
                     series_names=("r",),
                     ylim=None)

        if 'mse' in rgb_plots:
            self.rgb_plot(mse, t,
                     title="mse",
                     out_png=str(out_dir / f"mse_xt_x_prev_{stem}_{sufix}.png"),
                     series_names=("mse",),
                     ylim=None)

        if 'mu_x_prev_c_mu_xt_c' in rgb_scatters:
            self.rgb_scatter(mu_xt_c, mu_x_prev_c - mu_xt_c,
                        title='mu_x_prev - mu_xt_c',
                        out_png=str(out_dir / f"mu_xt_c_mu_x_prev_c_{stem}_{sufix}.png"),
                        series_name='diff')

        if 'diff_x_prev_xt' in rgb_plots:
            self.rgb_plot(mu_x_prev_c - mu_xt_c, t,
                     title='diff_x_prev_xt',
                     out_png=str(out_dir / f"diff_x_prev_xt_c_{stem}_{sufix}.png"),
                     series_names='diff',
                     ylim=None)

                    

        """
        # 以下、0260302追加分（仮実装）
     
        diff_obs = mu_x_prev_c - mu_xt_c        # (T,C)

        a    = self.diffuser.alphas[t]
        abar = self.diffuser.alpha_bars[t]

        eps = 1e-12
        inv_sqrt_a = 1.0 / np.sqrt(a + eps)
        coef_eps   = inv_sqrt_a * (1.0 - a) / np.sqrt(1.0 - abar + eps)

        term_eps = -coef_eps[:,None] * mu_eps_recon_c   # (T,C)

        coef_x  = inv_sqrt_a - 1.0
        term_x  = coef_x[:,None] * mu_xt_c
        diff_th = term_x + term_eps

        self.rgb_plot((diff_obs, term_eps, term_x, diff_th), t,
                 title='diff_obs_term_eps_term_x_diff_th',
                 out_png=str(out_dir / f"diff_obs_term_eps_termx_diff_th_{stem}_{sufix}.png"),
                 series_names=('diff_obs', 'term_eps', 'term_x', 'diff_th'),
                 ylim=None)
        
        self.rgb_plot(diff_obs, t,
                 title='diff_obs',
                 out_png=str(out_dir / f"diff_obs_{stem}_{sufix}.png"),
                 series_names='diff_obs',
                 ylim=None)

        self.rgb_plot((term_eps, term_x, diff_th), t,
                 title='term_eps_term_x_diff_th',
                 out_png=str(out_dir / f"term_eps_termx_diff_th_{stem}_{sufix}.png"),
                 series_names=('term_eps', 'term_x', 'diff_th'),
                 ylim=None)
        """

        self.rgb_plot((mu_xt_c, std_xt_c), t,
                 title='mu_and_std_xt_c',
                 out_png=str(out_dir / f"mu_and_std_xt_c_{stem}_{sufix}.png"),
                 series_names=('mu_xt_c', 'std_xt_c'),
                 ylim=None)
        
        self.rgb_plot((mu_x0_pre_c, std_x0_pre_c), t,
                 title='mu_and_std_x0_pre_c',
                 out_png=str(out_dir / f"mu_and_std_x0_pre_c_{stem}_{sufix}.png"),
                 series_names=('mu_x0_pre_c', 'std_x0_pre_c'),
                 ylim=None)

        self.rgb_plot((mu_x0_hat_c, std_x0_hat_c), t,
                 title='mu_and_std_x0_hat_c',
                 out_png=str(out_dir / f"mu_and_std_x0_hat_c_{stem}_{sufix}.png"),
                 series_names=('mu_x0_hat_c', 'std_x0_hat_c'),
                 ylim=None)

        self.rgb_plot((mu_eps_c, std_eps_c), t,
                 title='mu_and_std_eps_c',
                 out_png=str(out_dir / f"mu_and_std_eps_c_{stem}_{sufix}.png"),
                 series_names=('mu_eps_c', 'std_eps_c'),
                 ylim=None)

        self.rgb_plot((mu_eps_recon_c, std_eps_recon_c), t,
                 title='mu_and_std_eps_recon_c',
                 out_png=str(out_dir / f"mu_and_std_eps_recon_c_{stem}_{sufix}.png"),
                 series_names=('mu_eps_recon_c', 'std_eps_recon_c'),
                 ylim=None)

        # 以下追加分20260309
        a_xt           = data['a_xt']
        b_xt           = data['b_xt']
        c_xt           = data['c_xt']
        plane_rms_xt   = data['plane_rms_xt']
        total_rms_xt   = data['total_rms_xt']
        plane_ratio_xt = data['plane_ratio_xt']

        a_x0_pre           = data['a_x0_pre']
        b_x0_pre           = data['b_x0_pre']
        c_x0_pre           = data['c_x0_pre']
        plane_rms_x0_pre   = data['plane_rms_x0_pre']
        total_rms_x0_pre   = data['total_rms_x0_pre']
        plane_ratio_x0_pre = data['plane_ratio_x0_pre']

        a_x0_hat           = data['a_x0_hat']
        b_x0_hat           = data['b_x0_hat']
        c_x0_hat           = data['c_x0_hat']
        plane_rms_x0_hat   = data['plane_rms_x0_hat']
        total_rms_x0_hat   = data['total_rms_x0_hat']
        plane_ratio_x0_hat = data['plane_ratio_x0_hat']        


        a_x0_pre_c           = a_x0_pre.mean(axis=1)
        b_x0_pre_c           = b_x0_pre.mean(axis=1)
        c_x0_pre_c           = c_x0_pre.mean(axis=1)
        plane_rms_x0_pre_c   = plane_rms_x0_pre.mean(axis=1)
        plane_ratio_x0_pre_c = plane_ratio_x0_pre.mean(axis=1)

        a_x0_hat_c           = a_x0_hat.mean(axis=1)
        b_x0_hat_c           = b_x0_hat.mean(axis=1)
        c_x0_hat_c           = c_x0_hat.mean(axis=1)
        plane_rms_x0_hat_c   = plane_rms_x0_hat.mean(axis=1)
        plane_ratio_x0_hat_c = plane_ratio_x0_hat.mean(axis=1)

        a_xt_c           = a_xt.mean(axis=1)
        b_xt_c           = b_xt.mean(axis=1)
        c_xt_c           = c_xt.mean(axis=1)
        plane_rms_xt_c   = plane_rms_xt.mean(axis=1)
        plane_ratio_xt_c = plane_ratio_xt.mean(axis=1)

        b_x0_pre_c_stdB         = b_x0_pre.std(axis=1)
        c_x0_pre_c_stdB         = c_x0_pre.std(axis=1)
        plane_rms_x0_pre_c_stdB = plane_rms_x0_pre.std(axis=1)
        plane_ratio_x0_pre_c_stdB = plane_ratio_x0_pre.std(axis=1)

        if 'plane_a_x0_pre' in rgb_plots:
            self.rgb_plot(a_x0_pre_c, t,
                     title='plane_a_x0_pre',
                     out_png=str(out_dir / f"plane_a_x0_pre_{stem}_{sufix}.png"),
                     series_names=('a_x0_pre',),
                     ylim=None)

        if 'plane_b_x0_pre' in rgb_plots:
            self.rgb_plot(b_x0_pre_c, t,
                     title='plane_b_x0_pre',
                     out_png=str(out_dir / f"plane_b_x0_pre_{stem}_{sufix}.png"),
                     series_names=('b_x0_pre',),
                     ylim=None)

        if 'plane_c_x0_pre' in rgb_plots:
            self.rgb_plot(c_x0_pre_c, t,
                     title='plane_c_x0_pre',
                     out_png=str(out_dir / f"plane_c_x0_pre_{stem}_{sufix}.png"),
                     series_names=('c_x0_pre',),
                     ylim=None)

        if 'plane_rms_x0_pre' in rgb_plots:
            self.rgb_plot(plane_rms_x0_pre_c, t,
                     title='plane_rms_x0_pre',
                     out_png=str(out_dir / f"plane_rms_x0_pre_{stem}_{sufix}.png"),
                     series_names=('plane_rms_x0_pre',),
                     ylim=None)

        if 'plane_ratio_x0_pre' in rgb_plots:
            self.rgb_plot(plane_ratio_x0_pre_c, t,
                     title='plane_ratio_x0_pre',
                     out_png=str(out_dir / f"plane_ratio_x0_pre_{stem}_{sufix}.png"),
                     series_names=('plane_ratio_x0_pre',),
                     ylim=None)

        if 'plane_b_x0_hat' in rgb_plots:
            self.rgb_plot(b_x0_hat_c, t,
                     title='plane_b_x0_hat',
                     out_png=str(out_dir / f"plane_b_x0_hat_{stem}_{sufix}.png"),
                     series_names=('b_x0_hat',),
                     ylim=None)

        if 'plane_c_x0_hat' in rgb_plots:
            self.rgb_plot(c_x0_hat_c, t,
                     title='plane_c_x0_hat',
                     out_png=str(out_dir / f"plane_c_x0_hat_{stem}_{sufix}.png"),
                     series_names=('c_x0_hat',),
                     ylim=None)

        if 'plane_rms_x0_hat' in rgb_plots:
            self.rgb_plot(plane_rms_x0_hat_c, t,
                     title='plane_rms_x0_hat',
                     out_png=str(out_dir / f"plane_rms_x0_hat_{stem}_{sufix}.png"),
                     series_names=('plane_rms_x0_hat',),
                     ylim=None)

        if 'plane_ratio_x0_hat' in rgb_plots:
            self.rgb_plot(plane_ratio_x0_hat_c, t,
                     title='plane_ratio_x0_hat',
                     out_png=str(out_dir / f"plane_ratio_x0_hat_{stem}_{sufix}.png"),
                     series_names=('plane_ratio_x0_hat',),
                     ylim=None)


        lowfreq_delta = data['lowfreq_delta']        
        lowfreq_delta_c = lowfreq_delta.mean(axis=1)
        x0_hat_row_std_dc = data['x0_hat_row_std_dc']
        x0_hat_col_std_dc = data['x0_hat_col_std_dc']
        lowfreq_energy_xt = data['lowfreq_energy_xt']        
        lowfreq_energy_x_prev = data['lowfreq_energy_x_prev']        
        lowfreq_energy_x0_hat = data['lowfreq_energy_x0_hat']        
        lowfreq_delta_energy = data['lowfreq_delta_energy']        
        lowfreq_gain = data['lowfreq_gain']        
        cum_gain = data['cum_gain']

        e_no_dc_xt        = data['e_no_dc_xt']
        e_all_xt          = data['e_all_xt']
        dc_xt             = data['dc_xt']
        e_no_dc_x_prev    = data['e_no_dc_x_prev']
        e_all_x_prev      = data['e_all_x_prev']
        dc_x_prev         = data['dc_x_prev']

        

        if 'lowfreq_delta' in rgb_plots:
            self.rgb_plot(lowfreq_delta_c, t,
                     title='lowfreq_delta',
                     out_png=str(out_dir / f"lowfreq_delta_{stem}_{sufix}.png"),
                     series_names=('lowfreq_delta',),
                     ylim=None)
            
        self.rgb_plot((mu_x0_hat_c, mu_x0_pre_c, lowfreq_delta_c), t,
                 title="mu_x0_hat_c mu_x0_pre_c lowfreq_delta_c",
                 out_png=str(out_dir / f"mu_x0_hat_c_mu_x0_pre_c_lowfreq_delta_c_{stem}_{sufix}.png"),
                 series_names=("mu_x0_hat_c", "mu_x0_pre_c", "lowfreq_delta"),
                 ylim=None)

        self.rgb_plot((x0_hat_row_std_dc, x0_hat_col_std_dc, lowfreq_delta_c), t,
                 title="x0_hat_row_std_dc x0_hat_col_std_dc lowfreq_delta_c",
                 out_png=str(out_dir / f"x0_hat_row_std_dc_x0_hat_col_std_dc_lowfreq_delta_c_{stem}_{sufix}.png"),
                 series_names=("x0_hat_row_std_dc", "x0_hat_col_std_dc", "lowfreq_delta"),
                 ylim=None)

        #self.rgb_plot((lowfreq_energy_xt, lowfreq_energy_x_prev), t,
        #         title="lowfreq_energy_xt lowfreq_energy_x_prev",
        #         out_png=str(out_dir / f"lowfreq_energy_xt_lowfreq_energy_x_prev_{stem}_{sufix}.png"),
        #         series_names=('lowfreq_energy_xt', 'lowfreq_energy_x_prev'),
        #         ylim=None)

        self.rgb_plot((e_no_dc_xt, e_no_dc_x_prev), t,
                 title="lowfreq_energy_xt_x_prev_no_dc",
                 out_png=str(out_dir / f"lowfreq_energy_xt_x_prev_no_dc_{stem}_{sufix}.png"),
                 series_names=('e_no_dc_xt', 'e_no_dc_x_prev'),
                 ylim=None)

        self.rgb_plot((e_all_xt, e_all_x_prev), t,
                 title="lowfreq_energy_xt_x_prev_all",
                 out_png=str(out_dir / f"lowfreq_energy_xt_x_prev_all_{stem}_{sufix}.png"),
                 series_names=('e_all_xt', 'e_all_x_prev'),
                 ylim=None)

        self.rgb_plot((dc_xt, dc_x_prev), t,
                 title="lowfreq_energy_xt_x_prev_dc",
                 out_png=str(out_dir / f"lowfreq_energy_xt_x_prev_dc_{stem}_{sufix}.png"),
                 series_names=('dc_xt', 'dc_x_prev'),
                 ylim=None)

        self.rgb_plot((lowfreq_delta_energy, lowfreq_energy_x0_hat), t,
                 title="lowfreq_delta_energy lowfreq_energy_x0_hat",
                 out_png=str(out_dir / f"lowfreq_delta_energy_lowfreq_energy_x0_hat_{stem}_{sufix}.png"),
                 series_names=('lowfreq_delta_energy', 'lowfreq_energy_x0_hat'),
                 ylim=None)

        self.rgb_plot((lowfreq_gain, cum_gain), t,
                 title="lowfreq_gain cum_gain",
                 out_png=str(out_dir / f"lowfreq_gain_cum_gain_{stem}_{sufix}.png"),
                 series_names=('lowfreq_gain', 'cum_gain'),
                 ylim=None)

        #lowfreq_cos = data['lowfreq_cos']
        #plt.plot(t.tolist(), lowfreq_cos.tolist())
        #plt.gca().invert_xaxis()
        #plt.grid()
        #plt.show()


    def _luma_chroma(self, x, axis=-1):
        """
        x が色毎の平均(DC成分,形状(T,C)|(C,))の時に
        明るさ luma(形状(T,)|(1,)) と 色差 chroma(形状(T,C)|(C,)) を求める

        """
        luma   = x.mean(axis=axis, keepdims=True) 
        chroma = x - luma
        luma = np.squeeze(luma, axis=axis)
        return luma, chroma

    def _fit_affine_per_channel(self, x, y, axis=-2, eps=1e-12):
        """
        x, y が形状(T,B,C)/(B,C)のサンプル毎＆チャネル毎の場合に、
        xc, yc : x, y の平均からの乖離として、 
        yc = a * xc + b  の係数 a, b を求める
        axis=-2によりバッチ軸を指す

        """
        # x,y: (B,C)
        mx = x.mean(axis=axis, keepdims=True)
        my = y.mean(axis=axis, keepdims=True)

        xc = x - mx # バッチ平均からの個々のサンプルの隔たり
        yc = y - my # バッチ平均からの個々のサンプルの隔たり

        var_x  = (xc * xc).mean(axis=axis, keepdims=True)  # xcの分散
        cov_xy = (xc * yc).mean(axis=axis, keepdims=True) # xcとycの共分散

        a = cov_xy / (var_x + eps)
        b = my - a * mx

        var_y  = (yc * yc).mean(axis=axis, keepdims=True)  # ycの分散
        r = cov_xy / (np.sqrt(var_x * var_y) + eps)
        mse = ((y - (a * x + b)) ** 2).mean(axis=axis)

        a = np.squeeze(a, axis=axis)
        b = np.squeeze(b, axis=axis)
        r = np.squeeze(r, axis=axis)
        return a, b, r, mse

    def _global_mean_std_from_spatial(self, mu_bc, sigma_bc, axis=-2):
        """
        サンプル毎＆チャネル毎の平均と標準偏差から、
        バッチ毎＆チャネル毎の平均と標準偏差を得る
        mu_bc, sigma_bc : 形状 (T,B,C)|(B,C)
        mu_c,  sigma_c  : 形状 (T,C)  |(C,)

        """
        # 1. global mean
        mu_c = mu_bc.mean(axis=axis)
        # 2. reconstruct mean of squares
        mean_sq_bc = sigma_bc**2 + mu_bc**2
        # 3. global mean of squares
        mean_sq_c = mean_sq_bc.mean(axis=axis)
        # 4. variance and std
        var_c = mean_sq_c - mu_c**2
        sigma_c = np.sqrt(var_c)
        return mu_c, sigma_c


    def get_log(self, key=None):
        """ keyの指定するlogを格納したファイルから読出す """
        data = np.load(self.file)
        return data[key]

    def _fit_plane_per_channel(self, x, eps=1e-12):
        """
        x : shape (B,C,H,W)
        各サンプル・各チャネルごとに
            z(y,x) = a + b*x + c*y
        を最小二乗で当てる

        Returns
        -------
        a, b, c      : shape (B,C)
        plane_rms    : shape (B,C)
        total_rms    : shape (B,C)
        plane_ratio  : shape (B,C)   # plane_rms / total_rms
        """
        if x.ndim != 4:
            raise ValueError(f"_fit_plane_per_channel expects x.ndim==4, got {x.ndim}")

        B, C, H, W = x.shape
        dtype = x.dtype

        yy = np.linspace(-1.0, 1.0, H, dtype=dtype).reshape(H, 1)
        xx = np.linspace(-1.0, 1.0, W, dtype=dtype).reshape(1, W)

        ones = np.ones((H, W), dtype=dtype)
        X0 = ones.reshape(-1)                            # a
        X1 = np.broadcast_to(xx, (H, W)).reshape(-1)    # b*x
        X2 = np.broadcast_to(yy, (H, W)).reshape(-1)    # c*y

        # design matrix: (HW,3)
        A = np.stack([X0, X1, X2], axis=1)
        A_pinv = np.linalg.pinv(A)   # (3,HW)

        xf = x.reshape(B, C, H * W)  # (B,C,HW)
        coeff = np.matmul(xf, A_pinv.T)   # (B,C,3)

        a = coeff[..., 0]
        b = coeff[..., 1]
        c = coeff[..., 2]

        plane_flat = np.matmul(coeff, A.T)              # (B,C,HW)
        plane = plane_flat.reshape(B, C, H, W)

        plane_rms = np.sqrt((plane * plane).mean(axis=(2, 3)))
        total_rms = np.sqrt((x * x).mean(axis=(2, 3)))
        plane_ratio = plane_rms / (total_rms + eps)

        return a, b, c, plane_rms, total_rms, plane_ratio


    def _plane_stats_dict(self, prefix, x):
        """
        x から plane 統計量を計算し、
        append_log の辞書へそのまま **展開できる dict を返す
        """
        a, b, c, plane_rms, total_rms, plane_ratio = self._fit_plane_per_channel(x)
        return {
            f'a_{prefix}': a,
            f'b_{prefix}': b,
            f'c_{prefix}': c,
            f'plane_rms_{prefix}': plane_rms,
            f'total_rms_{prefix}': total_rms,
            f'plane_ratio_{prefix}': plane_ratio,
        }


    def lowfreq_energy(self, x, k=4, return_stats=False):
        """
        x : (B,C,H,W)

        k×k の低周波 DCT エネルギーを返す

        Parameters
        ----------
        x : ndarray, shape (B,C,H,W)
        k : int
            low-band size
        return_stats : bool, default False
            False: これまで通り DC除去後の energy だけ返す
            True : (energy_no_dc, energy_all, dc) を返す

        Returns
        -------
        if return_stats == False:
            energy_no_dc : (B,C)

        if return_stats == True:
            energy_no_dc : (B,C)
            energy_all   : (B,C)
            dc           : (B,C)
        """
        import numpy

        if np.__name__ == 'cupy':
            x = np.asnumpy(x)

        B, C, H, W = x.shape

        energy_no_dc = numpy.zeros((B, C), dtype=Config.dtype)

        if return_stats:
            energy_all = numpy.zeros((B, C), dtype=Config.dtype)
            dc = numpy.zeros((B, C), dtype=Config.dtype)

        for b in range(B):
            for c in range(C):

                img = x[b, c]

                # 2D DCT
                d = dct(dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

                # 低周波領域
                low = d[:k, :k]

                if return_stats:
                    dc[b, c] = low[0, 0]
                    energy_all[b, c] = numpy.sqrt((low**2).sum())

                # DC除去
                low_no_dc = low.copy()
                low_no_dc[0, 0] = 0

                energy_no_dc[b, c] = numpy.sqrt((low_no_dc**2).sum())

        if return_stats:
            return np.asarray(energy_no_dc), np.asarray(energy_all), np.asarray(dc)

        return np.asarray(energy_no_dc)

    def lowfreq_energy_bkup(self, x, k=4):
        """
        x : (B,C,H,W)

        k×k の低周波 DCT エネルギーを返す
        """
        import numpy

        if np.__name__=='cupy':
            x = np.asnumpy(x)
        
        B,C,H,W = x.shape

        e = numpy.zeros((B,C), dtype=Config.dtype)

        for b in range(B):
            for c in range(C):

                img = x[b,c]

                # 2D DCT
                d = dct(dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

                # 低周波領域
                low = d[:k,:k]

                # DC除去
                low[0,0] = 0

                e[b,c] = numpy.sqrt((low**2).sum())

        return np.asarray(e)

    def lowfreq_axis_stats(self, prefix, x):
        # x: BCHW
        x_dc = x - x.mean(axis=(2,3), keepdims=True)   # per-image per-channel DC除去

        row_prof = x_dc.mean(axis=3)   # B,C,H
        col_prof = x_dc.mean(axis=2)   # B,C,W

        row_std = row_prof.std(axis=2).mean(axis=0)
        col_std = col_prof.std(axis=2).mean(axis=0)

        return {
            f'{prefix}_row_std_dc' : row_std,
            f'{prefix}_col_std_dc' : col_std,
            }

    def lowfreq_vector(self, x, k=4):
        """
        x : (B,C,H,W)

        k×k の低周波 DCT 成分ベクトルを返す
        """
        import numpy

        if np.__name__=='cupy':
            x = np.asnumpy(x)

        B,C,H,W = x.shape

        vecs = []

        for b in range(B):
            for c in range(C):

                img = x[b,c]

                # 2D DCT
                d = dct(dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

                # 低周波領域
                low = d[:k,:k].copy()

                # DC除去
                low[0,0] = 0

                # ベクトル化
                vec = low.flatten()

                vecs.append(vec)

        return np.asarray(vecs)

    def lowfreq_cosine_theta(self, x, x_prev):
        v_t = self.lowfreq_vector(x)
        v_prev = self.lowfreq_vector(x_prev)
        dot = (v_t * v_prev).sum(axis=1)
        norm_t = np.sqrt((v_t**2).sum(axis=1))
        norm_prev = np.sqrt((v_prev**2).sum(axis=1))
        cos_theta = dot / (norm_t * norm_prev + 1e-12)
        return {'lowfreq_cos': cos_theta.mean()}

    def lowfreq_delta_energy_gain(self, x, x_prev): 
        lowfreq_energy_xt = self.lowfreq_energy(x)
        lowfreq_energy_x_prev = self.lowfreq_energy(x_prev)
        lowfreq_delta_energy = lowfreq_energy_x_prev - lowfreq_energy_xt
        lowfreq_gain = lowfreq_energy_x_prev / (lowfreq_energy_xt + 1e-12)
        return {
            'lowfreq_energy_xt'     : self.lowfreq_energy(x),
            'lowfreq_energy_x_prev' : self.lowfreq_energy(x_prev),
            'lowfreq_delta_energy'  : lowfreq_delta_energy,
            'lowfreq_gain'          : lowfreq_gain,
            }

class JacobianAnalizer:
    """ ddimのJacobian解析専用 """
    def __init__(self, diffuser, eps_fd=1e-3):
        self.diffuser = diffuser
        #self.make_probe = self.make_probe_dc
        self.make_probe = self.make_probe_lowfreq
        self.eps_fd = eps_fd
        self.logs = {'t': [], 'gain': [], 'mean_rgb': []}
        
    def make_probe_dc(self, x):
        c = np.random.randn(1, x.shape[1], 1, 1).astype(x.dtype)
        v = np.ones_like(x) * c
        v /= np.sqrt(np.sum(v * v)) + 1e-12
        return v

    def blur2d_simple(self, x, repeat=1):
        """ (B,C,H,W) に対する簡易3x3平滑化 """
        y = x.copy()
        for _ in range(repeat):
            y_pad = np.pad(y, ((0,0),(0,0),(1,1),(1,1)), mode='reflect')
            y = (
                y_pad[:, :, 0:-2, 0:-2] + 2*y_pad[:, :, 0:-2, 1:-1] + y_pad[:, :, 0:-2, 2:] +
                2*y_pad[:, :, 1:-1, 0:-2] + 4*y_pad[:, :, 1:-1, 1:-1] + 2*y_pad[:, :, 1:-1, 2:] +
                y_pad[:, :, 2:  , 0:-2] + 2*y_pad[:, :, 2:  , 1:-1] + y_pad[:, :, 2:  , 2:]
            ) / 16.0
        return y


    def make_probe_lowfreq(self, x, blur_repeat=4):
        """ 低周波方向の probe を作る """
        v = np.random.randn(*x.shape).astype(x.dtype)
        v = self.blur2d_simple(v, repeat=blur_repeat)
        v /= np.sqrt(np.sum(v * v)) + 1e-12
        return v  


    def measure_step(self, model, denoise_fn, x, t, t_prev):
        v = self.make_probe(x)
        x0 = denoise_fn(model, x, t, t_prev)
        x1 = denoise_fn(model, x + self.eps_fd * v, t, t_prev)
        dx = x1 - x0
        gain = np.sqrt(np.sum(dx * dx)) / self.eps_fd
        mean_rgb = dx.mean(axis=(0,2,3)) / self.eps_fd

        self.logs['t'].append(int(t))
        self.logs['gain'].append(float(gain))
        self.logs['mean_rgb'].append(mean_rgb)


def summarize_and_save(diffuser, epoch, out_dir: Path, sufix=None,
                       rgb_plots = (
                                    ##'plane_a_x0_pre',
                                    #'plane_b_x0_pre',
                                    #'plane_c_x0_pre',
                                    #'plane_rms_x0_pre',
                                    #'plane_ratio_x0_pre',
                                    #'lowfreq_delta',
                                     ),
                       #rgb_plots=(),#'mu_xt_c_and_mu_x_prev_c',
                                  #'mu_eps_c',
                                  #'a','b','r','mse',
                                  #'diff_x_prev_xt'),
                       rgb_scatters=()
                       ):
    diffuser.analizer.save_log(epoch, out_dir)
    diffuser.analizer.analize_and_draw(epoch, out_dir, sufix, rgb_plots, rgb_scatters)
    
def make_lowfreq_noise(x_shape,
                       dtype=Config.dtype,
                       cutoff_ratio=0.125,
                       normalize_std=True,
                       per_channel=True,
                       eps=1e-8):
    """
    低周波成分だけを残したノイズを生成する。

    Parameters
    ----------
    x_shape : tuple
        (B, C, H, W) を想定
    xp : module
        numpy または cupy
    dtype : str or dtype
        出力dtype
    cutoff_ratio : float
        通す低周波領域の半径比（0 < cutoff_ratio <= 0.5程度）
        小さいほど強いlow-passになる
    normalize_std : bool
        Trueなら low-pass後のノイズを std≈1 に再正規化する
    per_channel : bool
        True:
            各(B,C)ごとに個別ノイズを作る
        False:
            (B,1,H,W) で作ってチャネル方向へ複製
    eps : float
        ゼロ割防止

    Returns
    -------
    noise : ndarray
        shape = x_shape, dtype = dtype
    """
    B, C, H, W = x_shape

    if per_channel:
        base_shape = (B, C, H, W)
    else:
        base_shape = (B, 1, H, W)

    # 白色ノイズ
    noise = np.random.randn(*base_shape).astype(dtype)

    # 周波数グリッド（fftshift後の座標系に合わせる）
    fy = np.arange(H) - H // 2
    fx = np.arange(W) - W // 2
    yy, xx = np.meshgrid(fy, fx, indexing='ij')

    # 半径（正規化）
    ry = yy / max(H // 2, 1)
    rx = xx / max(W // 2, 1)
    rr = np.sqrt(ry * ry + rx * rx)

    # 円形 low-pass mask
    mask = (rr <= cutoff_ratio).astype(dtype)   # shape (H, W)
    mask = mask[None, None, :, :]               # broadcast用

    # FFT -> shift -> mask -> inverse shift -> IFFT
    F = np.fft.fft2(noise, axes=(-2, -1))
    F = np.fft.fftshift(F, axes=(-2, -1))
    F = F * mask
    F = np.fft.ifftshift(F, axes=(-2, -1))
    noise_low = np.fft.ifft2(F, axes=(-2, -1)).real.astype(dtype)

    # stdを揃える（重要）
    if normalize_std:
        std = noise_low.std(axis=(-2, -1), keepdims=True)
        noise_low = noise_low / (std + eps)

    # チャネル共通ノイズにしたい場合
    if not per_channel:
        noise_low = np.repeat(noise_low, C, axis=1)

    return noise_low.astype(dtype)
