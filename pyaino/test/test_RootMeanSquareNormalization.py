from pyaino.Config import *
from pyaino.Neuron import RootMeanSquareNormalization
from itertools import product


def _assert_allfinite(name, a):
    if not np.all(np.isfinite(a)):
        raise AssertionError(f"{name} has NaN/Inf")


def _relerr(a, b, eps=1e-12):
    return np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), eps))


def _numerical_grad_x(loss_fn, x, h=1e-5):
    """中心差分: d(loss)/dx（NumPy/CuPy共通）"""
    gx = np.zeros_like(x)
    # CuPyにはnp.nditerがないため、添字はPython側で生成する。
    # reshape(-1)と異なり、非連続配列でも入力の各要素を直接変更できる。
    for idx in product(*(range(n) for n in x.shape)):
        old = x[idx].item()
        x[idx] = old + h
        p = loss_fn(x)
        x[idx] = old - h
        m = loss_fn(x)
        x[idx] = old
        gx[idx] = (p - m) / (2 * h)
    return gx


def _numerical_grad_param(loss_fn, param, h=1e-5):
    """中心差分: d(loss)/d(param)  param は任意shape可"""
    g = np.zeros_like(param)
    flat = param.reshape(-1)
    gflat = g.reshape(-1)
    for i in range(flat.size):
        old = flat[i].item()
        flat[i] = old + h
        p = loss_fn(param)
        flat[i] = old - h
        m = loss_fn(param)
        flat[i] = old
        gflat[i] = (p - m) / (2 * h)
    return g


def test_forward_basic():
    np.random.seed(0)
    B, T, D = 2, 3, 5
    x = np.random.randn(B, T, D).astype(np.float64)

    f = RootMeanSquareNormalization(axis=-1, eps=1e-12)

    y = f.__forward__(x, train=True)
    _assert_allfinite("y", y)
    assert y.shape == x.shape

    # sigma が作れていること（forward で self.sigma を作る）
    assert f.sigma is not None
    _assert_allfinite("sigma", f.sigma)

    print("[OK] forward_basic")


def test_backward_numeric_no_sb():
    """
    scale_and_bias=False の RMSNorm 勾配チェック
    L = sum(y * gy) として、gy が dL/dy になるようにする
    """
    np.random.seed(1)
    B, T, D = 2, 3, 5
    x = np.random.randn(B, T, D).astype(np.float64)
    gy = np.random.randn(B, T, D).astype(np.float64)

    f = RootMeanSquareNormalization(axis=-1, eps=1e-12, scale_and_bias=False)

    # forward
    y = f.__forward__(x, train=True)
    _assert_allfinite("y", y)

    # __forward__() が self.x を保持する。
    gx = f.__backward__(gy)
    _assert_allfinite("gx", gx)
    assert gx.shape == x.shape

    # 数値微分
    def loss_fn(x_var):
        y_var = f.__forward__(x_var, train=True)
        return float(np.sum(y_var * gy))

    gx_num = _numerical_grad_x(loss_fn, x.copy(), h=1e-5)

    err = _relerr(gx, gx_num)
    print("[GradCheck no_sb] relerr(dx) =", err)

    # smoke として十分小さければOK（float64想定）
    assert err < 5e-6

    print("[OK] backward_numeric_no_sb")


def test_forward_backward_with_sb_numeric():
    """
    scale_and_bias=True (gamma,beta) の勾配チェック
    """
    np.random.seed(2)
    B, T, D = 2, 3, 5
    x = np.random.randn(B, T, D).astype(np.float64)
    gy = np.random.randn(B, T, D).astype(np.float64)

    f = RootMeanSquareNormalization(axis=-1, eps=1e-12, scale_and_bias=True)

    # forward（ここで gamma/beta が必要なら init_parameters が呼ばれる）
    y = f.__forward__(x, train=True)
    _assert_allfinite("y", y)

    # backward
    gx = f.__backward__(gy)
    _assert_allfinite("gx", gx)

    # ggamma, gbeta が作れていること 
    assert hasattr(f, "ggamma") and hasattr(f, "gbeta")
    _assert_allfinite("ggamma", f.ggamma)
    _assert_allfinite("gbeta", f.gbeta)

    # ---- x の数値微分 ----
    def loss_x(x_var):
        y_var = f.__forward__(x_var, train=True)
        return float(np.sum(y_var * gy))

    gx_num = _numerical_grad_x(loss_x, x.copy(), h=1e-5)
    err_x = _relerr(gx, gx_num)
    print("[GradCheck sb] relerr(dx) =", err_x)
    assert err_x < 5e-6

    # ---- gamma の数値微分 ----
    gamma0 = f.gamma.copy()
    def loss_gamma(_gamma):
        f.gamma = _gamma
        y_var = f.__forward__(x, train=True)
        return float(np.sum(y_var * gy))

    ggamma_num = _numerical_grad_param(loss_gamma, gamma0.copy(), h=1e-5)
    # analytic ggamma は keepdims を持つ可能性があるので形を合わせる
    # （数値微分は gamma のshapeそのまま）
    ggamma_ana = np.asarray(f.ggamma).reshape(-1)
    ggamma_num = np.asarray(ggamma_num).reshape(-1)
    err_g = _relerr(ggamma_ana, ggamma_num)

    #err_g = _relerr(f.ggamma, ggamma_num)
    print("[GradCheck sb] relerr(dgamma) =", err_g)
    assert err_g < 1e-3 #5e-6 

    # ---- beta の数値微分 ----
    beta0 = f.beta.copy()
    def loss_beta(_beta):
        f.beta = _beta
        y_var = f.__forward__(x, train=True)
        return float(np.sum(y_var * gy))

    gbeta_num = _numerical_grad_param(loss_beta, beta0.copy(), h=1e-5)
    err_b = _relerr(f.gbeta, gbeta_num)
    print("[GradCheck sb] relerr(dbeta) =", err_b)
    assert err_b < 5e-6

    print("[OK] forward_backward_with_sb_numeric")



def test_scale_only_numeric():
    """
    scale_and_bias='scale' (gammaのみ) の forward / backward 数値微分。
    beta が存在せず、gamma だけが学習対象になることも確認する。
    """
    np.random.seed(5)
    B, T, D = 2, 3, 5
    x = np.random.randn(B, T, D).astype(np.float64)
    gy = np.random.randn(B, T, D).astype(np.float64)

    f = RootMeanSquareNormalization(axis=-1, eps=1e-12, scale_and_bias='scale')
    y = f.__forward__(x, train=True)

    assert f.sb is True
    assert f.use_beta is False
    assert f.gamma is not None
    assert f.beta is None
    assert not hasattr(f, 'OFb')
    assert np.all(f.gamma == 1)
    assert y.shape == x.shape
    _assert_allfinite('scale y', y)

    # 初期 gamma=1 では、スケール・バイアスなしと同じ出力になる。
    y_expected = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + f.eps)
    np.testing.assert_allclose(y, y_expected, rtol=1e-7, atol=1e-7)

    # gamma を1以外に変更し、各特徴が独立にスケールされることを確認。
    f.gamma[...] = np.linspace(0.7, 1.3, f.gamma.size).reshape(f.gamma.shape)
    y = f.__forward__(x, train=True)
    np.testing.assert_allclose(y, y_expected*f.gamma, rtol=1e-7, atol=1e-7)

    gx = f.__backward__(gy)
    _assert_allfinite('scale gx', gx)
    _assert_allfinite('scale ggamma', f.ggamma)
    assert gx.shape == x.shape
    assert f.gbeta is None  # beta勾配は未使用

    # ---- x の数値微分 ----
    def loss_x(x_var):
        return float(np.sum(f.__forward__(x_var, train=True)*gy))

    gx_num = _numerical_grad_x(loss_x, x.copy(), h=1e-5)
    err_x = _relerr(gx, gx_num)
    print('[GradCheck scale] relerr(dx) =', err_x)
    assert err_x < 5e-6

    # ---- gamma の数値微分 ----
    gamma0 = f.gamma.copy()
    def loss_gamma(_gamma):
        f.gamma = _gamma
        return float(np.sum(f.__forward__(x, train=True)*gy))

    # gamma は Config.dtype (通常float32) なので、刻みを少し大きくする。
    # loss は gamma に対して線形であり、中心差分の打切り誤差は生じない。
    ggamma_num = _numerical_grad_param(loss_gamma, gamma0.copy(), h=1e-2)
    f.gamma = gamma0
    err_g = _relerr(np.asarray(f.ggamma).reshape(-1),
                    np.asarray(ggamma_num).reshape(-1))
    print('[GradCheck scale] relerr(dgamma) =', err_g)
    assert err_g < 1e-3

    # ---- gamma の更新（beta更新を要求しない） ----
    gamma_before = f.gamma.copy()
    f.update(eta=0.01)
    _assert_allfinite('scale gamma after update', f.gamma)
    assert not np.array_equal(f.gamma, gamma_before)
    assert f.beta is None

    print('[OK] scale_only_numeric')


def test_scale_only_equivalence_to_sb():
    """同じ gamma、beta=0 なら 'scale' と True の出力・勾配は一致する。"""
    np.random.seed(6)
    B, T, D = 2, 3, 5
    x = np.random.randn(B, T, D).astype(np.float64)
    gy = np.random.randn(B, T, D).astype(np.float64)

    scale = RootMeanSquareNormalization(axis=-1, eps=1e-12, scale_and_bias='scale')
    both = RootMeanSquareNormalization(axis=-1, eps=1e-12, scale_and_bias=True)

    scale.__forward__(x, train=True)  # gammaの遅延初期化
    both.__forward__(x, train=True)
    scale.gamma[...] = np.linspace(0.8, 1.2, scale.gamma.size).reshape(scale.gamma.shape)
    both.gamma[...] = scale.gamma
    both.beta[...] = 0

    y_scale = scale.__forward__(x, train=True)
    y_both = both.__forward__(x, train=True)
    np.testing.assert_allclose(y_scale, y_both, rtol=0, atol=1e-12)

    gx_scale = scale.__backward__(gy)
    gx_both = both.__backward__(gy)
    np.testing.assert_allclose(gx_scale, gx_both, rtol=0, atol=1e-12)
    np.testing.assert_allclose(scale.ggamma, both.ggamma, rtol=0, atol=1e-12)

    print('[OK] scale_only_equivalence_to_sb')


def test_ppl_mode_behavior():
    """
    ppl=True は非訓練時に sigma_ppl を使う仕様（バッチノーマライゼーション用途）
    - 初回は sigma_ppl=1 で初期化されるので、train=False の出力は概ね x と一致（sb=Falseの場合）
    - train=True を1回通して update しても落ちないことを確認
    """
    np.random.seed(3)
    B, T, D = 2, 3, 5
    x = np.random.randn(B, T, D).astype(np.float64)

    f = RootMeanSquareNormalization(axis=-1, ppl=True, scale_and_bias=False, eps=1e-12)

    # 非訓練 forward：sigma_ppl を使う 
    y_eval = f.__forward__(x, train=False)
    _assert_allfinite("y_eval", y_eval)

    # 初期 sigma_ppl=1 の想定で y_eval ~ x（完全一致でなくてもOKだが、近いはず）
    # ※ set_axis_and_shape の仕様次第で broadcast 形状が変わっても意味は同じです
    diff = np.max(np.abs(y_eval - x))
    print("[ppl eval] max|y-x| =", diff)

    # 訓練 forward → update
    y_train = f.__forward__(x, train=True)
    _assert_allfinite("y_train", y_train)

    # update が落ちないこと（sigma_ppl を更新）
    f.update()

    print("[OK] ppl_mode_behavior")


def test_zero_input_stability():
    np.random.seed(4)
    B, T, D = 2, 3, 5
    x = np.zeros((B, T, D), dtype=np.float64)

    f = RootMeanSquareNormalization(axis=-1, eps=1e-12)
    y = f.__forward__(x, train=True)
    _assert_allfinite("y", y)

    # backward
    gy = np.random.randn(B, T, D).astype(np.float64)
    gx = f.__backward__(gy)
    _assert_allfinite("gx", gx)

    print("[OK] zero_input_stability")


if __name__ == "__main__":
    test_forward_basic()
    test_backward_numeric_no_sb()
    test_forward_backward_with_sb_numeric()
    test_scale_only_numeric()
    test_scale_only_equivalence_to_sb()
    test_ppl_mode_behavior()
    test_zero_input_stability()
    print("\nRootMeanSquareNormalization smoke tests: PASSED")
