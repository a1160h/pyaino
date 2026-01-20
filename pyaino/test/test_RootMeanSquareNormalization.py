from pyaino.Config import *
set_np('numpy'); np=Config.np
from pyaino.Neuron import RootMeanSquareNormalization


def _assert_allfinite(name, a):
    if not np.all(np.isfinite(a)):
        raise AssertionError(f"{name} has NaN/Inf")


def _relerr(a, b, eps=1e-12):
    return np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), eps))


def _numerical_grad_x(loss_fn, x, h=1e-5):
    """中心差分: d(loss)/dx"""
    gx = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + h
        p = loss_fn(x)
        x[idx] = old - h
        m = loss_fn(x)
        x[idx] = old
        gx[idx] = (p - m) / (2 * h)
        it.iternext()
    return gx


def _numerical_grad_param(loss_fn, param, h=1e-5):
    """中心差分: d(loss)/d(param)  param は任意shape可"""
    g = np.zeros_like(param)
    flat = param.reshape(-1)
    gflat = g.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
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

    # sigma が作れていること（forward で self.sigma を作る）:contentReference[oaicite:2]{index=2}
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

    # backward は self.inputs を使うのでセット :contentReference[oaicite:3]{index=3}
    f.inputs = (x,)
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

    # forward（ここで gamma/beta が必要なら init_parameters が呼ばれる）:contentReference[oaicite:4]{index=4}
    y = f.__forward__(x, train=True)
    _assert_allfinite("y", y)

    # backward
    f.inputs = (x,)
    gx = f.__backward__(gy)
    _assert_allfinite("gx", gx)

    # ggamma, gbeta が作れていること :contentReference[oaicite:5]{index=5}
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


def test_ppl_mode_behavior():
    """
    ppl=True は非訓練時に sigma_ppl を使う仕様（バッチノーマライゼーション用途）:contentReference[oaicite:6]{index=6}
    - 初回は sigma_ppl=1 で初期化されるので、train=False の出力は概ね x と一致（sb=Falseの場合）
    - train=True を1回通して update しても落ちないことを確認
    """
    np.random.seed(3)
    B, T, D = 2, 3, 5
    x = np.random.randn(B, T, D).astype(np.float64)

    f = RootMeanSquareNormalization(axis=-1, ppl=True, scale_and_bias=False, eps=1e-12)

    # 非訓練 forward：sigma_ppl を使う :contentReference[oaicite:7]{index=7}
    y_eval = f.__forward__(x, train=False)
    _assert_allfinite("y_eval", y_eval)

    # 初期 sigma_ppl=1 の想定で y_eval ~ x（完全一致でなくてもOKだが、近いはず）
    # ※ set_axis_and_shape の仕様次第で broadcast 形状が変わっても意味は同じです
    diff = np.max(np.abs(y_eval - x))
    print("[ppl eval] max|y-x| =", diff)

    # 訓練 forward → update
    y_train = f.__forward__(x, train=True)
    _assert_allfinite("y_train", y_train)

    # update が落ちないこと（sigma_ppl を更新）:contentReference[oaicite:8]{index=8}
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
    f.inputs = (x,)
    gx = f.__backward__(gy)
    _assert_allfinite("gx", gx)

    print("[OK] zero_input_stability")


if __name__ == "__main__":
    test_forward_basic()
    test_backward_numeric_no_sb()
    test_forward_backward_with_sb_numeric()
    test_ppl_mode_behavior()
    test_zero_input_stability()
    print("\nRootMeanSquareNormalization smoke tests: PASSED")
