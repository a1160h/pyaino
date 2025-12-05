from pyaino.Config import *

from pyaino.Neuron import (
                           BatchNormalization,
                           BatchNorm2d,
                           LayerNormalization,
                           LayerNorm2d,
                           InstanceNorm2d,
                           )

def summarize_stats(name, y, norm_axes):
    """平均・分散を出してざっくり確認用"""
    mean = y.mean(axis=norm_axes)
    var  = y.var(axis=norm_axes)
    print(f"[{name}] mean shape={mean.shape}, var shape={var.shape}")
    print(f"  mean (first few) = {mean.reshape(-1)[:5]}")
    print(f"  var  (first few) = {var.reshape(-1)[:5]}")
    return mean, var


def assert_close(name, arr, target, atol=1e-4, rtol=1e-4):
    if not np.allclose(arr, target, atol=atol, rtol=rtol):
        diff = np.abs(arr - target).max()
        raise AssertionError(f"{name}: not close to {target}, max diff = {diff}")
    print(f"{name}: OK (max diff = {np.abs(arr - target).max():.3e})")


def test_batch_normalization_fc():
    """
    BatchNormalization:
      - 形状 (B, D)
      - 正規化軸: バッチ0
      - scale_and_bias軸: 特徴軸 (1)
    """
    print("=== test_batch_normalization_fc ===")
    B, D = 16, 10
    x = np.random.randn(B, D).astype(np.float32)

    # scale_and_bias=True でガンマ/ベータありの想定
    norm = BatchNormalization(scale_and_bias=False, ppl=False)

    # GeneralNormalizationBase が __call__ で順伝播すると仮定
    y = norm(x, train=True)  # 必要なら norm(x, training=True) などに調整

    # バッチ軸0で平均0, 分散1になっているか
    mean, var = summarize_stats("BatchNormalization FC", np.asarray(y), norm_axes=(0,))

    assert_close("BatchNormalization FC mean ~ 0", mean, 0.0, atol=1e-5, rtol=1e-5)
    assert_close("BatchNormalization FC var  ~ 1", var, 1.0, atol=1e-3, rtol=1e-3)


def test_batchnorm2d():
    """
    BatchNorm2d:
      - 形状 (B, C, H, W)
      - 正規化軸: バッチ + 空間 (0, 2, 3)
      - scale_and_bias軸: チャンネル (1)
    """
    print("=== test_batchnorm2d ===")
    B, C, H, W = 8, 3, 5, 7
    x = np.random.randn(B, C, H, W).astype(np.float32)

    norm = BatchNorm2d(scale_and_bias=True)

    y = norm(x, train=True)  # 必要なら norm(x, training=True) に変更

    # (B, H, W) 方向に対してチャンネルごとに mean≈0, var≈1 か
    y_np = np.asarray(y)
    mean, var = summarize_stats("BatchNorm2d", y_np, norm_axes=(0, 2, 3))

    assert_close("BatchNorm2d mean ~ 0", mean, 0.0, atol=1e-5, rtol=1e-5)
    assert_close("BatchNorm2d var  ~ 1", var, 1.0, atol=1e-3, rtol=1e-3)


def test_layernorm2d():
    """
    LayerNorm2d:
      - 形状 (B, C, H, W)
      - 正規化軸: 特徴全体 (C, H, W) = (-3, -2, -1)
      - scale_and_bias軸も同じ (exclude=False)
      → サンプル(B)ごとに、「自分自身の (C,H,W)」で標準化されるはず
    """
    print("=== test_layernorm2d ===")
    B, C, H, W = 4, 3, 5, 7
    x = np.random.randn(B, C, H, W).astype(np.float32)

    norm = LayerNorm2d(scale_and_bias=True)

    y = norm(x)

    y_np = np.asarray(y)
    # サンプルごとに (C,H,W) で mean≈0, var≈1 のはず
    # 形状は (B,) になるように axis を指定
    mean = y_np.mean(axis=(1, 2, 3))
    var  = y_np.var(axis=(1, 2, 3))

    print(f"[LayerNorm2d] mean per sample = {mean}")
    print(f"[LayerNorm2d] var  per sample = {var}")

    assert_close("LayerNorm2d mean ~ 0", mean, 0.0, atol=1e-5, rtol=1e-5)
    assert_close("LayerNorm2d var  ~ 1", var, 1.0, atol=1e-3, rtol=1e-3)

def test_instancenorm2d():
    print("=== test_instance_norm2d ===")

    B, C, H, W = 4, 3, 8, 8
    x = np.random.randn(B, C, H, W).astype(np.float32)

    norm = InstanceNorm2d(scale_and_bias=False)
    y = norm(x)  # train フラグは不要のはず

    y_np = np.asarray(y)

    # 各 (b,c) について H,W 方向で mean≈0, var≈1 か？
    mean = y_np.mean(axis=(-2, -1))  # shape: (B, C)
    var  = y_np.var(axis=(-2, -1))   # shape: (B, C)

    print(f"[InstanceNorm2d] mean per sample = {mean}")
    print(f"[InstanceNorm2d] var  per sample = {var}")

    assert_close("InstanceNorm2d mean ~ 0", mean, 0.0, atol=1e-5, rtol=1e-5)
    assert_close("InstanceNorm2d var  ~ 1", var, 1.0, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    # 必要なテストだけ順次呼び出してください
    test_batch_normalization_fc()
    test_batchnorm2d()
    test_layernorm2d()
    test_instancenorm2d()
    print("All forward smoke tests passed.")



