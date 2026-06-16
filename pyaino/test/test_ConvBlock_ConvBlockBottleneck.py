from pyaino.Config import *
set_derivative(True)
from pyaino import stems_blocks_heads as sbh


def smoke_test_convblock(Block):
    print("=== ConvBlock smoke test ===")
        
    B = 2
    H = W = 8
    in_ch = 3
    mid_ch = 4
    out_ch = 4
    v_dim = 16

    x = np.random.randn(B, in_ch, H, W).astype(Config.dtype)
    v = np.random.randn(B, v_dim).astype(Config.dtype)

    # ------------------------------------------------------------
    # 1. residual=False, projなし
    # ------------------------------------------------------------
    print("[1] residual=False, proj=None")

    block = Block(out_ch, residual=False, proj=None)

    y = block(x, train=True)
    gy = np.random.randn(*y.shape).astype(Config.dtype)

    gx = block.backward(gy)

    print(" y.shape =", y.shape)
    print("gx.shape =", gx.shape)
    assert gx.shape == x.shape

    # ------------------------------------------------------------
    # 2. residual=True, in_ch == out_ch, identity residual
    # ------------------------------------------------------------
    print("[2] residual=True, identity shortcut")

    x2 = np.random.randn(B, out_ch, H, W).astype(Config.dtype)

    block = Block(out_ch, residual=True, proj=None)

    y = block(x2, train=True)
    gy = np.random.randn(*y.shape).astype(Config.dtype)

    gx = block.backward(gy)

    print(" y.shape =", y.shape)
    print("gx.shape =", gx.shape)
    assert gx.shape == x2.shape

    # ------------------------------------------------------------
    # 3. residual=True, in_ch != out_ch, shortcut 使用
    # ------------------------------------------------------------
    print("[3] residual=True, shortcut used")

    block = Block(out_ch, residual=True, proj=None)

    y = block(x, train=True)
    gy = np.random.randn(*y.shape).astype(Config.dtype)

    gx = block.backward(gy)

    print(" y.shape =", y.shape)
    print("gx.shape =", gx.shape)
    assert gx.shape == x.shape
    assert block.shortcut_used is True

    # ------------------------------------------------------------
    # 4. proj 使用
    # ------------------------------------------------------------
    print("[4] proj used")

    block = Block(out_ch, residual=True, proj=v_dim)

    y = block(x, v, train=True)
    gy = np.random.randn(*y.shape).astype(Config.dtype)

    gx, gv = block.backward(gy)

    print(" y.shape =", y.shape)
    print("gx.shape =", gx.shape)
    print("gv.shape =", gv.shape)

    assert gx.shape == x.shape
    assert gv.shape == v.shape
    assert block.proj_used is True

    print("=== all smoke tests passed ===")

def _to_scalar(a):
    if Config.np.__name__ == "cupy":
        return float(Config.np.asnumpy(a))
    return float(a)

def _max_abs_diff(a, b):
    return _to_scalar(np.max(np.abs(a - b)))

def check_backtrace_vs_backward(block, x_data, v_data=None, train=True, name="case"):
    print(f"\n[{name}]")

    # ------------------------------------------------------------
    # 1. y.backtrace(gy) 経路
    # ------------------------------------------------------------
    x = np.hdarray(x_data.copy())

    if v_data is not None:
        v = np.hdarray(v_data.copy())
        y = block(x, v, train=train)
    else:
        v = None
        y = block(x, train=train)

    gy = np.random.randn(*y.shape).astype(Config.dtype)

    y.backtrace(gy)

    gx_bt = x.grad.copy()
    gv_bt = v.grad.copy() if v is not None else None

    # ------------------------------------------------------------
    # 2. block.backward(gy) 経路
    # ------------------------------------------------------------
    x2 = np.hdarray(x_data.copy())

    if v_data is not None:
        v2 = np.hdarray(v_data.copy())
        y2 = block(x2, v2, train=train)
    else:
        v2 = None
        y2 = block(x2, train=train)

    ret = block.backward(gy)

    if isinstance(ret, tuple):
        gx_bw = ret[0]
        gv_bw = ret[1]
    else:
        gx_bw = ret
        gv_bw = None

    print(" y.shape  =", y.shape)
    print(" gx shape =", gx_bw.shape)
    print(" gx diff  =", _max_abs_diff(gx_bt, gx_bw))

    assert gx_bt.shape == gx_bw.shape
    assert _max_abs_diff(gx_bt, gx_bw) < 1e-5

    if v_data is not None:
        print(" gv shape =", gv_bw.shape)
        print(" gv diff  =", _max_abs_diff(gv_bt, gv_bw))

        assert gv_bt.shape == gv_bw.shape
        assert _max_abs_diff(gv_bt, gv_bw) < 1e-5

    print(" OK")


def smoke_test_convblock_backtrace_vs_backward(BlockClass):
    print("=== ConvBlock backtrace vs backward smoke test ===")

    B = 2
    Ih = Iw = 8
    in_ch = 3
    out_ch = 4
    v_dim = 16

    x = np.random.randn(B, in_ch, Ih, Iw).astype(Config.dtype)
    x_same = np.random.randn(B, out_ch, Ih, Iw).astype(Config.dtype)
    v = np.random.randn(B, v_dim).astype(Config.dtype)

    # 1. residual=False, projなし
    block = BlockClass(out_ch, residual=False, proj=None)
    check_backtrace_vs_backward(
        block, x, None,
        train=True,
        name="1 residual=False, proj=None",
    )

    # 2. residual=True, in_ch == out_ch
    block = BlockClass(out_ch, residual=True, proj=None)
    check_backtrace_vs_backward(
        block, x_same, None,
        train=True,
        name="2 residual=True, identity shortcut",
    )

    # 3. residual=True, in_ch != out_ch
    block = BlockClass(out_ch, residual=True, proj=None)
    check_backtrace_vs_backward(
        block, x, None,
        train=True,
        name="3 residual=True, shortcut used",
    )

    # 4. proj 使用
    block = BlockClass(out_ch, residual=True, proj=v_dim)
    check_backtrace_vs_backward(
        block, x, v,
        train=True,
        name="4 residual=True, proj used",
    )

    print("\n=== all backtrace vs backward smoke tests passed ===")


smoke_test_convblock(sbh.ConvBlock)
print('x'*50)
smoke_test_convblock(sbh.ConvBlockBottleneck)
print('x'*50)
smoke_test_convblock_backtrace_vs_backward(sbh.ConvBlock)
print('x'*50)
smoke_test_convblock_backtrace_vs_backward(sbh.ConvBlockBottleneck)
