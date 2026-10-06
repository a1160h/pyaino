from pyaino.Config import *
from pyaino.Neuron import AttentionUnit, QueryChunkAttentionUnit


# ============================================================
# QueryChunkAttentionUnit GQA / MQA equivalence test
# ============================================================

set_seed(200)
np = Config.np

print("np           =", np.__name__)
print("Config.dtype =", Config.dtype)

B = 2
T = 7
H = 4

heads = [
    (4, 2),       # GQA
    (4, 1),       # MQA
]

chunk_sizes = [1, 2, 3, 4, 7]

atol = 1e-4
rtol = 1e-4

print("shape B,T,H  =", (B, T, H))
print("heads        =", heads)
print("chunk sizes  =", chunk_sizes)
print("atol / rtol  =", atol, rtol)


for head in heads:

    hq, hv = head
    hk = hv

    Cq = hq * H
    Ck = hk * H
    Cv = hv * H

    print()
    print("=" * 72)
    print("head =", head)
    print("Cq, Ck, Cv =", Cq, Ck, Cv)
    print("=" * 72)

    # --------------------------------------------------------
    # 同じ入力で全条件を比較
    # --------------------------------------------------------

    np.random.seed(200)

    q0 = np.random.randn(B, T, Cq).astype(Config.dtype)
    k0 = np.random.randn(B, T, Ck).astype(Config.dtype)
    v0 = np.random.randn(B, T, Cv).astype(Config.dtype)

    # Attention出力幅は hq * Hv
    gy0 = np.random.randn(B, T, hq * H).astype(Config.dtype)


    for rope in [False, True]:

        print()
        print("-" * 72)
        print("rope =", rope)
        print("-" * 72)

        # ====================================================
        # 通常 AttentionUnit を基準値とする
        # ====================================================

        unit0 = AttentionUnit(
            head=head,
            causality=True,
            scale=True,
            rope=rope,
            regularizer=None,
        )

        q = q0.copy()
        k = k0.copy()
        v = v0.copy()

        y0 = unit0.forward(
            q, k, v,
            mask=None,
            dropout=0.0,
        )

        gq0, gk0, gv0 = unit0.backward(gy0.copy())


        # ====================================================
        # QueryChunkAttentionUnit
        # ====================================================

        for chunk_size in chunk_sizes:

            print()
            print("--- chunk_size =", chunk_size, "---")

            unit1 = QueryChunkAttentionUnit(
                head=head,
                chunk_size=chunk_size,
                causality=True,
                scale=True,
                rope=rope,
                regularizer=None,
            )

            q = q0.copy()
            k = k0.copy()
            v = v0.copy()

            y1 = unit1.forward(
                q, k, v,
                mask=None,
                dropout=0.0,
            )

            gq1, gk1, gv1 = unit1.backward(gy0.copy())


            # ------------------------------------------------
            # forward
            # ------------------------------------------------

            err = float(np.max(np.abs(y1 - y0)))
            ok = np.allclose(y1, y0, atol=atol, rtol=rtol)

            print(
                f"forward y          max error = {err:.8e}",
                "OK" if ok else "NG"
            )

            assert ok, (
                f"forward mismatch: "
                f"head={head}, rope={rope}, chunk={chunk_size}"
            )


            # ------------------------------------------------
            # backward q
            # ------------------------------------------------

            err = float(np.max(np.abs(gq1 - gq0)))
            ok = np.allclose(gq1, gq0, atol=atol, rtol=rtol)

            print(
                f"backward gq        max error = {err:.8e}",
                "OK" if ok else "NG"
            )

            assert ok, (
                f"gq mismatch: "
                f"head={head}, rope={rope}, chunk={chunk_size}"
            )


            # ------------------------------------------------
            # backward k
            # ------------------------------------------------

            err = float(np.max(np.abs(gk1 - gk0)))
            ok = np.allclose(gk1, gk0, atol=atol, rtol=rtol)

            print(
                f"backward gk        max error = {err:.8e}",
                "OK" if ok else "NG"
            )

            assert ok, (
                f"gk mismatch: "
                f"head={head}, rope={rope}, chunk={chunk_size}"
            )


            # ------------------------------------------------
            # backward v
            # ------------------------------------------------

            err = float(np.max(np.abs(gv1 - gv0)))
            ok = np.allclose(gv1, gv0, atol=atol, rtol=rtol)

            print(
                f"backward gv        max error = {err:.8e}",
                "OK" if ok else "NG"
            )

            assert ok, (
                f"gv mismatch: "
                f"head={head}, rope={rope}, chunk={chunk_size}"
            )


print()
print("=" * 72)
print("All GQA / MQA AttentionUnit / QueryChunkAttentionUnit tests passed.")
print("=" * 72)
