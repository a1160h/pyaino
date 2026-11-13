from pyaino.Config import *
from pyaino.Neuron import LowRankEmbedding 

l, n, m = 23, 12, 4

model = LowRankEmbedding(l, n, m, optimize='SGD')

x = np.array([[2, 8, 2], [8, 3, 17]], dtype='int32')
mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=Config.dtype)

y = model.forward(x, mask=mask)
A = model.embed.parameters.w
B = model.project.parameters.w

assert y.shape == x.shape + (n,)
np.testing.assert_allclose(y, (A[x] @ B) * mask[..., None], rtol=1e-6, atol=1e-6)

# Analytic gradients for a random upstream gradient; repeated indices must accumulate.
gy = np.random.normal(0.0, 1.0, size=y.shape).astype(Config.dtype)

gz = (gy @ B.T) * mask[..., None]
gA = np.zeros_like(A)
np.add.at(gA, x, gz)
gB = (A[x] * mask[..., None]).reshape(-1, m).T @ gy.reshape(-1, n)
model.backward(gy)
np.testing.assert_allclose(model.embed.parameters.grad_w, gA, rtol=1e-5, atol=2e-6)
np.testing.assert_allclose(model.project.parameters.grad_w, gB, rtol=1e-5, atol=2e-6)

A0, B0 = A.copy(), B.copy()
model.update(eta=0.001)
assert not np.array_equal(model.embed.parameters.w, A0)
assert not np.array_equal(model.project.parameters.w, B0)
print('PASS: forward, masks, repeated-token gradients, update')

