import numpy as np

# ---------- 1. Data: two interleaved spirals (not linearly separable) ----------
def make_spirals(n=200, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.sqrt(rng.random(n)) * 3 * np.pi
    a = np.c_[theta * np.cos(theta), theta * np.sin(theta)]
    b = -a
    X = np.r_[a, b] + rng.normal(0, noise, (2 * n, 2))
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    X = (X - X.mean(0)) / X.std(0)          # standardize inputs
    return X, y

# ---------- 2. Building blocks ----------
def relu(z):        return np.maximum(0, z)
def relu_grad(z):   return (z > 0).astype(z.dtype)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)    # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(probs, y):
    return -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()

# ---------- 3. The model ----------
class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        # He initialization: good default for ReLU
        self.W = [rng.normal(0, np.sqrt(2 / m), (m, n)) for m, n in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n) for n in sizes[1:]]

    def forward(self, X):
        self.cache = [(None, X)]            # (pre-activation z, activation a)
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = softmax(z) if i == len(self.W) - 1 else relu(z)
            self.cache.append((z, a))
        return a

    def backward(self, y, l2=0.0):
        n = len(y)
        probs = self.cache[-1][1]
        delta = probs.copy()
        delta[np.arange(n), y] -= 1         # dL/dz for softmax + cross-entropy
        delta /= n
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            a_prev = self.cache[i][1]
            gW.insert(0, a_prev.T @ delta + l2 * self.W[i])
            gb.insert(0, delta.sum(0))
            if i > 0:
                z_prev = self.cache[i][0]
                delta = (delta @ self.W[i].T) * relu_grad(z_prev)
        return gW, gb

# ---------- 4. Adam optimizer ----------
class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)   # in-place update

# ---------- 5. Gradient check (always do this once when writing backprop) ----------
def grad_check(model, X, y, eps=1e-5):
    model.forward(X); gW, _ = model.backward(y)
    W = model.W[0]; i, j = 0, 0
    old = W[i, j]
    W[i, j] = old + eps; lp = cross_entropy(model.forward(X), y)
    W[i, j] = old - eps; lm = cross_entropy(model.forward(X), y)
    W[i, j] = old
    num = (lp - lm) / (2 * eps)
    return abs(num - gW[0][i, j]) / max(1e-12, abs(num) + abs(gW[0][i, j]))

# ---------- 6. Train ----------
X, y = make_spirals()
rng = np.random.default_rng(1)
idx = rng.permutation(len(X)); split = int(0.8 * len(X))
Xtr, ytr, Xte, yte = X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]

model = MLP([2, 64, 64, 2])
print("gradient check relative error:", grad_check(model, Xtr[:20], ytr[:20]))
opt = Adam(model.W + model.b, lr=1e-2)

for epoch in range(1, 501):
    perm = rng.permutation(len(Xtr))
    for k in range(0, len(Xtr), 32):                      # mini-batches of 32
        bi = perm[k:k + 32]
        model.forward(Xtr[bi])
        gW, gb = model.backward(ytr[bi], l2=1e-4)
        opt.step(model.W + model.b, gW + gb)
    if epoch % 100 == 0:
        p_tr = model.forward(Xtr); loss = cross_entropy(p_tr, ytr)
        acc_tr = (p_tr.argmax(1) == ytr).mean()
        acc_te = (model.forward(Xte).argmax(1) == yte).mean()
        print(f"epoch {epoch:3d}  loss {loss:.4f}  train acc {acc_tr:.3f}  test acc {acc_te:.3f}")
