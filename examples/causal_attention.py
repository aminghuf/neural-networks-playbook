import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def causal_self_attention(X, Wq, Wk, Wv):
    """X: (T, d_model). Returns (T, d_head) and the (T, T) attention weights."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                 # each (T, d_head)
    scores = Q @ K.T / np.sqrt(K.shape[-1])          # (T, T): how much token i attends to j
    mask = np.triu(np.ones_like(scores), k=1)        # 1 above the diagonal = the future
    scores = np.where(mask == 1, -np.inf, scores)    # a token may not look ahead
    weights = softmax(scores, axis=-1)               # each row sums to 1
    return weights @ V, weights

rng = np.random.default_rng(0)
T, d_model, d_head = 4, 8, 4
X = rng.normal(size=(T, d_model))
Wq, Wk, Wv = (rng.normal(size=(d_model, d_head)) / np.sqrt(d_model) for _ in range(3))
out, w = causal_self_attention(X, Wq, Wk, Wv)
print(out.shape)          # (4, 4)
print(np.round(w, 2))     # lower-triangular; row sums = 1
