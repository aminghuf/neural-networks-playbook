import math
x, t = 1.0, 1.0
w1, b1, w2, b2 = 0.5, 0.0, 2.0, 0.0
lr = 0.1
sig = lambda z: 1 / (1 + math.exp(-z))
z1 = w1 * x + b1
h = sig(z1)
y = w2 * h + b2
L = 0.5 * (y - t) ** 2
print(f"z1={z1:.4f} h={h:.4f} y={y:.4f} L={L:.4f}")
dy = y - t
dw2 = dy * h
db2 = dy
dh = dy * w2
dz1 = dh * h * (1 - h)
dw1 = dz1 * x
db1 = dz1
print(f"dy={dy:.4f} dw2={dw2:.4f} db2={db2:.4f} dh={dh:.4f} dz1={dz1:.4f} dw1={dw1:.4f} db1={db1:.4f}")
w1 -= lr * dw1; b1 -= lr * db1; w2 -= lr * dw2; b2 -= lr * db2
print(f"new w1={w1:.4f} b1={b1:.4f} w2={w2:.4f} b2={b2:.4f}")
h = sig(w1 * x + b1); y = w2 * h + b2; L2 = 0.5 * (y - t) ** 2
print(f"new y={y:.4f} L={L2:.4f}")
# numeric gradient check for w1
eps = 1e-6
def loss(w1_):
    h_ = sig(w1_ * x + 0.0); return 0.5 * (2.0 * h_ - t) ** 2
print("numeric dw1", (loss(0.5 + eps) - loss(0.5 - eps)) / (2 * eps))
