"""Finite-difference check of the hand-written backprop (float64)."""
import numpy as np

from rnn_attn import SeqModel, make_split

rng = np.random.default_rng(0)
data = make_split(6, 7, rng)
X, y = data["X"].astype(np.float64), data["y"]

for name, kw in [("rnn", dict(attention=False)),
                 ("attn-bilinear", dict(attention=True, bilinear=True)),
                 ("attn-dot", dict(attention=True, bilinear=False))]:
    m = SeqModel(hidden=5, seed=1, dtype=np.float64, **kw)
    _, g = m.loss_and_grads(X, y)
    worst = 0.0
    for k, w in m.p.items():
        flat = w.reshape(-1)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + 1e-6
            lp, _ = m.loss_and_grads(X, y)
            flat[i] = old - 1e-6
            lm, _ = m.loss_and_grads(X, y)
            flat[i] = old
            num = (lp - lm) / 2e-6
            ana = g[k].reshape(-1)[i]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    print(f"{name:14s} max relative gradient error = {worst:.2e}",
          "OK" if worst < 1e-5 else "FAIL")
