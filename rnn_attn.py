"""Vanilla RNN vs. RNN + attention on a marked-value memory task.

Pure NumPy implementation (manual backprop through time) so the whole study is
reproducible without a deep-learning framework.

Task ("marked value recall")
    A sequence of L steps. Every step has a random value v_t in {0..9} (one-hot)
    and a marker flag. Exactly one step k carries marker=1. The label is v_k.
    The marked step k is uniform over 0..L-1, so the memory distance
    d = (L-1) - k is uniform over 0..L-1.

Model A (vanilla RNN):  h_t = tanh(Wxh x_t + Whh h_{t-1} + b);  logits = Wo h_T + bo
Model B (RNN + attn):   same encoder, keep H=[h_1..h_T]; query q = h_T;
                        e_t = (q Wa) . h_t ; alpha = softmax(e); c = sum_t alpha_t h_t;
                        logits = Wo c + bo   (only c is used for the prediction)
"""
import time

import numpy as np

C = 10        # number of value classes
D = C + 1     # one-hot value + marker flag


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def make_split(n, L, rng):
    vals = rng.integers(0, C, size=(n, L))
    k = rng.integers(0, L, size=n)
    X = np.zeros((n, L, D), dtype=np.float32)
    X[np.arange(n)[:, None], np.arange(L)[None, :], vals] = 1.0
    X[np.arange(n), k, C] = 1.0
    y = vals[np.arange(n), k]
    return dict(X=X, y=y, k=k, vals=vals)


def make_dataset(L, n_train=10000, n_val=2000, n_test=2000):
    """Same data for every model: the generator is seeded by L only."""
    rng = np.random.default_rng(10_000 + L)
    return {name: make_split(n, L, rng)
            for name, n in [("train", n_train), ("val", n_val), ("test", n_test)]}


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class SeqModel:
    def __init__(self, hidden=32, attention=False, bilinear=True, seed=0,
                 dtype=np.float32, orth_init=False):
        rng = np.random.default_rng(seed)
        H = hidden
        s = 1.0 / np.sqrt(H)                      # PyTorch nn.RNN default init range

        def u(*shape):
            return rng.uniform(-s, s, size=shape).astype(dtype)

        self.H, self.attention, self.bilinear, self.dtype = H, attention, bilinear, dtype
        self.p = {"Wxh": u(D, H), "Whh": u(H, H), "bh": u(H),
                  "Wo": u(H, C), "bo": u(C)}
        if attention and bilinear:
            self.p["Wa"] = u(H, H)
        if orth_init:                              # optional: orthogonal recurrent matrix
            Q, R = np.linalg.qr(rng.standard_normal((H, H)))
            self.p["Whh"] = (Q * np.sign(np.diag(R))).astype(dtype)

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def encode(self, X):
        p, (B, L, _) = self.p, X.shape
        XW = X @ p["Wxh"] + p["bh"]
        Hs = np.empty((B, L, self.H), dtype=self.dtype)
        h = np.zeros((B, self.H), dtype=self.dtype)
        Whh = p["Whh"]
        for t in range(L):
            h = np.tanh(XW[:, t] + h @ Whh)
            Hs[:, t] = h
        return Hs

    def forward(self, X):
        p = self.p
        Hs = self.encode(X)
        q = Hs[:, -1]
        if self.attention:
            v = q @ p["Wa"] if self.bilinear else q
            e = np.einsum("bh,blh->bl", v, Hs)
            e = e - e.max(1, keepdims=True)
            a = np.exp(e)
            a /= a.sum(1, keepdims=True)
            c = np.einsum("bl,blh->bh", a, Hs)
        else:
            v, a, c = None, None, q
        logits = c @ p["Wo"] + p["bo"]
        return logits, dict(Hs=Hs, q=q, v=v, a=a, c=c)

    def loss_and_grads(self, X, y):
        p = self.p
        B, L, _ = X.shape
        logits, ca = self.forward(X)
        Hs, q, v, a, c = ca["Hs"], ca["q"], ca["v"], ca["a"], ca["c"]
        z = logits - logits.max(1, keepdims=True)
        P = np.exp(z)
        P /= P.sum(1, keepdims=True)
        loss = float(-np.log(P[np.arange(B), y] + 1e-12).mean())
        dl = P.copy()
        dl[np.arange(B), y] -= 1.0
        dl /= B

        g = {"Wo": c.T @ dl, "bo": dl.sum(0)}
        dc = dl @ p["Wo"].T
        dHs = np.zeros_like(Hs)
        if self.attention:
            dHs += a[:, :, None] * dc[:, None, :]                 # c = sum a_t h_t
            da = np.einsum("bh,blh->bl", dc, Hs)
            de = a * (da - (a * da).sum(1, keepdims=True))        # softmax
            dHs += de[:, :, None] * v[:, None, :]                 # e_t = v . h_t
            dv = np.einsum("bl,blh->bh", de, Hs)
            if self.bilinear:
                g["Wa"] = q.T @ dv
                dq = dv @ p["Wa"].T
            else:
                dq = dv
            dHs[:, -1] += dq                                      # q = h_T
        else:
            dHs[:, -1] += dc

        G = 1.0 - Hs ** 2
        dpre = np.empty_like(Hs)
        WhhT = p["Whh"].T
        dh_next = np.zeros((B, self.H), dtype=self.dtype)
        for t in range(L - 1, -1, -1):                            # BPTT
            dp = (dHs[:, t] + dh_next) * G[:, t]
            dpre[:, t] = dp
            dh_next = dp @ WhhT
        Hprev = np.concatenate([np.zeros((B, 1, self.H), self.dtype), Hs[:, :-1]], 1)
        flat = dpre.reshape(-1, self.H)
        g["Whh"] = Hprev.reshape(-1, self.H).T @ flat
        g["Wxh"] = X.reshape(-1, D).T @ flat
        g["bh"] = flat.sum(0)
        return loss, g


# ----------------------------------------------------------------------------
# optimisation
# ----------------------------------------------------------------------------
class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            params[k] -= self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)


def evaluate(model, split, bs=500):
    X, y = split["X"], split["y"]
    probs, alphas = [], []
    for i in range(0, len(y), bs):
        logits, ca = model.forward(X[i:i + bs])
        z = logits - logits.max(1, keepdims=True)
        P = np.exp(z)
        P /= P.sum(1, keepdims=True)
        probs.append(P)
        if ca["a"] is not None:
            alphas.append(ca["a"])
    P = np.concatenate(probs)
    pred = P.argmax(1)
    out = dict(loss=float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean()),
               acc=float((pred == y).mean()), pred=pred, probs=P)
    if alphas:
        out["alpha"] = np.concatenate(alphas)
    return out


def train(model, data, epochs=40, lr=3e-3, bs=64, clip=1.0, seed=0, log=None):
    """Adam + global-norm clipping, fixed epoch budget, best-val-accuracy checkpoint."""
    rng = np.random.default_rng(seed + 12345)
    opt = Adam(model.p, lr)
    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    n = len(ytr)
    best = dict(val_acc=-1.0, val_loss=np.inf, params=None, epoch=0)
    hist, train_time = [], 0.0
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        tl, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss, g = model.loss_and_grads(Xtr[idx], ytr[idx])
            norm = np.sqrt(sum(float((v ** 2).sum()) for v in g.values()))
            if norm > clip:
                for k in g:
                    g[k] = g[k] * (clip / norm)
            opt.step(model.p, g)
            tl += loss
            nb += 1
        train_time += time.perf_counter() - t0
        ev = evaluate(model, data["val"])
        hist.append(dict(epoch=ep, train_loss=tl / nb, val_loss=ev["loss"], val_acc=ev["acc"]))
        if ev["acc"] > best["val_acc"] or (ev["acc"] == best["val_acc"] and ev["loss"] < best["val_loss"]):
            best = dict(val_acc=ev["acc"], val_loss=ev["loss"], epoch=ep,
                        params={k: v.copy() for k, v in model.p.items()})
        if log:
            log(f"    ep {ep:2d} train_loss {tl / nb:.4f} val_loss {ev['loss']:.4f} val_acc {ev['acc']:.4f}")
    for k in model.p:
        model.p[k][...] = best["params"][k]
    return dict(history=hist, best_epoch=best["epoch"], best_val_acc=best["val_acc"],
                train_time=train_time)
