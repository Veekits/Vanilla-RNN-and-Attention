"""Tables, plots and diagnostics from results/runs. Writes results/*.csv|json and figs/*.png."""
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rnn_attn import SeqModel, make_dataset

os.makedirs("figs", exist_ok=True)
LENGTHS = [10, 25, 50, 100]
# categorical slots 1..5 of the reference palette (light mode)
COL = {"rnn": "#2a78d6", "attn": "#eb6834", "attn_dot": "#1baf7a", "rnn_orth": "#e87ba4", "rnn_long": "#6250d6"}
NAME = {"rnn": "A: vanilla RNN", "attn": "B: RNN + attention", "attn_dot": "B (dot-product, no extra params)",
        "rnn_orth": "A with orthogonal init", "rnn_long": "A, 5x training budget"}
INK, GRID = "#0b0b0b", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight"})

# ---------------------------------------------------------------- load
recs = [json.load(open(f)) for f in sorted(glob.glob("results/runs/*.json"))]
df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in recs])
hist = {(r["model"], r["L"], r["seed"]): r["history"] for r in recs}
DATA = {L: make_dataset(L) for L in LENGTHS}


def npz(model, L, seed):
    return np.load(f"results/runs/{model}_L{L}_s{seed}.npz")


def load_model(name, L, seed):
    z = npz(name, L, seed)
    att = name.startswith("attn")
    m = SeqModel(attention=att, bilinear=(name == "attn"), seed=0)
    m.p = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("w_")}
    return m


out = {}

# ---------------------------------------------------------------- 1. accuracy vs length
g = df.groupby(["model", "L"])
tab = g.agg(acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"), n=("seed", "count"),
            time_s=("train_time", "mean"), params=("params", "first"), epochs=("epochs", "first"),
            best_epoch=("best_epoch", "mean")).reset_index()
tab.to_csv("results/summary.csv", index=False)
out["summary"] = tab.to_dict("records")
print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

fig, ax = plt.subplots(figsize=(6.4, 3.8))
for m, lw, ls in [("rnn_long", 1.4, "--"), ("rnn_orth", 1.4, "--"), ("attn_dot", 1.4, "--"),
                  ("rnn", 2.4, "-"), ("attn", 2.4, "-")]:
    t = tab[tab.model == m].sort_values("L")
    if t.empty:
        continue
    ax.errorbar(t.L, t.acc_mean, yerr=t.acc_std.fillna(0), color=COL[m], lw=lw, ls=ls, marker="o", ms=5,
                capsize=3, label=NAME[m])
ax.axhline(0.1, color="#8a8984", lw=1, ls=":")
ax.text(10.5, 0.115, "chance (10%)", color="#52514e", fontsize=8.5)
ax.set_xscale("log")
ax.set_xticks(LENGTHS)
ax.set_xticklabels(LENGTHS)
ax.minorticks_off()
ax.set_xlabel("sequence length L")
ax.set_ylabel("test accuracy")
ax.set_ylim(0, 1.04)
ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.45))
ax.set_title("Test accuracy vs. sequence length (mean ± std over seeds)", fontsize=10.5, loc="left")
fig.savefig("figs/fig1_accuracy_vs_length.png")
plt.close(fig)

# ---------------------------------------------------------------- 2. accuracy vs memory distance
EDGES = [0, 5, 10, 25, 50, 75, 100]


def dist_bins(L):
    e = [x for x in EDGES if x < L] + [L]
    return list(zip(e[:-1], e[1:]))


rows = []
for L in LENGTHS:
    d = (L - 1) - DATA[L]["test"]["k"]
    y = DATA[L]["test"]["y"]
    for m in ("rnn", "attn", "rnn_orth", "attn_dot"):
        for s in range(5):
            pred = npz(m, L, s)["pred"]
            for lo, hi in dist_bins(L):
                sel = (d >= lo) & (d < hi)
                rows.append(dict(model=m, L=L, seed=s, lo=lo, hi=hi, n=int(sel.sum()),
                                 acc=float((pred[sel] == y[sel]).mean())))
dd = pd.DataFrame(rows)
dd.to_csv("results/acc_by_distance.csv", index=False)
bd = dd.groupby(["model", "L", "lo", "hi"]).acc.agg(["mean", "std"]).reset_index()
out["acc_by_distance"] = bd.to_dict("records")

fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True)
for ax, L in zip(axes, LENGTHS):
    for m in ("rnn", "attn"):
        t = bd[(bd.model == m) & (bd.L == L)].sort_values("lo")
        x = (t.lo + t.hi - 1) / 2
        ax.errorbar(x, t["mean"], yerr=t["std"].fillna(0), color=COL[m], lw=2.2, marker="o", ms=4.5,
                    capsize=2.5, label=NAME[m])
    ax.axhline(0.1, color="#8a8984", lw=1, ls=":")
    ax.set_title(f"L = {L}", fontsize=10, loc="left")
    ax.set_xlabel("memory distance  T−k  (steps)")
    ax.set_ylim(0, 1.04)
axes[0].set_ylabel("test accuracy")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("Accuracy by memory distance (bins: 0-4, 5-9, 10-24, 25-49, 50-74, 75-99)", x=0.01, ha="left", fontsize=10.5, y=1.03)
fig.savefig("figs/fig2_accuracy_vs_distance.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. training curves
fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
for ax, L in zip(axes, (25, 50, 100)):
    for m in ("rnn", "attn"):
        H = np.array([[h["val_acc"] for h in hist[(m, L, s)]] for s in range(5)])
        ax.plot(np.arange(1, H.shape[1] + 1), H.mean(0), color=COL[m], lw=2.2, label=NAME[m])
        ax.fill_between(np.arange(1, H.shape[1] + 1), H.min(0), H.max(0), color=COL[m], alpha=0.15, lw=0)
    ax.set_title(f"L = {L}", fontsize=10, loc="left")
    ax.set_xlabel("epoch")
axes[0].set_ylabel("validation accuracy")
axes[0].legend(frameon=False, fontsize=8, loc="center right")
fig.savefig("figs/fig3_training_curves.png")
plt.close(fig)

# ---------------------------------------------------------------- 4. attention localisation
loc_rows = []
for L in LENGTHS:
    k, y = DATA[L]["test"]["k"], DATA[L]["test"]["y"]
    for m in ("attn", "attn_dot"):
        for s in range(5):
            z = npz(m, L, s)
            a, pred = z["alpha"], z["pred"]
            am = a.argmax(1)
            ok = pred == y
            loc_rows.append(dict(model=m, L=L, seed=s, acc=float(ok.mean()),
                                 loc_exact=float((am == k).mean()),
                                 loc_pm1=float((np.abs(am - k) <= 1).mean()),
                                 loc_last=float((am == L - 1).mean()),
                                 mass_k=float(a[np.arange(len(k)), k].mean()),
                                 loc_exact_correct=float((am[ok] == k[ok]).mean()) if ok.any() else np.nan,
                                 loc_exact_wrong=float((am[~ok] == k[~ok]).mean()) if (~ok).any() else np.nan,
                                 n_wrong=int((~ok).sum())))
lr = pd.DataFrame(loc_rows)
lr.to_csv("results/attention_localization.csv", index=False)
lsum = lr.groupby(["model", "L"]).agg(["mean", "std"]).drop(columns="seed").reset_index()
out["localization"] = lr.groupby(["model", "L"]).mean().reset_index().to_dict("records")
print(lr.groupby(["model", "L"])[["acc", "loc_exact", "loc_pm1", "loc_last", "mass_k", "loc_exact_correct"]].mean().round(4))

# mean attention profile around the true position (L=100 and 50), model attn
fig, axes = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True)
for ax, L in zip(axes, (50, 100)):
    k = DATA[L]["test"]["k"]
    offs = np.arange(-4, 7)
    prof = []
    for s in range(5):
        a = npz("attn", L, s)["alpha"]
        P = []
        for o in offs:
            idx = k + o
            ok = (idx >= 0) & (idx < L)
            P.append(a[np.arange(len(k))[ok], idx[ok]].mean())
        prof.append(P)
    prof = np.array(prof)
    ax.bar(offs, prof.mean(0), color=COL["attn"], width=0.75)
    ax.set_xticks(offs)
    ax.set_title(f"L = {L}: mean attention weight vs. offset t−k", fontsize=10, loc="left")
    ax.set_xlabel("offset from the true marked position")
axes[0].set_ylabel("mean α")
fig.savefig("figs/fig5_attention_profile.png")
plt.close(fig)
out["attention_profile"] = {}

# five correctly classified examples, L=100, seed 0
L, s = 100, 0
d = DATA[L]["test"]
z = npz("attn", L, s)
a, pred = z["alpha"], z["pred"]
dist = (L - 1) - d["k"]
chosen = []
for target in (4, 25, 50, 75, 97):
    cand = np.where((pred == d["y"]) & (np.abs(dist - target) <= 3))[0]
    chosen.append(int(cand[0]))
fig, axes = plt.subplots(len(chosen), 1, figsize=(8, 1.5 * len(chosen) + 0.6), sharex=True)
ex = []
for ax, i in zip(axes, chosen):
    ax.bar(np.arange(L), a[i], color=COL["attn"], width=0.9)
    ax.axvline(d["k"][i], color=INK, lw=1, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 1])
    ax.set_title(f"true k={d['k'][i]}  (T−k={dist[i]})   label={d['y'][i]}   argmax α at t={a[i].argmax()}   α_k={a[i][d['k'][i]]:.2f}",
                 fontsize=8.2, loc="left", pad=2)
    ex.append(dict(idx=i, k=int(d["k"][i]), dist=int(dist[i]), label=int(d["y"][i]), argmax=int(a[i].argmax()),
                   alpha_k=float(a[i][d["k"][i]]), alpha_max=float(a[i].max())))
axes[-1].set_xlabel("time step t  (dashed line = true marked position k)")
axes[len(axes) // 2].set_ylabel("attention weight α_t")
fig.savefig("figs/fig4_attention_examples.png")
plt.close(fig)
out["attention_examples"] = ex
print(pd.DataFrame(ex))

# ---------------------------------------------------------------- 5. forgetting diagnostics (model A)
def sensitivity(name, L, seed, n=500):
    """||h_T(x) - h_T(x')|| where x' changes only the marked value; also Jacobian norm dh_T/dh_{T-d}."""
    m = load_model(name, L, seed)
    d_ = DATA[L]["test"]
    X = d_["X"][:n].copy()
    k = d_["k"][:n]
    X2 = X.copy()
    newv = (d_["y"][:n] + 1 + np.random.default_rng(0).integers(0, 9, size=n)) % 10
    X2[np.arange(n), k, :10] = 0
    X2[np.arange(n), k, newv] = 1
    dh = np.linalg.norm(m.encode(X)[:, -1] - m.encode(X2)[:, -1], axis=1)
    dist_ = (L - 1) - k
    Hs = m.encode(X[:100])
    Wt = m.p["Whh"].T.astype(np.float64)
    logJ = np.zeros((100, L))
    for i in range(100):
        P = np.eye(m.H)
        for dstep in range(1, L):
            t = L - dstep
            P = P @ (np.diag(1 - Hs[i, t].astype(np.float64) ** 2) @ Wt)
            logJ[i, dstep] = np.linalg.norm(P, 2)
    return dist_, dh, logJ


fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
diag = {}
for (name, L, col, ls) in [("rnn", 100, COL["rnn"], "-"), ("rnn", 25, "#52514e", "-"), ("rnn_orth", 100, COL["rnn_orth"], "--")]:
    dist_, dh, logJ = sensitivity(name, L, 0)
    bins = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 75), (75, 100)]
    xs, ys = [], []
    for lo, hi in bins:
        sel = (dist_ >= lo) & (dist_ < hi)
        if sel.sum() > 5:
            xs.append((lo + hi - 1) / 2)
            ys.append(dh[sel].mean())
    lab = f"{NAME[name]}, L={L}"
    axes[0].plot(xs, ys, color=col, ls=ls, marker="o", ms=4.5, lw=2, label=lab)
    jm = np.exp(np.log(logJ[:, 1:] + 1e-300).mean(0))
    axes[1].plot(np.arange(1, L), jm, color=col, ls=ls, lw=2, label=lab)
    diag[f"{name}_L{L}"] = dict(sens_bins=dict(zip(map(str, xs), map(float, ys))),
                                jac_norm_at={str(dd_): float(jm[dd_ - 1]) for dd_ in (1, 5, 10, 25, 50, 99) if dd_ < L})
axes[0].set_yscale("log")
axes[0].set_xlabel("memory distance T−k (steps)")
axes[0].set_ylabel("‖h_T(x) − h_T(x′)‖\n(x′ changes only the marked value)")
axes[0].set_title("Does the final state still depend on the marked value?", fontsize=9.5, loc="left")
axes[0].legend(frameon=False, fontsize=7.5)
axes[1].set_yscale("log")
axes[1].set_xlabel("steps back d")
axes[1].set_ylabel("‖∂h_T / ∂h_(T−d)‖₂  (geometric mean)")
axes[1].set_title("Backward path through the recurrence", fontsize=9.5, loc="left")
axes[1].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.savefig("figs/fig6_forgetting_diagnostics.png")
plt.close(fig)
out["forgetting"] = diag
print(json.dumps(diag, indent=1))

json.dump(out, open("results/analysis.json", "w"), indent=1, default=float)
