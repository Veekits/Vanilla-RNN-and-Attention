"""Every in-distribution error of Model B (all lengths, seeds), and the both-fail cases among them.
Re-draws figs/fig7_failure_cases.png (top: RNN fails / attention OK, bottom: both fail)."""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rnn_attn import SeqModel, evaluate, make_dataset

COL = {"rnn": "#2a78d6", "attn": "#eb6834"}
INK = "#0b0b0b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight"})


def load(name, L, seed):
    z = np.load(f"results/runs/{name}_L{L}_s{seed}.npz")
    m = SeqModel(attention=(name == "attn"), seed=0)
    m.p = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("w_")}
    return m


rows, alphas = [], []
for L in (10, 25, 50, 100):
    d = make_dataset(L)["test"]
    for s in range(5):
        eA, eB = evaluate(load("rnn", L, s), d), evaluate(load("attn", L, s), d)
        for i in np.where(eB["pred"] != d["y"])[0]:
            a = eB["alpha"][i]
            am = int(a.argmax())
            top2 = np.sort(a)[-2:][::-1]
            rows.append(dict(L=L, seed=s, idx=int(i), k=int(d["k"][i]), dist=int(L - 1 - d["k"][i]),
                             label=int(d["y"][i]), attn_pred=int(eB["pred"][i]), attn_conf=float(eB["probs"][i].max()),
                             rnn_pred=int(eA["pred"][i]), rnn_also_wrong=bool(eA["pred"][i] != d["y"][i]),
                             argmax=am, alpha_k=float(a[d["k"][i]]), alpha_max=float(a.max()),
                             top2_mass=float(top2.sum()), value_at_argmax=int(d["vals"][i, am]),
                             value_at_k=int(d["vals"][i, d["k"][i]]),
                             prob_true=float(eB["probs"][i, d["y"][i]])))
            alphas.append(a)
f = pd.DataFrame(rows)
f["error_type"] = np.where(f.alpha_k >= 0.5, "right state retrieved, wrong readout",
                           np.where(np.abs(f.argmax - f.k) <= 1, "mass split between k and neighbour",
                                    "wrong state retrieved"))
f.to_csv("results/attention_errors.csv", index=False)
print(f.drop(columns=["value_at_k"]).round(3).to_string(index=False))
print(f.error_type.value_counts())

both = f[f.rnn_also_wrong]
print("\nboth-fail cases:", len(both))

# --- figure ---------------------------------------------------------------------------
fa = json.load(open("results/failure_analysis.json"))["case_rnn_fail_attn_ok"]
L = 100
d = make_dataset(L)["test"]
eB = evaluate(load("attn", L, 0), d)
fig, axes = plt.subplots(2, 3, figsize=(11, 4.4))
for j, c in enumerate(fa):
    ax = axes[0, j]
    ax.bar(np.arange(L), eB["alpha"][c["idx"]], color=COL["attn"], width=0.9)
    ax.axvline(c["k"], color=INK, lw=1, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"RNN wrong, attention right  (L=100, seed 0)\nk={c['k']} (T−k={c['dist']}), label={c['label']}; "
                 f"RNN→{c['rnn_pred']} (p={c['rnn_conf']:.2f}), attn→{c['attn_pred']}", fontsize=8, loc="left")
for j, (ix, r) in enumerate(both.head(3).iterrows()):
    ax = axes[1, j]
    a = alphas[ix]
    ax.bar(np.arange(len(a)), a, color=COL["attn"], width=0.9)
    ax.axvline(r.k, color=INK, lw=1, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"both wrong  (L={r.L}, seed {r.seed})\nk={r.k}, label={r.label}; RNN→{r.rnn_pred}, attn→{r.attn_pred} "
                 f"(p={r.attn_conf:.2f}); α_k={r.alpha_k:.2f}", fontsize=8, loc="left")
for ax in axes[1]:
    ax.set_xlabel("time step t   (dashed = true marked position k)")
for ax in axes[:, 0]:
    ax.set_ylabel("attention weight α_t")
fig.tight_layout()
fig.savefig("figs/fig7_failure_cases.png")
plt.close(fig)
