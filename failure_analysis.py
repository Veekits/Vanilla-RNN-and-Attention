"""Failure analysis.
 (a) in-distribution, L=100: cases where the vanilla RNN fails and attention succeeds
 (b) both-fail cases. In-distribution Model B is (near-)perfect, so we also run a
     length-extrapolation stress test: models trained at L=100, tested at L=200/400/800.
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rnn_attn import SeqModel, evaluate, make_dataset, make_split

COL = {"rnn": "#2a78d6", "attn": "#eb6834"}
INK = "#0b0b0b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight"})
res = {}


def load(name, L, seed):
    z = np.load(f"results/runs/{name}_L{L}_s{seed}.npz")
    m = SeqModel(attention=(name == "attn"), seed=0)
    m.p = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("w_")}
    return m


# ------------------------------------------------ in-distribution both-fail count (all L, seeds)
cnt = []
for L in (10, 25, 50, 100):
    y = make_dataset(L)["test"]["y"]
    for s in range(5):
        pr = np.load(f"results/runs/rnn_L{L}_s{s}.npz")["pred"]
        pa = np.load(f"results/runs/attn_L{L}_s{s}.npz")["pred"]
        cnt.append(dict(L=L, seed=s, rnn_wrong=int((pr != y).sum()), attn_wrong=int((pa != y).sum()),
                        both_wrong=int(((pr != y) & (pa != y)).sum()),
                        rnn_wrong_attn_right=int(((pr != y) & (pa == y)).sum())))
cnt = pd.DataFrame(cnt)
cnt.to_csv("results/failure_counts_in_distribution.csv", index=False)
print(cnt.groupby("L").sum(numeric_only=True).drop(columns="seed"))
res["in_dist_counts"] = cnt.groupby("L").sum(numeric_only=True).drop(columns="seed").to_dict("index")

# ------------------------------------------------ (a) RNN fails, attention succeeds  (L=100, seed 0)
L, s = 100, 0
d = make_dataset(L)["test"]
mA, mB = load("rnn", L, s), load("attn", L, s)
evA, evB = evaluate(mA, d), evaluate(mB, d)
k, y, vals = d["k"], d["y"], d["vals"]
dist = L - 1 - k
sel = np.where((evA["pred"] != y) & (evB["pred"] == y))[0]
print(f"\nL=100 seed0: RNN wrong & attn right: {len(sel)} of {len(y)}")

# is the RNN output random, or driven by recent inputs?  (pred == value at last / last-3 steps)
wrong = np.where(evA["pred"] != y)[0]
res["rnn_wrong_stats"] = dict(
    n_wrong=int(len(wrong)),
    pred_eq_last_value=float((evA["pred"][wrong] == vals[wrong, -1]).mean()),
    pred_in_last3_values=float(np.mean([evA["pred"][i] in vals[i, -3:] for i in wrong])),
    pred_eq_label_given_wrong=0.0,
    mean_max_prob_wrong=float(evA["probs"][wrong].max(1).mean()),
    mean_prob_on_true_label_wrong=float(evA["probs"][wrong, y[wrong]].mean()))
# accuracy of the RNN by distance bin (for reference)
for lo, hi in [(0, 5), (5, 10), (10, 25), (25, 100)]:
    m_ = (dist >= lo) & (dist < hi)
    res["rnn_wrong_stats"][f"rnn_acc_dist_{lo}_{hi}"] = float((evA["pred"][m_] == y[m_]).mean())
print(json.dumps(res["rnn_wrong_stats"], indent=1))

# sensitivity of h_T to the marked value for each chosen case
def delta_h(model, i):
    X = d["X"][i:i + 1].copy()
    X2 = X.copy()
    nv = (y[i] + 3) % 10
    X2[0, k[i], :10] = 0
    X2[0, k[i], nv] = 1
    h1, h2 = model.encode(X)[0, -1], model.encode(X2)[0, -1]
    return float(np.linalg.norm(h1 - h2)), float(np.linalg.norm(h1))


cases_a = []
for target in (30, 60, 92):
    c = sel[np.argmin(np.abs(dist[sel] - target))]
    dh, hn = delta_h(mA, c)
    cases_a.append(dict(idx=int(c), k=int(k[c]), dist=int(dist[c]), label=int(y[c]),
                        rnn_pred=int(evA["pred"][c]), rnn_conf=float(evA["probs"][c].max()),
                        rnn_prob_true=float(evA["probs"][c, y[c]]),
                        value_at_last_step=int(vals[c, -1]),
                        attn_pred=int(evB["pred"][c]), attn_conf=float(evB["probs"][c].max()),
                        attn_argmax=int(evB["alpha"][c].argmax()), alpha_k=float(evB["alpha"][c, k[c]]),
                        dh_T=dh, h_T_norm=hn))
res["case_rnn_fail_attn_ok"] = cases_a
print(pd.DataFrame(cases_a).round(3).to_string(index=False))

# ------------------------------------------------ (b) length-extrapolation stress test
stress, both_cases = [], []
for Lt in (200, 400, 800):
    dt = make_split(2000, Lt, np.random.default_rng(777 + Lt))
    kk, yy = dt["k"], dt["y"]
    accs = {"rnn": [], "attn": []}
    for sd in range(5):
        a_ = evaluate(load("rnn", 100, sd), dt)
        b_ = evaluate(load("attn", 100, sd), dt)
        accs["rnn"].append(a_["acc"])
        accs["attn"].append(b_["acc"])
        both = np.where((a_["pred"] != yy) & (b_["pred"] != yy))[0]
        am = b_["alpha"].argmax(1)
        wrongB = b_["pred"] != yy
        stress.append(dict(L_test=Lt, seed=sd, rnn_acc=a_["acc"], attn_acc=b_["acc"], n_both_wrong=int(len(both)),
                           attn_wrong=int(wrongB.sum()),
                           attn_wrong_argmax_is_k=float((am[wrongB] == kk[wrongB]).mean()) if wrongB.any() else np.nan,
                           attn_wrong_mass_k=float(b_["alpha"][wrongB, kk[wrongB]].mean()) if wrongB.any() else np.nan,
                           attn_right_argmax_is_k=float((am[~wrongB] == kk[~wrongB]).mean())))
        if sd == 0:
            for i in both:
                both_cases.append(dict(L_test=Lt, idx=int(i), k=int(kk[i]), dist=int(Lt - 1 - kk[i]),
                                       label=int(yy[i]), rnn_pred=int(a_["pred"][i]),
                                       attn_pred=int(b_["pred"][i]), attn_conf=float(b_["probs"][i].max()),
                                       attn_argmax=int(am[i]), alpha_k=float(b_["alpha"][i, kk[i]]),
                                       alpha_max=float(b_["alpha"][i].max()),
                                       val_at_argmax=int(dt["vals"][i, am[i]]), alpha=b_["alpha"][i]))
st = pd.DataFrame(stress)
st.to_csv("results/stress_length_extrapolation.csv", index=False)
print(st.groupby("L_test").mean(numeric_only=True).round(4).to_string())
res["stress"] = st.groupby("L_test").mean(numeric_only=True).to_dict("index")

# choose 3 both-fail cases (different test lengths / error types when possible) ----------
bc = pd.DataFrame([{k_: v for k_, v in c.items() if k_ != "alpha"} for c in both_cases])
print(f"\nboth-fail cases found (seed 0, trained L=100): {len(bc)}")
picked = []
if len(bc):
    bc["err_type"] = np.where(bc.attn_argmax == bc.k, "decode (right state, wrong readout)",
                              np.where(bc.alpha_k < 0.1, "retrieval (attention missed k)", "diluted (mass split)"))
    print(bc.err_type.value_counts())
    # pick: prefer one from each err_type, then fill
    order = []
    for t in bc.err_type.unique():
        order += list(bc[bc.err_type == t].sort_values("L_test").index[:1])
    order += [i for i in bc.index if i not in order]
    picked = order[:3]
    print(bc.loc[picked].round(3).to_string(index=False))
    res["case_both_fail"] = bc.loc[picked].to_dict("records")
    res["both_fail_err_types"] = bc.err_type.value_counts().to_dict()

# ------------------------------------------------ figure: alpha for the 6 cases
fig, axes = plt.subplots(2, 3, figsize=(11, 4.4))
for j, c in enumerate(cases_a):
    ax = axes[0, j]
    ax.bar(np.arange(L), evB["alpha"][c["idx"]], color=COL["attn"], width=0.9)
    ax.axvline(c["k"], color=INK, lw=1, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"RNN fails, attention OK (L=100)\nk={c['k']}, T−k={c['dist']}, label={c['label']}; "
                 f"RNN→{c['rnn_pred']}, attn→{c['attn_pred']}", fontsize=8.2, loc="left")
for j, ci in enumerate(picked):
    ax = axes[1, j]
    c = both_cases[ci]
    ax.bar(np.arange(len(c["alpha"])), c["alpha"], color=COL["attn"], width=1.0)
    ax.axvline(c["k"], color=INK, lw=1, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"both fail (trained L=100, tested L={c['L_test']})\nk={c['k']}, label={c['label']}; "
                 f"RNN→{c['rnn_pred']}, attn→{c['attn_pred']}, argmax α at {c['attn_argmax']}", fontsize=8.2, loc="left")
for ax in axes[1]:
    ax.set_xlabel("time step t")
for ax in axes[:, 0]:
    ax.set_ylabel("attention weight α_t")
fig.tight_layout()
fig.savefig("figs/fig7_failure_cases.png")
plt.close(fig)

json.dump(res, open("results/failure_analysis.json", "w"), indent=1, default=float)
