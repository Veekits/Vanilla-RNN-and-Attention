"""Run all experiments. Resumable; shard across processes with  --worker i --workers n.

Main comparison (matched optimiser / budget / data):
    rnn   : Model A, vanilla RNN
    attn  : Model B, RNN + bilinear attention on the final state
Controls (not part of the required comparison, used to interpret it):
    attn_dot : Model B with parameter-free dot-product attention (same #params as A)
    rnn_orth : Model A with orthogonal recurrent init (same budget)
    rnn_long : Model A, default init, 5x training budget
"""
import argparse
import json
import os
import time

import numpy as np

from rnn_attn import SeqModel, evaluate, make_dataset, train

LENGTHS = [10, 25, 50, 100]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 60
LR = 3e-3          # chosen from {1e-3,3e-3,1e-2,3e-2} on validation accuracy of the vanilla RNN
HIDDEN = 32
OUT = "results"

MODELS = {
    "rnn":      dict(kw=dict(attention=False), epochs=EPOCHS),
    "attn":     dict(kw=dict(attention=True, bilinear=True), epochs=EPOCHS),
    "attn_dot": dict(kw=dict(attention=True, bilinear=False), epochs=EPOCHS),
    "rnn_orth": dict(kw=dict(attention=False, orth_init=True), epochs=EPOCHS),
    "rnn_long": dict(kw=dict(attention=False), epochs=5 * EPOCHS),
}


def jobs():
    out = []
    for L in LENGTHS:
        for s in SEEDS:
            for m in ("rnn", "attn", "attn_dot", "rnn_orth"):
                out.append((m, L, s))
    for L in (25, 50, 100):                       # long-budget control: 3 seeds
        for s in SEEDS[:3]:
            out.append(("rnn_long", L, s))
    # biggest jobs first for balanced sharding
    out.sort(key=lambda j: -(j[1] * MODELS[j[0]]["epochs"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(f"{OUT}/runs", exist_ok=True)
    cache = {}
    for i, (name, L, seed) in enumerate(jobs()):
        if i % a.workers != a.worker:
            continue
        tag = f"{name}_L{L}_s{seed}"
        if os.path.exists(f"{OUT}/runs/{tag}.json"):
            continue
        if L not in cache:
            cache = {L: make_dataset(L)}
        data = cache[L]
        cfg = MODELS[name]
        model = SeqModel(hidden=HIDDEN, seed=seed, **cfg["kw"])
        t0 = time.time()
        info = train(model, data, epochs=cfg["epochs"], lr=LR, seed=seed)
        ev = evaluate(model, data["test"])
        rec = dict(model=name, L=L, seed=seed, params=model.n_params(), epochs=cfg["epochs"],
                   lr=LR, test_acc=ev["acc"], test_loss=ev["loss"], val_acc=info["best_val_acc"],
                   best_epoch=info["best_epoch"], train_time=info["train_time"],
                   history=info["history"])
        np.savez_compressed(f"{OUT}/runs/{tag}.npz", pred=ev["pred"], probs=ev["probs"].astype(np.float32),
                            **({"alpha": ev["alpha"].astype(np.float32)} if "alpha" in ev else {}),
                            **{f"w_{k}": v for k, v in model.p.items()})
        json.dump(rec, open(f"{OUT}/runs/{tag}.json", "w"))
        print(f"[{time.strftime('%H:%M:%S')}] {tag:18s} test {ev['acc']:.4f} val {info['best_val_acc']:.4f} "
              f"best_ep {info['best_epoch']:3d} time {info['train_time']:.0f}s (wall {time.time() - t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
