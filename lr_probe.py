"""Pick ONE shared learning rate from a small grid using validation accuracy only (L=50, seed 0)."""
import json, sys
from rnn_attn import *

L, EPOCHS = int(sys.argv[1]), int(sys.argv[2])
data = make_dataset(L)
out = {}
for lr in (1e-3, 3e-3, 1e-2, 3e-2):
    for name, att in (("rnn", False), ("attn", True)):
        m = SeqModel(attention=att, seed=0)
        r = train(m, data, epochs=EPOCHS, lr=lr, seed=0)
        out[f"{name}_lr{lr}"] = r["best_val_acc"]
        print(f"L={L} lr={lr:<6} {name:5s} best val acc {r['best_val_acc']:.4f} (epoch {r['best_epoch']})", flush=True)
json.dump(out, open(f"lr_probe_L{L}.json", "w"), indent=1)
