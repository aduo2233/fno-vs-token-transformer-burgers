"""Train FNO1d and TokenTransformer on Burgers u0 -> u(x,T).

Fixed seeds, pure CPU, a few minutes each. Saves:
  data/fno.pt, data/transformer.pt
  results/train_curves.npz
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from models import FNO1d, TokenTransformer, count_params

SEED = 20260825
EPOCHS = 300
LR = 1e-3
BATCH = 50
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def load(name):
    d = np.load(os.path.join(DATA_DIR, name))
    return d["a"], d["u"]


def make_tensors(a, u):
    a = torch.from_numpy(a).float().unsqueeze(-1)
    u = torch.from_numpy(u).float()
    return a, u


def train_model(model, a_tr, u_tr, a_va, u_va, tag, epochs=EPOCHS):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    n = a_tr.shape[0]
    curves = {"train": [], "val": []}
    best = float("inf")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = a_tr[idx], u_tr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(a_va), u_va).item()
            tloss = tot / n
        curves["train"].append(tloss)
        curves["val"].append(vloss)
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(),
                       os.path.join(DATA_DIR, f"{tag}.pt"))
        if (ep + 1) % 50 == 0:
            print(f"[{tag}] ep {ep+1}/{epochs} train {tloss:.3e} val {vloss:.3e}")
    return curves, best


def main():
    a_tr, u_tr = load("burgers_train.npz")
    a_te, u_te = load("burgers_test.npz")
    a_tr, u_tr = make_tensors(a_tr, u_tr)
    a_te, u_te = make_tensors(a_te, u_te)

    fno = FNO1d(modes=16, width=32)
    trm = TokenTransformer(n_grid=a_tr.shape[1])
    print("FNO params:", count_params(fno))
    print("Transformer params:", count_params(trm))

    t0 = time.time()
    c1, b1 = train_model(fno, a_tr, u_tr, a_te, u_te, "fno")
    t1 = time.time()
    c2, b2 = train_model(trm, a_tr, u_tr, a_te, u_te, "transformer")
    t2 = time.time()
    print(f"FNO: {t1-t0:.1f}s best val {b1:.3e}")
    print(f"Transformer: {t2-t1:.1f}s best val {b2:.3e}")

    np.savez(os.path.join(RESULT_DIR, "train_curves.npz"),
             fno_train=np.array(c1["train"]), fno_val=np.array(c1["val"]),
             trm_train=np.array(c2["train"]), trm_val=np.array(c2["val"]))
    with open(os.path.join(RESULT_DIR, "train_metrics.json"), "w") as f:
        json.dump({"fno_time_s": t1 - t0, "trm_time_s": t2 - t1,
                   "fno_best_val": b1, "trm_best_val": b2,
                   "fno_params": count_params(fno),
                   "trm_params": count_params(trm)}, f, indent=2)


if __name__ == "__main__":
    main()
