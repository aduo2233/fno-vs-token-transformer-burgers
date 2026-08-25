"""Evaluation experiments.

Exp A (resolution / tokenization): train at 128, evaluate at 128/256/512.
  FNO is resolution-invariant (spectral convolution); the TokenTransformer
  is bound to 128 tokens and needs input interpolation to run elsewhere.

Exp B (multi-scale / extrapolation): trained on nu=0.1 smooth solutions,
  tested on nu=0.01 solutions with developing shocks.

Exp C (vs classical): FNO inference error AND wall-clock vs the
  pseudospectral reference solver on the same test ICs.

Saves results/results.json and prints a table.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models import FNO1d, TokenTransformer
from gen_burgers import solve_burgers

SEED = 20260825
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_DIR, exist_ok=True)
torch.manual_seed(SEED)


def load_models():
    fno = FNO1d(modes=16, width=32)
    trm = TokenTransformer(n_grid=128)
    fno.load_state_dict(torch.load(os.path.join(DATA_DIR, "fno.pt"),
                                   map_location="cpu"))
    trm.load_state_dict(torch.load(os.path.join(DATA_DIR, "transformer.pt"),
                                   map_location="cpu"))
    fno.eval()
    trm.eval()
    return fno, trm


def rel_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def interp1d_np(y, n_out):
    x_in = np.linspace(0, 1, y.shape[-1])
    x_out = np.linspace(0, 1, n_out)
    return np.stack([np.interp(x_out, x_in, yi) for yi in y])


def run_resolution(fno, trm):
    """Exp A: zero-shot resolution transfer."""
    d128 = np.load(os.path.join(DATA_DIR, "burgers_test.npz"))
    d512 = np.load(os.path.join(DATA_DIR, "burgers_fine_test.npz"))
    a128, u128 = d128["a"], d128["u"]
    a512, u512 = d512["a"], d512["u"]
    # 256 reference: spectral solve on 256 grid (fresh, exact)
    a256 = np.stack([np.interp(np.linspace(0, 2 * np.pi, 256, endpoint=False),
                               np.linspace(0, 2 * np.pi, 512, endpoint=False), ai)
                     for ai in a512])
    u256 = np.zeros_like(a256)
    for i in range(len(a256)):
        snap, _ = solve_burgers(a256[i], 0.1, 256)
        u256[i] = snap[-1]

    def fno_eval(a_ref):
        with torch.no_grad():
            pred = fno(torch.from_numpy(a_ref).float().unsqueeze(-1)).numpy()
        return pred

    def trm_eval_at128(a_ref, n_out):
        # interpolate input down to 128 tokens, run, interpolate output back
        a128i = interp1d_np(a_ref, 128)
        with torch.no_grad():
            pred = trm(torch.from_numpy(a128i).float().unsqueeze(-1)).numpy()
        return interp1d_np(pred, n_out)

    rows = {}
    for name, a_ref, u_ref, n in [("128", a128, u128, 128),
                                  ("256", a256, u256, 256),
                                  ("512", a512, u512, 512)]:
        p_fno = fno_eval(a_ref)
        p_trm = trm_eval_at128(a_ref, n)
        rows[name] = {
            "fno_rel_l2": rel_l2(p_fno, u_ref),
            "trm_rel_l2": rel_l2(p_trm, u_ref),
        }
        print(f"ExpA [{name:>3}] FNO {rows[name]['fno_rel_l2']:.4f} | "
              f"Transformer {rows[name]['trm_rel_l2']:.4f}")
    return rows


def run_extrapolation(fno, trm):
    """Exp B: trained nu=0.1, tested nu=0.01 at 512."""
    d = np.load(os.path.join(DATA_DIR, "burgers_lownu_test.npz"))
    a, u = d["a"], d["u"]

    def fno_eval(a_ref):
        with torch.no_grad():
            return fno(torch.from_numpy(a_ref).float().unsqueeze(-1)).numpy()

    p_fno = fno_eval(a)
    p_trm = trm_eval(a)
    rows = {
        "lownu_512": {
            "fno_rel_l2": rel_l2(p_fno, u),
            "trm_rel_l2": rel_l2(p_trm, u),
        }
    }
    print(f"ExpB [lownu 512] FNO {rows['lownu_512']['fno_rel_l2']:.4f} | "
          f"Transformer {rows['lownu_512']['trm_rel_l2']:.4f}")
    return rows


def trm_eval(a_ref):
    a128i = interp1d_np(a_ref, 128)
    with torch.no_grad():
        return interp1d_np(
            trm(torch.from_numpy(a128i).float().unsqueeze(-1)).numpy(),
            a_ref.shape[-1])


def run_classical(fno):
    """Exp C: FNO vs pseudospectral solver: error + wall-clock on 20 ICs."""
    d = np.load(os.path.join(DATA_DIR, "burgers_test.npz"))
    a128, u128 = d["a"], d["u"]
    n = 20
    a = a128[:n]
    u_ref = u128[:n]

    # classical: fresh spectral solve (that's the same code that generated data)
    t0 = time.perf_counter()
    u_class = np.zeros_like(u_ref)
    for i in range(n):
        snap, _ = solve_burgers(a[i], 0.1, 128)
        u_class[i] = snap[-1]
    t_class = time.perf_counter() - t0

    # FNO: batched inference, warm up once
    with torch.no_grad():
        for _ in range(3):
            fno(torch.from_numpy(a[:8]).float().unsqueeze(-1))
        t0 = time.perf_counter()
        p_fno = fno(torch.from_numpy(a).float().unsqueeze(-1)).numpy()
        t_fno = time.perf_counter() - t0

    rows = {
        "classical_vs_fno": {
            "classical_time_s_20ics": t_class,
            "fno_time_s_20ics": t_fno,
            "classical_rel_l2_vs_ref": rel_l2(u_class, u_ref),
            "fno_rel_l2_vs_ref": rel_l2(p_fno, u_ref),
            "speedup": t_class / max(t_fno, 1e-9),
        }
    }
    print(f"ExpC classical {t_class:.3f}s (err "
          f"{rows['classical_vs_fno']['classical_rel_l2_vs_ref']:.2e}) | "
          f"FNO {t_fno:.4f}s (err {rows['classical_vs_fno']['fno_rel_l2_vs_ref']:.4f}) "
          f"speedup {rows['classical_vs_fno']['speedup']:.0f}x")
    return rows


if __name__ == "__main__":
    fno, trm = load_models()
    out = {}
    out.update(run_resolution(fno, trm))
    out.update(run_extrapolation(fno, trm))
    out.update(run_classical(fno))
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/results.json")
