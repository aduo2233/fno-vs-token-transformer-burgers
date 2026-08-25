"""Plot experiment results for the article. English labels, PNG at 2x."""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_resolution(res):
    """Bar chart: rel L2 at 128/256/512 for FNO vs TokenTransformer."""
    names = [n for n in res if n not in ("classical_vs_fno", "lownu_512")]
    fno = [res[n]["fno_rel_l2"] for n in names]
    trm = [res[n]["trm_rel_l2"] for n in names]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w / 2, fno, w, label="FNO (spectral conv)", color="#1f6feb")
    b2 = ax.bar(x + w / 2, trm, w, label="TokenTransformer (fixed tokens)",
                color="#f0883e")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2,
                                                  b.get_height()),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"grid {n}" for n in names])
    ax.set_ylabel("relative L2 error vs spectral reference")
    ax.set_title("Zero-shot resolution transfer (trained at grid 128)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig_resolution.png"), bbox_inches="tight")
    plt.close(fig)


def plot_curves():
    d = np.load(os.path.join(RESULT_DIR, "train_curves.npz"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogy(d["fno_val"], label="FNO val", color="#1f6feb")
    ax.semilogy(d["trm_val"], label="Transformer val", color="#f0883e")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (log)")
    ax.set_title("Validation loss during training (Burgers, u0 -> u(T))")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig_curves.png"), bbox_inches="tight")
    plt.close(fig)


def plot_classical(res):
    r = res["classical_vs_fno"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["pseudospectral\nsolver", "FNO inference"]
    times = [r["classical_time_s_20ics"], r["fno_time_s_20ics"]]
    errs = [r["classical_rel_l2_vs_ref"], r["fno_rel_l2_vs_ref"]]
    x = np.arange(2)
    ax.bar(x - 0.2, times, 0.4, color="#57606a")
    ax2 = ax.twinx()
    ax2.plot(x + 0.2, errs, "o", color="#cf222e", markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("wall-clock (s, 20 ICs)")
    ax2.set_ylabel("relative L2 vs reference (red dots)")
    ax.set_title(f"Speed vs accuracy: FNO {r['speedup']:.0f}x faster "
                 f"but err {r['fno_rel_l2_vs_ref']:.3f}")
    for i, t in enumerate(times):
        ax.annotate(f"{t:.3f}s", (i - 0.2, t), ha="center", va="bottom")
    for i, e in enumerate(errs):
        ax2.annotate(f"{e:.3f}", (i + 0.2, e), ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig_classical.png"), bbox_inches="tight")
    plt.close(fig)


def plot_extrapolation_example():
    """One sample: nu=0.01 shock vs FNO prediction (trained on nu=0.1)."""
    import torch
    from models import FNO1d
    d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "burgers_lownu_test.npz"))
    a, u = d["a"], d["u"]
    fno = FNO1d(modes=16, width=32)
    fno.load_state_dict(torch.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "data", "fno.pt"), map_location="cpu"))
    fno.eval()
    with torch.no_grad():
        pred = fno(torch.from_numpy(a[:1]).float().unsqueeze(-1)).numpy()[0]
    x = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, u[0], label="spectral reference (nu=0.01)", color="#0969da")
    ax.plot(x, pred, "--", label="FNO prediction (trained nu=0.1)",
            color="#cf222e")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Extrapolation: shock regime unseen in training")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig_extrapolation.png"),
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    with open(os.path.join(RESULT_DIR, "results.json")) as f:
        res = json.load(f)
    plot_resolution(res)
    plot_curves()
    plot_classical(res)
    plot_extrapolation_example()
    print("figures written to results/")
