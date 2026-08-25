"""Generate 1D viscous Burgers' equation dataset with a pseudospectral solver.

Equation:  du/dt + u * du/dx = nu * d2u/dx2,  x in [0, 2*pi), periodic BC.
Initial condition: random band-limited Fourier modes (same family as the FNO
paper's Burgers benchmark). Reference solver: Fourier spectral derivative +
RK4 with a viscosity-aware adaptive dt, then snapshots at t_out.

This solver is BOTH the data generator and the "traditional numerical method"
baseline in the article.

Outputs (saved to data/):
  burgers_train.npz    a:[N_TRAIN,128] u:[N_TRAIN,128]     nu=0.1
  burgers_test.npz     a:[N_TEST,128]  u:[N_TEST,128]      nu=0.1
  burgers_fine_test.npz a:[N_TEST,512] u:[N_TEST,512]      nu=0.1 (zero-shot)
  burgers_lownu_test.npz a:[N_TEST,512] u:[N_TEST,512]     nu=0.01 (extrapolation)
  burger_ics.npz       the shared 120 test ICs at all resolutions

Fixed seeds. Pure CPU. A few minutes total.
"""

import os
import numpy as np

SEED = 20260825
N_TRAIN = 500
N_TEST = 120
N_GRID = 128
N_FINE = 512
T_FINAL = 1.0
N_MODES = 8
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def make_ic(rng, n_grid, n_modes=N_MODES):
    """Random band-limited initial condition on [0, 2*pi)."""
    x = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    u0 = np.zeros(n_grid)
    for _ in range(n_modes):
        k = int(rng.integers(1, 5))
        amp = rng.uniform(-1.0, 1.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        u0 += amp * (np.sin(k * x + phase) if rng.random() < 0.5
                     else np.cos(k * x + phase))
    u0 = u0 / (np.max(np.abs(u0)) + 1e-8)
    u0 = 0.5 * u0 + 0.1 * np.sin(x)
    return u0.astype(np.float64)


def solve_burgers(u0, nu, n_grid, t_final=T_FINAL, n_out=21):
    """Pseudospectral RK4, returns u(x, t_final) and the snapshot matrix.

    dt is chosen from CFL and the viscous stability limit nu*k_max^2*dt < 1.
    Output snapshots are linearly interpolated onto t_out grid.
    """
    x = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    k = np.fft.fftfreq(n_grid, d=2.0 * np.pi / n_grid) * 2.0 * np.pi
    ik = 1j * k
    dealias = np.abs(k) <= (2.0 / 3.0) * (n_grid / 2.0)

    def rhs(u_hat):
        u = np.fft.ifft(u_hat).real
        ux = np.fft.ifft(ik * u_hat).real
        nl = np.fft.fft(u * ux)
        nl[~dealias] = 0.0
        return -nl - nu * (k * k) * u_hat

    u_hat = np.fft.fft(u0).astype(np.complex128)
    k_max = n_grid / 2.0
    dt_cfl = 2.0 * np.pi / n_grid / (np.max(np.abs(u0)) + 1e-8)
    dt_vis = 0.8 / (nu * k_max * k_max + 1e-12)
    dt = min(dt_cfl, dt_vis, 1e-2)

    t_out = np.linspace(0.0, t_final, n_out)
    t = 0.0
    snap = np.zeros((n_out, n_grid))
    snap[0] = u0
    idx = 1
    while idx < n_out:
        h = min(dt, t_out[idx] - t)
        k1 = rhs(u_hat)
        k2 = rhs(u_hat + 0.5 * h * k1)
        k3 = rhs(u_hat + 0.5 * h * k2)
        k4 = rhs(u_hat + h * k3)
        u_hat = u_hat + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        if t >= t_out[idx] - 1e-12:
            u = np.fft.ifft(u_hat).real
            snap[idx] = u
            idx += 1
    return snap, x


def main():
    rng = np.random.default_rng(SEED)

    def build(n, n_grid, nu, seed_off):
        r = np.random.default_rng(SEED + seed_off)
        a = np.zeros((n, n_grid))
        u = np.zeros((n, n_grid))
        for i in range(n):
            u0 = make_ic(r, n_grid)
            snap, _ = solve_burgers(u0, nu, n_grid)
            a[i] = u0
            u[i] = snap[-1]
        return a, u

    print("train set (nu=0.1, 128) ...")
    a_tr, u_tr = build(N_TRAIN, N_GRID, 0.1, 0)
    np.savez(os.path.join(DATA_DIR, "burgers_train.npz"), a=a_tr, u=u_tr)

    print("test set (nu=0.1, 128) ...")
    a_te, u_te = build(N_TEST, N_GRID, 0.1, 1000)
    np.savez(os.path.join(DATA_DIR, "burgers_test.npz"), a=a_te, u=u_te)

    print("fine test set (nu=0.1, 512) ...")
    a_fi, u_fi = build(N_TEST, N_FINE, 0.1, 2000)
    np.savez(os.path.join(DATA_DIR, "burgers_fine_test.npz"), a=a_fi, u=u_fi)

    print("low-nu test set (nu=0.01, 512) ...")
    a_ln, u_ln = build(N_TEST, N_FINE, 0.01, 3000)
    np.savez(os.path.join(DATA_DIR, "burgers_lownu_test.npz"), a=a_ln, u=u_ln)

    np.savez(os.path.join(DATA_DIR, "burger_ics.npz"),
             a128=a_te, a512=a_fi)

    # sanity: solution should have moved
    rel = np.linalg.norm(u_te - a_te) / np.linalg.norm(a_te)
    print(f"train {a_tr.shape} test {a_te.shape} fine {a_fi.shape} lownu {a_ln.shape}")
    print(f"|u(T)-u0|_rel on test: {rel:.3f} (should be >0)")
    # lownu should have sharper gradients -> bigger movement
    rel_ln = np.linalg.norm(u_ln - a_ln) / np.linalg.norm(a_ln)
    print(f"|u(T)-u0|_rel on lownu test: {rel_ln:.3f}")


if __name__ == "__main__":
    main()
