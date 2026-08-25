"""Models used in the experiment.

1. FNO1D - Fourier Neural Operator (Li et al. 2020, arXiv:2010.08895).
   Spectral convolution in Fourier space -> resolution-invariant by design.
2. TokenTransformer - a vanilla Transformer encoder that treats each grid
   point as a token (learned positional embedding, no coordinates).
   Resolution-bound: it is trained on exactly N_GRID tokens.

Both map the initial condition u0(x) to the solution u(x, T).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, 2))

    def compl_mul1d(self, x, weights):
        # x: [B, C, M] complex; weights: [C_in, C_out, M, 2]
        w = torch.view_as_complex(weights)
        return torch.einsum("bcm,com->bom", x, w)

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.out_channels, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes], self.weights)
        return torch.fft.irfft(out_ft, n=N)


class FNO1d(nn.Module):
    """1D FNO: lifting -> 4 spectral blocks -> projection."""

    def __init__(self, modes=16, width=32, in_channels=1, out_channels=1):
        super().__init__()
        self.modes = modes
        self.width = width
        self.lifting = nn.Linear(in_channels, width)
        self.spectral = nn.ModuleList()
        for _ in range(4):
            self.spectral.append(SpectralConv1d(width, width, modes))
        self.ws = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(4)])
        self.projection = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, out_channels))

    def forward(self, x):
        # x: [B, N, C]
        x = x.transpose(1, 2)              # [B, C, N]
        x = self.lifting(x.transpose(1, 2)).transpose(1, 2)
        for sc, w in zip(self.spectral, self.ws):
            x = F.relu(sc(x) + w(x))
        x = x.transpose(1, 2)              # [B, N, C]
        return self.projection(x).squeeze(-1)


class TokenTransformer(nn.Module):
    """Vanilla Transformer encoder, grid points as tokens.

    Uses LEARNED positional embeddings -> tied to the training resolution.
    No coordinate information: this is the "naive tokenization" baseline
    that must fail resolution transfer.
    """

    def __init__(self, n_grid, d_model=64, nhead=4, n_layers=2,
                 in_channels=1, out_channels=1):
        super().__init__()
        self.n_grid = n_grid
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_grid, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
            batch_first=True, dropout=0.0, activation="relu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, out_channels))

    def forward(self, x):
        # x: [B, N, C]
        h = self.input_proj(x) + self.pos
        h = self.encoder(h)
        return self.output_proj(h).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 128, 1)
    fno = FNO1d(modes=16, width=32)
    tr = TokenTransformer(n_grid=128)
    print("FNO out", tuple(fno(x).shape), "params", count_params(fno))
    print("Transformer out", tuple(tr(x).shape), "params", count_params(tr))
