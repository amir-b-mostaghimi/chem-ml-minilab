#!/usr/bin/env python3
"""Mini-Lab #01: Can a Neural Network Learn to Count?

Compares standard neural architectures (MLP, GRU, Mini-Transformer) against a
Physics-Informed Translation-Equivariant Network (PI-Net) on variable-length
integer counting sequences:
    [1, 2, 3] -> 4,   [10, 11, 12, 13, 14] -> 15,   [18, 19, 20] -> 21

All models are trained strictly in-distribution on numbers in [1, 50] and
evaluated both in-distribution and across three out-of-distribution (OOD)
extrapolation regimes: [60, 150], [500, 1000], and [5000, 10000].

Outputs:
  - results.json
  - fig_counting_extrapolation.png (300 DPI editorial 2-panel figure)
  - fig_data_efficiency.png (300 DPI sample-efficiency curve)
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_minilab")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

SEED = 42
MAX_LEN = 6
MIN_LEN = 2
OUT_DIR = Path(__file__).resolve().parent


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(n_samples: int, low: int, high: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate variable-length consecutive integer sequences in [low, high].

    Returns:
      x_pad: (N, MAX_LEN) float tensor, right-aligned (last element is always the final token)
      mask:  (N, MAX_LEN) float tensor, 1.0 for real tokens, 0.0 for left-padding
      y:     (N, 1) float tensor, next integer after the final token
    """
    rng = random.Random(seed)
    xs, masks, ys = [], [], []
    for _ in range(n_samples):
        length = rng.randint(MIN_LEN, MAX_LEN)
        start = rng.randint(low, high - length)
        seq = [float(start + i) for i in range(length)]
        target = float(start + length)

        pad_n = MAX_LEN - length
        xs.append([0.0] * pad_n + seq)
        masks.append([0.0] * pad_n + [1.0] * length)
        ys.append([target])

    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(masks, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
    )


class StandardMLP(nn.Module):
    """Feedforward MLP taking normalized right-aligned sequence + mask."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MAX_LEN * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Standard input scaling by training range (~50)
        x_norm = (x / 50.0) * mask
        feat = torch.cat([x_norm, mask], dim=-1)
        return self.net(feat) * 50.0


class CountingGRU(nn.Module):
    """Recurrent GRU processing the variable-length sequence step by step."""

    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_norm = (x / 50.0) * mask
        inp = torch.stack([x_norm, mask], dim=-1)  # (B, T, 2)
        out, _ = self.gru(inp)
        last_h = out[:, -1, :]
        return self.head(last_h) * 50.0


class CountingTransformer(nn.Module):
    """Compact Transformer encoder over (value, mask) tokens."""

    def __init__(self, d_model: int = 32, nhead: int = 4) -> None:
        super().__init__()
        self.in_proj = nn.Linear(2, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, MAX_LEN, d_model) * 0.05)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_norm = (x / 50.0) * mask
        inp = torch.stack([x_norm, mask], dim=-1)
        h = self.in_proj(inp) + self.pos_emb
        key_padding_mask = mask == 0
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.head(h[:, -1, :]) * 50.0


class PhysicsInformedCountingNet(nn.Module):
    """Translation-equivariant neural network: f(x + c) = f(x) + c.

    Physical inductive bias:
      1. Shift coordinates into the local frame relative to the last observation:
             x_rel = (x - x_last) * mask
         so the representation is strictly invariant to global origin shifts.
      2. Predict the local increment delta_y = g_theta(x_rel, mask).
      3. Reconstruct y_hat = x_last + delta_y (exact translation equivariance).
    """

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(MAX_LEN * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_last = x[:, -1:]  # (B, 1)
        x_rel = (x - x_last) * mask
        feat = torch.cat([x_rel, mask], dim=-1)
        delta = self.delta_net(feat)
        return x_last + delta


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    m_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int = 900,
    lr: float = 3e-3,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_train, m_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
        scheduler.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate_mae(model: nn.Module, x: torch.Tensor, m: torch.Tensor, y: torch.Tensor) -> float:
    pred = model(x, m)
    return float(torch.mean(torch.abs(pred - y)).item())


@torch.no_grad()
def predict_single(model: nn.Module, seq: list[int]) -> float:
    pad_n = MAX_LEN - len(seq)
    x = torch.tensor([[0.0] * pad_n + [float(v) for v in seq]], dtype=torch.float32)
    m = torch.tensor([[0.0] * pad_n + [1.0] * len(seq)], dtype=torch.float32)
    return float(model(x, m).item())


def make_figures(
    trained_models: dict[str, nn.Module],
    regime_maes: dict[str, dict[str, float]],
    param_counts: dict[str, int],
    efficiency_data: dict[str, list[float]],
    sample_sizes: list[int],
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cfc9bc",
        "axes.linewidth": 0.9,
        "figure.facecolor": "#faf9f6",
        "axes.facecolor": "#ffffff",
    })

    palette = {
        "MLP (ReLU)": "#9a3412",
        "GRU (RNN)": "#b45309",
        "Transformer": "#64748b",
        "Physics-Informed Net": "#1e3a5f",
    }
    linestyles = {
        "MLP (ReLU)": "--",
        "GRU (RNN)": "-.",
        "Transformer": ":",
        "Physics-Informed Net": "-",
    }

    # --- FIGURE 1: 2-Panel Extrapolation & Regime MAE ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=300)
    fig.patch.set_facecolor("#faf9f6")

    # Panel A: Sweep len-3 sequences [k-2, k-1, k] -> k+1 for k in 3..180
    ks = list(range(3, 181, 2))
    true_vals = [k + 1 for k in ks]

    ax1.axvspan(1, 50, color="#edf2f7", alpha=0.85, label="Training range [1–50]")
    ax1.axvline(50, color="#1e3a5f", linestyle=":", linewidth=1.1, alpha=0.7)
    ax1.plot(true_vals, true_vals, color="#18181b", linewidth=1.2, alpha=0.35, label="Ideal (y = x)")

    for name, model in trained_models.items():
        preds = [predict_single(model, [k - 2, k - 1, k]) for k in ks]
        lw = 2.6 if "Physics" in name else 1.9
        ax1.plot(
            true_vals,
            preds,
            label=f"{name} ({param_counts[name]:,} params)",
            color=palette[name],
            linestyle=linestyles[name],
            linewidth=lw,
        )

    ax1.text(25, 168, "In-Distribution\n(Trained)", ha="center", va="top", fontsize=8.5, color="#1e3a5f", fontweight="bold")
    ax1.text(115, 168, "Out-of-Distribution Extrapolation\n(Never Seen in Training)", ha="center", va="top", fontsize=8.5, color="#9a3412", fontweight="bold")
    ax1.set_xlim(0, 185)
    ax1.set_ylim(0, 185)
    ax1.set_xlabel("True Next Number  (e.g., [100, 101, 102] → 103)", fontsize=9.5, color="#18181b", fontweight="medium")
    ax1.set_ylabel("Predicted Next Number", fontsize=9.5, color="#18181b", fontweight="medium")
    ax1.set_title("A. Extrapolation Beyond Training Range [1–50]", fontsize=11, fontweight="bold", color="#18181b", pad=10)
    ax1.grid(True, color="#ebe8e0", linewidth=0.7)
    ax1.legend(loc="lower right", fontsize=7.6, frameon=True, facecolor="#ffffff", edgecolor="#e4e1d8")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Grouped Bar Chart of Log-Scale MAE across 4 Regimes
    regimes = ["ID [1–50]", "Near OOD\n[60–150]", "Far OOD\n[500–1k]", "Extreme OOD\n[5k–10k]"]
    regime_keys = ["id_1_50", "ood_60_150", "ood_500_1000", "ood_5000_10000"]
    model_names = list(trained_models.keys())
    x = np.arange(len(regimes))
    width = 0.19

    for i, name in enumerate(model_names):
        vals = [max(regime_maes[name][rk], 1e-5) for rk in regime_keys]
        offset = (i - 1.5) * width
        bars = ax2.bar(
            x + offset,
            vals,
            width=width * 0.92,
            label=name,
            color=palette[name],
            edgecolor="#ffffff",
            linewidth=0.5,
        )
        if "Physics" in name:
            for bar, v in zip(bars, vals):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    v * 1.45,
                    f"{v:.1e}" if v < 0.01 else f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    color="#1e3a5f",
                    fontweight="bold",
                )

    ax2.set_yscale("log")
    ax2.set_ylim(1e-5, 3e4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes, fontsize=8.8)
    ax2.set_ylabel("Mean Absolute Error (MAE, log scale)", fontsize=9.5, color="#18181b", fontweight="medium")
    ax2.set_title("B. Error Across Distribution Shifts (Lower is Better)", fontsize=11, fontweight="bold", color="#18181b", pad=10)
    ax2.grid(True, axis="y", color="#ebe8e0", linewidth=0.7)
    ax2.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor="#ffffff", edgecolor="#e4e1d8")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Can a Neural Network Learn to Count? Translation Symmetry Enables Exact Extrapolation",
        fontsize=12.5,
        fontweight="bold",
        color="#18181b",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Amir Mostaghimi, Ph.D. · ChemML Mini-Lab #01 · amir-b-mostaghimi.github.io",
        ha="center",
        fontsize=8,
        color="#71717a",
    )
    plt.tight_layout(rect=[0, 0.035, 1, 0.94])
    fig.savefig(OUT_DIR / "fig_counting_extrapolation.png", dpi=300)
    plt.close(fig)

    # --- FIGURE 2: Data Efficiency Curve ---
    fig2, ax = plt.subplots(figsize=(7.6, 4.4), dpi=300)
    fig2.patch.set_facecolor("#faf9f6")

    for name in model_names:
        vals = [max(v, 1e-5) for v in efficiency_data[name]]
        lw = 2.6 if "Physics" in name else 1.8
        ax.plot(
            sample_sizes,
            vals,
            marker="o",
            markersize=5.5,
            label=f"{name} ({param_counts[name]:,} params)",
            color=palette[name],
            linestyle=linestyles[name],
            linewidth=lw,
        )

    ax.set_yscale("log")
    ax.set_ylim(1e-5, 2e2)
    ax.set_xticks(sample_sizes)
    ax.set_xlabel("Number of Training Sequences (sampled from [1–50])", fontsize=9.5, fontweight="medium")
    ax.set_ylabel("Near-OOD [60–150] MAE (log scale)", fontsize=9.5, fontweight="medium")
    ax.set_title("Data Efficiency: Physics-Informed Symmetry vs. Standard Models", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, color="#ebe8e0", linewidth=0.7)
    ax.legend(loc="center right", fontsize=8, frameon=True, facecolor="#ffffff", edgecolor="#e4e1d8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig2.text(
        0.5,
        0.01,
        "Amir Mostaghimi, Ph.D. · ChemML Mini-Lab #01 · amir-b-mostaghimi.github.io",
        ha="center",
        fontsize=8,
        color="#71717a",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig2.savefig(OUT_DIR / "fig_data_efficiency.png", dpi=300)
    plt.close(fig2)


def main() -> None:
    set_seed(SEED)

    # Datasets
    x_train, m_train, y_train = make_dataset(200, 1, 50, seed=100)
    test_sets = {
        "id_1_50": make_dataset(100, 1, 50, seed=201),
        "ood_60_150": make_dataset(100, 60, 150, seed=202),
        "ood_500_1000": make_dataset(100, 500, 1000, seed=203),
        "ood_5000_10000": make_dataset(100, 5000, 10000, seed=204),
    }

    factories = {
        "MLP (ReLU)": lambda: StandardMLP(hidden=64),
        "GRU (RNN)": lambda: CountingGRU(hidden=48),
        "Transformer": lambda: CountingTransformer(d_model=32, nhead=4),
        "Physics-Informed Net": lambda: PhysicsInformedCountingNet(hidden=16),
    }

    trained_models: dict[str, nn.Module] = {}
    regime_maes: dict[str, dict[str, float]] = {}
    param_counts: dict[str, int] = {}

    for name, fn in factories.items():
        set_seed(SEED)
        model = fn()
        param_counts[name] = count_params(model)
        train_model(model, x_train, m_train, y_train, epochs=450, lr=5e-3)
        trained_models[name] = model
        regime_maes[name] = {
            rk: evaluate_mae(model, *ds) for rk, ds in test_sets.items()
        }
        print(f"Trained {name} ({param_counts[name]} params): ID MAE={regime_maes[name]['id_1_50']:.4f}, Near-OOD MAE={regime_maes[name]['ood_60_150']:.4f}", flush=True)

    # Example concrete predictions
    sample_probes = [
        [1, 2, 3],
        [10, 11, 12, 13, 14],
        [18, 19, 20],
        [100, 101, 102],
        [742, 743, 744, 745],
        [9997, 9998, 9999],
    ]
    probe_results = []
    for seq in sample_probes:
        row = {"input": seq, "true_next": seq[-1] + 1, "predictions": {}}
        for name, model in trained_models.items():
            row["predictions"][name] = round(predict_single(model, seq), 3)
        probe_results.append(row)

    # Sample efficiency sweep on Near-OOD [60-150]
    sample_sizes = [10, 25, 50, 100, 200]
    efficiency_data: dict[str, list[float]] = {name: [] for name in factories}
    for n_tr in sample_sizes:
        xt, mt, yt = make_dataset(n_tr, 1, 50, seed=300 + n_tr)
        for name, fn in factories.items():
            set_seed(SEED)
            m = fn()
            train_model(m, xt, mt, yt, epochs=350, lr=5e-3)
            efficiency_data[name].append(evaluate_mae(m, *test_sets["ood_60_150"]))

    make_figures(trained_models, regime_maes, param_counts, efficiency_data, sample_sizes)

    summary = {
        "seed": SEED,
        "train_range": [1, 50],
        "train_samples": 200,
        "param_counts": param_counts,
        "regime_maes": regime_maes,
        "probe_predictions": probe_results,
        "sample_sizes": sample_sizes,
        "ood_efficiency_mae": efficiency_data,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
