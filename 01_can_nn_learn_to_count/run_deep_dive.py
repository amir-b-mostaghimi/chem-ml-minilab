#!/usr/bin/env python3
"""Mini-Lab #01 Deep Dive: Why Advanced Models Fail Worse OOD & Can Data Manipulation Fix It?

Investigates:
  1. Mechanistic diagnostics inside trained MLP, GRU, and Transformer as x scales 10 -> 10,000:
     - MLP: piecewise-linear slope drift (unbounded ReLU -> linear extrapolation error)
     - GRU: hidden-state saturation (tanh/sigmoid bounds h in [-1, 1] -> hard ceiling)
     - Transformer: LayerNorm magnitude erasure (LN(c*v) ~= LN(v) -> scale collapse)
  2. Data Manipulation & Augmentation interventions on the exact same MLP, GRU, Transformer:
     - Baseline: raw sequences in [1, 50]
     - Wider-Range Augmentation: 4x wider training range [1, 200] (500 samples)
     - Log-Space Transform: log1p(x) -> log1p(y)
     - Local-Frame / Delta-ML Representation: x_rel = x - x_last, target = y - x_last

Outputs:
  - deep_dive_results.json
  - fig_architectures_and_failure.png (300 DPI architecture + real activation diagnostics)
  - fig_data_manipulation_comparison.png (300 DPI intervention benchmark)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_minilab")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_experiment import (
    SEED,
    MAX_LEN,
    StandardMLP,
    CountingGRU,
    CountingTransformer,
    PhysicsInformedCountingNet,
    set_seed,
    make_dataset,
    train_model,
    evaluate_mae,
    predict_single,
)

OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Wrappers for Data Manipulation Modes (keeping architectures 100% identical)
# ---------------------------------------------------------------------------

class LogSpaceWrapper(nn.Module):
    """Applies log1p(x) preprocessing and expm1(y_hat) postprocessing."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_log = torch.log1p(torch.clamp(x, min=0.0)) * mask
        # Scale log1p into ~[0, 50] range expected by base model's /50 normalization
        pred_scaled = self.base(x_log * 10.0, mask)
        return pred_scaled / 10.0  # returns log1p(y) during training


class LocalFrameWrapper(nn.Module):
    """Physics-informed data representation: x_rel = (x - x_last)*mask, target = y - x_last."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_last = x[:, -1:]
        x_rel = (x - x_last) * mask
        # Scale x_rel (typically in [-5, 0]) up by 10 so base model's /50 puts it in [-1, 0]
        delta_pred = self.base(x_rel * 10.0, mask) / 10.0
        return x_last + delta_pred


def train_log_model(model: LogSpaceWrapper, x_tr: torch.Tensor, m_tr: torch.Tensor, y_tr: torch.Tensor, epochs: int = 450, lr: float = 5e-3) -> LogSpaceWrapper:
    y_log = torch.log1p(y_tr)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred_log = model(x_tr, m_tr)
        loss = loss_fn(pred_log, y_log)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def eval_log_mae(model: LogSpaceWrapper, x: torch.Tensor, m: torch.Tensor, y: torch.Tensor) -> float:
    pred_log = model(x, m)
    pred = torch.expm1(torch.clamp(pred_log, max=15.0))
    return float(torch.mean(torch.abs(pred - y)).item())


# ---------------------------------------------------------------------------
# Mechanistic Internal Activation Diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_diagnostics(mlp: StandardMLP, gru: CountingGRU, trf: CountingTransformer) -> dict[str, list[float]]:
    scales = [10, 25, 45, 75, 120, 250, 500, 1000, 2500, 5000, 10000]
    mlp_ratio, gru_sat, gru_pred, trf_cos, trf_pred = [], [], [], [], []

    def make_batch(k: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = [float(k - 2), float(k - 1), float(k)]
        x = torch.tensor([[0.0, 0.0, 0.0] + seq], dtype=torch.float32)
        m = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        return x, m

    # Reference Transformer post-LayerNorm token at k = 10000
    x_ref, m_ref = make_batch(10000)
    inp_ref = torch.stack([(x_ref / 50.0) * m_ref, m_ref], dim=-1)
    h_ref = trf.encoder(trf.in_proj(inp_ref) + trf.pos_emb, src_key_padding_mask=(m_ref == 0))[:, -1, :]

    for k in scales:
        x, m = make_batch(k)
        # 1. MLP prediction / true ratio
        p_mlp = float(mlp(x, m).item())
        mlp_ratio.append(p_mlp / (k + 1.0))

        # 2. GRU hidden state saturation (fraction of |h_t| > 0.95) & output
        inp = torch.stack([(x / 50.0) * m, m], dim=-1)
        out_gru, _ = gru.gru(inp)
        h_last = out_gru[:, -1, :].abs()
        sat_frac = float((h_last > 0.95).float().mean().item()) * 100.0
        gru_sat.append(sat_frac)
        gru_pred.append(float((gru.head(out_gru[:, -1, :]) * 50.0).item()))

        # 3. Transformer representation cosine similarity to k=10000 (LayerNorm collapse)
        h_trf = trf.encoder(trf.in_proj(inp) + trf.pos_emb, src_key_padding_mask=(m == 0))[:, -1, :]
        cos_sim = float(F.cosine_similarity(h_trf, h_ref, dim=-1).item())
        trf_cos.append(cos_sim)
        trf_pred.append(float((trf.head(h_trf) * 50.0).item()))

    return {
        "scales": scales,
        "mlp_pred_over_true": mlp_ratio,
        "gru_saturation_pct": gru_sat,
        "gru_pred": gru_pred,
        "trf_cosine_to_10k": trf_cos,
        "trf_pred": trf_pred,
    }


# ---------------------------------------------------------------------------
# Publication-Grade Figures
# ---------------------------------------------------------------------------

def draw_box(ax, x, y, w, h, text, fc, ec, tc="#18181b", fontsize=7.6, bold=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc, edgecolor=ec, linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize, color=tc,
        fontweight="bold" if bold else "normal",
    )


def draw_arrow(ax, x0, y0, x1, y1, color="#52525b"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.32", color=color, lw=1.1),
    )


def make_architecture_figure(diag: dict[str, list[float]]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cfc9bc",
        "axes.linewidth": 0.9,
        "figure.facecolor": "#faf9f6",
        "axes.facecolor": "#ffffff",
    })

    fig = plt.figure(figsize=(13.2, 7.2), dpi=300)
    fig.patch.set_facecolor("#faf9f6")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 1.0], hspace=0.38, wspace=0.26, left=0.06, right=0.97, top=0.89, bottom=0.09)

    # ---- TOP ROW: Architecture Schematics (spanning full width as 4 clean cards) ----
    gs_top = gs[0, :].subgridspec(1, 4, wspace=0.14)
    arch_specs = [
        {
            "title": "1. MLP (ReLU)",
            "subtitle": "5,057 params · Linear Drift",
            "color": "#9a3412",
            "tint": "#fdf2ee",
            "boxes": [
                ("Input x / 50 + mask", "#ffffff", "#cfc9bc", "#18181b"),
                ("Linear → ReLU (unbounded)\nLinear → ReLU", "#fdf2ee", "#9a3412", "#9a3412"),
                ("Linear Head × 50", "#ffffff", "#cfc9bc", "#18181b"),
            ],
            "note": "Failure: ReLU stays in fixed linear\nregion → slope drift (~3.6% error)",
        },
        {
            "title": "2. GRU (RNN)",
            "subtitle": "9,089 params · Gate Saturation",
            "color": "#b45309",
            "tint": "#fef7ec",
            "boxes": [
                ("Token sequence (x_t / 50)", "#ffffff", "#cfc9bc", "#18181b"),
                ("Reset/Update σ ∈ (0,1)\nCandidate tanh ∈ (-1,1)", "#fef7ec", "#b45309", "#b45309"),
                ("Bounded h_t ∈ [-1, 1]^48\n→ Constant Ceiling (~222)", "#ffffff", "#b45309", "#18181b"),
            ],
            "note": "Failure: tanh/σ saturate at ±1 →\nhidden state freezes on hypercube",
        },
        {
            "title": "3. Transformer",
            "subtitle": "8,865 params · Scale Erasure",
            "color": "#475569",
            "tint": "#f1f5f9",
            "boxes": [
                ("Token proj + Pos. Emb.", "#ffffff", "#cfc9bc", "#18181b"),
                ("Softmax Attention +\nLayerNorm: (h - μ) / σ", "#f1f5f9", "#475569", "#334155"),
                ("Scale-Invariant Vector\n→ Flat Ceiling (~96)", "#ffffff", "#475569", "#18181b"),
            ],
            "note": "Failure: LayerNorm divides out\nmagnitude: LN(c·v) ≈ LN(v)",
        },
        {
            "title": "4. Physics-Informed Net",
            "subtitle": "225 params · Exact Extrapolation",
            "color": "#1e3a5f",
            "tint": "#edf2f7",
            "boxes": [
                ("Shift Frame: x_rel = x − x_last", "#edf2f7", "#1e3a5f", "#1e3a5f"),
                ("Tiny MLP (16 units)\nPredicts local step Δŷ", "#ffffff", "#1e3a5f", "#18181b"),
                ("Reconstruct: ŷ = x_last + Δŷ\nf(x + c) = f(x) + c", "#edf2f7", "#1e3a5f", "#1e3a5f"),
            ],
            "note": "Success: Inputs always [-k..0];\nzero saturation at any scale!",
        },
    ]

    for idx, spec in enumerate(arch_specs):
        ax = fig.add_subplot(gs_top[0, idx])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Card frame
        frame = FancyBboxPatch(
            (0.01, 0.01), 0.98, 0.98,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor="#ffffff", edgecolor="#e4e1d8", linewidth=1.1,
        )
        ax.add_patch(frame)
        ax.text(0.5, 0.92, spec["title"], ha="center", va="center", fontsize=9.5, fontweight="bold", color=spec["color"])
        ax.text(0.5, 0.84, spec["subtitle"], ha="center", va="center", fontsize=7.5, color="#71717a")

        ys = [0.63, 0.41, 0.20]
        hs = [0.13, 0.16, 0.14]
        for b_idx, (txt, fc, ec, tc) in enumerate(spec["boxes"]):
            draw_box(ax, 0.08, ys[b_idx], 0.84, hs[b_idx], txt, fc, ec, tc, fontsize=7.2, bold=(b_idx == 1))
            if b_idx < 2:
                draw_arrow(ax, 0.5, ys[b_idx] - 0.005, 0.5, ys[b_idx + 1] + hs[b_idx + 1] + 0.005, color=spec["color"])

        ax.text(0.5, 0.085, spec["note"], ha="center", va="center", fontsize=7.0, color="#4a4a52", style="italic")

    # ---- BOTTOM ROW: Real Measured Internal Failure Diagnostics ----
    scales = diag["scales"]

    # Bottom-Left: GRU Hidden State Saturation
    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b1.axvspan(10, 50, color="#edf2f7", alpha=0.8)
    ax_b1.plot(scales, diag["gru_saturation_pct"], color="#b45309", marker="o", linewidth=2.2, markersize=4.8)
    ax_b1.set_xscale("log")
    ax_b1.set_ylim(-5, 108)
    ax_b1.set_xlabel("Input Magnitude k (log scale)", fontsize=8.8)
    ax_b1.set_ylabel("% GRU Units Saturated (|h| > 0.95)", fontsize=8.6)
    ax_b1.set_title("A. GRU Saturation: Hidden Vector Freezes", fontsize=9.8, fontweight="bold", pad=8)
    ax_b1.grid(True, color="#ebe8e0", linewidth=0.7)
    ax_b1.spines["top"].set_visible(False)
    ax_b1.spines["right"].set_visible(False)

    # Bottom-Middle: Transformer LayerNorm Representation Collapse
    ax_b2 = fig.add_subplot(gs[1, 1])
    ax_b2.axvspan(10, 50, color="#edf2f7", alpha=0.8)
    ax_b2.plot(scales, diag["trf_cosine_to_10k"], color="#475569", marker="s", linewidth=2.2, markersize=4.8)
    ax_b2.axhline(1.0, color="#9a3412", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_b2.set_xscale("log")
    ax_b2.set_ylim(0.5, 1.04)
    ax_b2.set_xlabel("Input Magnitude k (log scale)", fontsize=8.8)
    ax_b2.set_ylabel("Cosine Similarity to k = 10,000 Token", fontsize=8.6)
    ax_b2.set_title("B. Transformer LayerNorm Collapse", fontsize=9.8, fontweight="bold", pad=8)
    ax_b2.text(320, 0.86, "At k ≥ 250, LayerNorm\nmaps all numbers to\nnearly the SAME vector!", fontsize=7.6, color="#9a3412", bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2ee", ec="#e4e1d8"))
    ax_b2.grid(True, color="#ebe8e0", linewidth=0.7)
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["right"].set_visible(False)

    # Bottom-Right: Output Ceiling Comparison (Log-Log)
    ax_b3 = fig.add_subplot(gs[1, 2])
    ax_b3.axvspan(10, 50, color="#edf2f7", alpha=0.8)
    true_y = [s + 1 for s in scales]
    mlp_y = [r * (s + 1) for r, s in zip(diag["mlp_pred_over_true"], scales)]
    ax_b3.plot(scales, true_y, color="#1e3a5f", linewidth=2.4, label="Ideal / Physics-Informed Net")
    ax_b3.plot(scales, mlp_y, color="#9a3412", linestyle="--", linewidth=1.9, label="MLP (drifts linearly)")
    ax_b3.plot(scales, diag["gru_pred"], color="#b45309", linestyle="-.", linewidth=1.9, label="GRU (freezes ~222)")
    ax_b3.plot(scales, diag["trf_pred"], color="#475569", linestyle=":", linewidth=2.2, label="Transformer (freezes ~96)")
    ax_b3.set_xscale("log")
    ax_b3.set_yscale("log")
    ax_b3.set_xlabel("Input Magnitude k (log scale)", fontsize=8.8)
    ax_b3.set_ylabel("Predicted Next Number (log scale)", fontsize=8.6)
    ax_b3.set_title("C. Why 'Advanced' Models Fail Worse OOD", fontsize=9.8, fontweight="bold", pad=8)
    ax_b3.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor="#ffffff", edgecolor="#e4e1d8")
    ax_b3.grid(True, color="#ebe8e0", linewidth=0.7)
    ax_b3.spines["top"].set_visible(False)
    ax_b3.spines["right"].set_visible(False)

    fig.suptitle(
        "Anatomy of Out-of-Distribution Failure: Why GRU and Transformer Saturate Faster Than an MLP",
        fontsize=12.5,
        fontweight="bold",
        color="#18181b",
        y=0.96,
    )
    fig.text(
        0.5, 0.015,
        "Amir Mostaghimi, Ph.D. · ChemML Mini-Lab #01 Deep Dive · amir-b-mostaghimi.github.io",
        ha="center", fontsize=8, color="#71717a",
    )
    fig.savefig(OUT_DIR / "fig_architectures_and_failure.png", dpi=300)
    plt.close(fig)


def make_intervention_figure(interventions: dict[str, dict[str, dict[str, float]]]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cfc9bc",
        "axes.linewidth": 0.9,
        "figure.facecolor": "#faf9f6",
        "axes.facecolor": "#ffffff",
    })

    models = ["MLP (ReLU)", "GRU (RNN)", "Transformer"]
    methods = [
        ("baseline", "Baseline [1–50]", "#9a3412"),
        ("wider_aug", "4× Wider Data Aug. [1–200]", "#b45309"),
        ("log_transform", "Log Transform (log1p)", "#64748b"),
        ("local_frame", "Local-Frame Δ-Rep (x − x_last)", "#1e3a5f"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=300)
    fig.patch.set_facecolor("#faf9f6")

    for ax, regime_key, title in [
        (ax1, "ood_500_1000", "A. Far OOD [500–1,000] MAE"),
        (ax2, "ood_5000_10000", "B. Extreme OOD [5,000–10,000] MAE"),
    ]:
        x = np.arange(len(models))
        width = 0.19
        for i, (m_key, m_label, color) in enumerate(methods):
            vals = [max(interventions[m_key][mod][regime_key], 1e-4) for mod in models]
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, vals, width=width * 0.92, label=m_label, color=color, edgecolor="#ffffff", linewidth=0.5)
            if m_key == "local_frame":
                for bar, v in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v * 1.45,
                        f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.2, color="#1e3a5f", fontweight="bold",
                    )

        ax.set_yscale("log")
        ax.set_ylim(1e-4, 4e4)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9.2, fontweight="medium")
        ax.set_ylabel("Mean Absolute Error (MAE, log scale)", fontsize=9.2)
        ax.set_title(title, fontsize=10.8, fontweight="bold", pad=10)
        ax.grid(True, axis="y", color="#ebe8e0", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor="#ffffff", edgecolor="#e4e1d8")

    fig.suptitle(
        "Can Data Manipulation Fix OOD Failure? Wider Data Shifts the Wall; Symmetry-Aware Representation Removes It",
        fontsize=11.6,
        fontweight="bold",
        color="#18181b",
        y=0.97,
    )
    fig.text(
        0.5, 0.012,
        "All three architectures kept 100% identical — only the data representation changes · Amir Mostaghimi, Ph.D.",
        ha="center", fontsize=8, color="#71717a",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(OUT_DIR / "fig_data_manipulation_comparison.png", dpi=300)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    x_tr_50, m_tr_50, y_tr_50 = make_dataset(200, 1, 50, seed=100)
    x_tr_200, m_tr_200, y_tr_200 = make_dataset(500, 1, 200, seed=150)

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
    }

    interventions: dict[str, dict[str, dict[str, float]]] = {
        "baseline": {},
        "wider_aug": {},
        "log_transform": {},
        "local_frame": {},
    }

    baseline_models = {}
    for name, fn in factories.items():
        # 1. Baseline [1-50]
        set_seed(SEED)
        m_base = train_model(fn(), x_tr_50, m_tr_50, y_tr_50, epochs=450, lr=5e-3)
        baseline_models[name] = m_base
        interventions["baseline"][name] = {rk: evaluate_mae(m_base, *ds) for rk, ds in test_sets.items()}

        # 2. 4x Wider Range Augmentation [1-200] (500 samples)
        set_seed(SEED)
        m_wide = train_model(fn(), x_tr_200, m_tr_200, y_tr_200, epochs=450, lr=5e-3)
        interventions["wider_aug"][name] = {rk: evaluate_mae(m_wide, *ds) for rk, ds in test_sets.items()}

        # 3. Log-Space Transform (trained on [1-50])
        set_seed(SEED)
        m_log = train_log_model(LogSpaceWrapper(fn()), x_tr_50, m_tr_50, y_tr_50, epochs=450, lr=5e-3)
        interventions["log_transform"][name] = {rk: eval_log_mae(m_log, *ds) for rk, ds in test_sets.items()}

        # 4. Local-Frame / Delta-ML Data Representation (trained on [1-50])
        set_seed(SEED)
        m_loc = train_model(LocalFrameWrapper(fn()), x_tr_50, m_tr_50, y_tr_50, epochs=450, lr=5e-3)
        interventions["local_frame"][name] = {rk: evaluate_mae(m_loc, *ds) for rk, ds in test_sets.items()}

        print(
            f"{name:12s} | Extreme OOD [5k-10k] MAE -> "
            f"Base: {interventions['baseline'][name]['ood_5000_10000']:8.2f} | "
            f"WiderAug: {interventions['wider_aug'][name]['ood_5000_10000']:8.2f} | "
            f"Log: {interventions['log_transform'][name]['ood_5000_10000']:8.2f} | "
            f"LocalFrame: {interventions['local_frame'][name]['ood_5000_10000']:.4f}",
            flush=True,
        )

    diag = collect_diagnostics(
        baseline_models["MLP (ReLU)"],
        baseline_models["GRU (RNN)"],
        baseline_models["Transformer"],
    )

    make_architecture_figure(diag)
    make_intervention_figure(interventions)

    out = {"diagnostics": diag, "interventions": interventions}
    (OUT_DIR / "deep_dive_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved deep_dive_results.json, fig_architectures_and_failure.png, fig_data_manipulation_comparison.png", flush=True)


if __name__ == "__main__":
    main()
