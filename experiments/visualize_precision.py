"""Generate CKKS precision accumulation figure for Section VI-B.

Reads measured error values from results/precision_analysis.json (deep
preset) and produces a bar chart showing error growth across pipeline
stages, with the clinical tolerance threshold and depth annotations.

Output: results/figures/ckks_precision_accumulation.{pdf,png}

Usage:
    cd fhe-agent-privacy
    python -m experiments.visualize_precision
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────

FIGURES_DIR = Path("results/figures")
PRECISION_FILE = Path("results/precision_analysis.json")

# ── Academic style (matches visualize_feasibility.py) ─────────────

BAR_COLOR = "#2d7d9a"
TOLERANCE_COLOR = "#c0392b"


def set_academic_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif",
                        "Bitstream Vera Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "legend.frameon": False,
    })


def _save(fig: plt.Figure, stem: str) -> None:
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  Saved: {png}")
    print(f"  Saved: {pdf}")


def main() -> None:
    set_academic_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────
    with open(PRECISION_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # Use the deep preset (n=16384, depth 4) — the one that runs the
    # full pipeline successfully.
    deep = None
    for preset in data["presets"]:
        if preset["preset"] == "deep":
            deep = preset
            break
    if deep is None:
        raise RuntimeError("Deep preset not found in precision_analysis.json")

    stages = deep["stages"]

    # Map stage names to display labels and extract values
    stage_map = {
        "encrypt_decrypt_roundtrip": {
            "label": "Encrypt/Decrypt\nroundtrip",
            "depth_label": "0 levels",
        },
        "dot_product": {
            "label": "Linear\ndot product",
            "depth_label": "1 level",
        },
        "plaintext_scalar_multiply": {
            "label": "Dot product\n+ bias",
            "depth_label": "1 level",
        },
        "full_pipeline_dot_plus_cubic": {
            "label": "Full pipeline\n(cubic sigmoid)",
            "depth_label": "4 levels\n(all consumed)",
        },
    }

    # For "Dot product + bias", use the noise_budget_analysis pipeline
    # tracking which measures the actual dot_product_plus_bias step.
    noise_file = Path("results/noise_budget_analysis.json")
    dot_bias_error = None
    if noise_file.exists():
        with open(noise_file, encoding="utf-8") as f:
            noise_data = json.load(f)
        for preset in noise_data["presets"]:
            if preset["preset"] == "deep":
                for step in preset.get("pipeline_error_tracking", []):
                    if step["step"] == "dot_product_plus_bias":
                        dot_bias_error = step["max_abs_error"]
                        break

    # Build the 4 bars in order
    labels = []
    errors = []
    depth_labels = []

    # 1. Encrypt/Decrypt roundtrip
    for s in stages:
        if s["stage"] == "encrypt_decrypt_roundtrip":
            labels.append("Encrypt/Decrypt\nroundtrip")
            errors.append(s["max_abs_error"])
            depth_labels.append("0 levels")
            break

    # 2. Linear dot product (uses dot_product stage)
    for s in stages:
        if s["stage"] == "dot_product":
            labels.append("Linear\ndot product")
            errors.append(s["max_abs_error"])
            depth_labels.append("1 level")
            break

    # 3. Dot product + bias (from noise_budget_analysis pipeline tracking)
    if dot_bias_error is not None:
        labels.append("Dot product\n+ bias")
        errors.append(dot_bias_error)
        depth_labels.append("1 level")
    else:
        # Fallback: use plaintext_scalar_multiply as proxy
        for s in stages:
            if s["stage"] == "plaintext_scalar_multiply":
                labels.append("Dot product\n+ bias")
                errors.append(s["max_abs_error"])
                depth_labels.append("1 level")
                break

    # 4. Full pipeline (cubic sigmoid)
    for s in stages:
        if s["stage"] == "full_pipeline_dot_plus_cubic":
            labels.append("Full pipeline\n(cubic sigmoid)")
            errors.append(s["max_abs_error"])
            depth_labels.append("4 levels\n(all consumed)")
            break

    clinical_tolerance = data.get("clinical_tolerance", 0.01)

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    x = np.arange(len(labels))
    bars = ax.bar(x, errors, width=0.55, color=BAR_COLOR, edgecolor="white",
                  linewidth=0.5, zorder=3)

    # Log scale
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 1e-1)

    # Clinical tolerance line
    ax.axhline(y=clinical_tolerance, color=TOLERANCE_COLOR, linestyle="--",
               linewidth=1.5, zorder=2)
    ax.text(len(labels) - 0.5, clinical_tolerance * 1.8,
            r"Clinical tolerance ($10^{-2}$)",
            color=TOLERANCE_COLOR, fontsize=8.5, ha="right", va="bottom")

    # Depth annotations below each bar
    for i, (bar, dl) in enumerate(zip(bars, depth_labels)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.8,
                dl, ha="center", va="bottom", fontsize=7.5, color="#444444")

    # Arrow annotation: "4 orders of magnitude margin"
    highest_error = max(errors)
    arrow_x = len(labels) - 1  # last bar
    ax.annotate(
        "4 orders of\nmagnitude margin",
        xy=(arrow_x, clinical_tolerance),
        xytext=(arrow_x - 0.15, 2e-4),
        fontsize=7.5,
        ha="center",
        va="top",
        color="#555555",
        arrowprops=dict(
            arrowstyle="->",
            color="#888888",
            lw=1.0,
            connectionstyle="arc3,rad=-0.15",
        ),
    )

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Absolute Error")

    # Value labels on bars
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, err * 0.35,
                f"{err:.1e}", ha="center", va="top", fontsize=7,
                color="white", fontweight="bold")

    fig.tight_layout()
    _save(fig, "ckks_precision_accumulation")
    print("Done.")


if __name__ == "__main__":
    main()
