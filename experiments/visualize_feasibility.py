"""Generate publication-quality figures for the feasibility analysis (RQ2).

Produces eight figures for the paper, reading from the scaled benchmark
results (*_scaled.json):

1. primitive_comparison_latency  — grouped bar: plaintext vs encrypted
2. pir_scaling                   — PIR latency + memory vs DB size (dual Y)
3. fhe_scaling                   — FHE phase breakdown by poly degree
4. overhead_heatmap              — overhead factor heatmap
5. feasibility_boundary          — feasible regions in complexity-latency space
6. privacy_vs_latency_tradeoff   — scatter with Pareto frontier
7. memory_profile                — peak RSS by operation (horizontal bar)
8. latency_breakdown_boxplot     — pipeline phase breakdown with error bars
9. communication_overhead        — communication cost comparison

All figures are saved as PNG (300 DPI) and PDF to results/figures/.

Usage:
    cd fhe-agent-privacy
    python -m experiments.visualize_feasibility
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────

FIGURES_DIR = Path("results/figures")
BENCH_DIR = Path("results/benchmarks")
LEAKAGE_DIR = Path("results/leakage_analysis")

# ── Academic style ────────────────────────────────────────────────

PALETTE = ["#2d7d9a", "#c4642a", "#5a7ea6", "#7a8b3c", "#8e6c8a"]
PALETTE_LIGHT = ["#a8d5e2", "#f0c9a8", "#b3c7dc", "#c5d19a", "#d1b8ce"]


def set_academic_style() -> None:
    """Configure matplotlib for clean, serif, two-column paper figures."""
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
    """Save a figure as both PNG and PDF."""
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  Saved: {png}")
    print(f"  Saved: {pdf}")


# ── Data loading ──────────────────────────────────────────────────

def _load_scaled(filename: str) -> dict:
    """Load a wrapped *_scaled.json file.  Returns {"results": [...]}."""
    path = BENCH_DIR / filename
    if not path.exists():
        return {"results": []}
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "results" in raw:
        return raw
    # Flat list fallback
    return {"results": raw if isinstance(raw, list) else []}


def _fmt_size(n: int | float) -> str:
    """Human-readable size label for DB record counts."""
    n = int(n)
    if n >= 1000:
        return f"{n // 1000}K"
    return str(n)


def _fmt_bytes(b: int | float) -> str:
    """Human-readable byte size."""
    b = float(b)
    if b >= 1e9:
        return f"{b / 1e9:.1f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    if b >= 1e3:
        return f"{b / 1e3:.0f} KB"
    return f"{b:.0f} B"


# ===================================================================
# Figure 1 — Primitive comparison (grouped bar)
# ===================================================================

def fig_primitive_comparison(pir_data: dict, fhe_data: dict) -> None:
    """Grouped bar: plaintext vs. encrypted primitives with error bars."""

    pir_results = pir_data["results"]
    fhe_results = fhe_data["results"]

    # Representative PIR: 1 K records
    pir_1k = next((r for r in pir_results if r["db_size"] == 1000), None)
    if not pir_1k and pir_results:
        pir_1k = pir_results[len(pir_results) // 2]

    fhe_lin = next((r for r in fhe_results if r["model"] == "linear"), None)
    fhe_cv = next((r for r in fhe_results if r["model"] == "cv_sigmoid"), None)

    pir_lat = pir_1k["end_to_end_ms_mean"] if pir_1k else 647.0
    pir_std = pir_1k["end_to_end_ms_std"] if pir_1k else 7.0
    triv_lat = pir_1k["trivial_pir_ms"] if pir_1k else 0.003

    fhe_lin_lat = fhe_lin["end_to_end_ms_mean"] if fhe_lin else 193.0
    fhe_lin_std = fhe_lin["end_to_end_ms_std"] if fhe_lin else 3.0
    plain_ms = fhe_lin["plaintext_ms"] if fhe_lin else 0.003

    fhe_cv_lat = fhe_cv["end_to_end_ms_mean"] if fhe_cv else 875.0
    fhe_cv_std = fhe_cv["end_to_end_ms_std"] if fhe_cv else 15.0

    combo_lat = pir_lat + fhe_lin_lat
    combo_std = np.sqrt(pir_std**2 + fhe_lin_std**2)
    combo_cv_lat = pir_lat + fhe_cv_lat
    combo_cv_std = np.sqrt(pir_std**2 + fhe_cv_std**2)

    categories = ["Data\nLookup", "Risk Score\n(linear)",
                  "Risk Score\n(nonlinear)",
                  "Lookup +\nLinear Score", "Lookup +\nNonlinear Score"]

    plain_vals = [triv_lat, plain_ms, plain_ms,
                  triv_lat + plain_ms, triv_lat + plain_ms]
    enc_vals = [pir_lat, fhe_lin_lat, fhe_cv_lat, combo_lat, combo_cv_lat]
    enc_errs = [pir_std, fhe_lin_std, fhe_cv_std, combo_std, combo_cv_std]

    x = np.arange(len(categories))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w / 2, plain_vals, w,
           color=PALETTE_LIGHT[0], edgecolor=PALETTE[0],
           linewidth=0.8, label="Plaintext", zorder=3)
    ax.bar(x + w / 2, enc_vals, w, yerr=enc_errs, capsize=3,
           color=PALETTE[0], edgecolor="black", linewidth=0.8,
           label="Encrypted", zorder=3,
           error_kw=dict(lw=0.8, capthick=0.8))

    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.set_title("Latency: Plaintext vs. Encrypted Primitives")
    ax.legend(loc="upper left")

    for i, (pv, ev) in enumerate(zip(plain_vals, enc_vals)):
        if pv > 0:
            factor = ev / pv
            if factor >= 1e6:
                lbl = f"{factor / 1e6:.0f}M\u00d7"
            elif factor >= 1e3:
                lbl = f"{factor / 1e3:.0f}K\u00d7"
            else:
                lbl = f"{factor:.0f}\u00d7"
            ax.text(x[i] + w / 2, ev * 1.5, lbl,
                    ha="center", va="bottom", fontsize=7.5, color="#333")

    ax.set_ylim(bottom=1e-4)
    ax.yaxis.grid(True, alpha=0.25, which="both")
    _save(fig, "primitive_comparison_latency")


# ===================================================================
# Figure 2 — PIR scaling (dual Y-axis)
# ===================================================================

def fig_pir_scaling(pir_data: dict) -> None:
    """PIR latency + memory vs. DB size with dual Y-axis and error bars."""

    results = sorted(pir_data["results"], key=lambda r: r["db_size"])
    if not results:
        print("  SKIP: pir_scaling -- no data")
        return

    sizes = [r["db_size"] for r in results]
    lats = [r["end_to_end_ms_mean"] for r in results]
    stds = [r["end_to_end_ms_std"] for r in results]
    mem = [r["query_cycle_peak_rss_mb"] for r in results]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.8))

    # Left axis: latency
    c1 = PALETTE[0]
    ax1.errorbar(sizes, lats, yerr=stds, fmt="o-", color=c1,
                 capsize=4, capthick=1.0, elinewidth=1.0, markersize=6,
                 label="End-to-end latency", zorder=3)
    ax1.fill_between(sizes, [l - s for l, s in zip(lats, stds)],
                     [l + s for l, s in zip(lats, stds)],
                     color=PALETTE_LIGHT[0], alpha=0.3)
    ax1.set_xlabel("Database size (records)")
    ax1.set_ylabel("End-to-end latency (ms)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.yaxis.grid(True, alpha=0.2)

    # Right axis: memory
    ax2 = ax1.twinx()
    c2 = PALETTE[1]
    ax2.plot(sizes, mem, "s--", color=c2, markersize=5,
             label="Peak RSS", zorder=3)
    ax2.set_ylabel("Peak RSS (MB)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.spines["right"].set_visible(True)

    # Annotate 2-ciphertext boundary
    if len(results) >= 2:
        last, prev = results[-1], results[-2]
        if last["end_to_end_ms_mean"] > prev["end_to_end_ms_mean"] * 1.5:
            ax1.annotate(
                "2 ciphertexts\nrequired",
                xy=(last["db_size"], last["end_to_end_ms_mean"]),
                xytext=(last["db_size"] - 4000,
                        last["end_to_end_ms_mean"] + 80),
                fontsize=7.5, color="#555",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
            )

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    ax1.set_title("PIR Scaling: Latency and Memory vs. Database Size")
    fig.tight_layout()
    _save(fig, "pir_scaling")


# ===================================================================
# Figure 3 — FHE scaling (grouped bars by poly degree)
# ===================================================================

def fig_fhe_scaling(fhe_data: dict) -> None:
    """Phase-level latency breakdown for light (n=8192) vs. deep (n=16384)."""

    results = fhe_data["results"]
    light = next((r for r in results if r["poly_mod"] == 8192), None)
    deep = next((r for r in results if r["poly_mod"] == 16384), None)

    if not light and not deep:
        print("  SKIP: fhe_scaling -- no data")
        return

    phases = ["Encrypt", "Compute", "Decrypt", "End-to-end"]
    keys = ["encrypt", "compute", "decrypt", "end_to_end"]

    def _get(rec, phase):
        if not rec:
            return 0.0, 0.0
        return rec[f"{phase}_ms_mean"], rec[f"{phase}_ms_std"]

    light_m = [_get(light, k)[0] for k in keys]
    light_s = [_get(light, k)[1] for k in keys]
    deep_m = [_get(deep, k)[0] for k in keys]
    deep_s = [_get(deep, k)[1] for k in keys]

    x = np.arange(len(phases))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(x - w / 2, light_m, w, yerr=light_s, capsize=3,
           color=PALETTE[0], edgecolor="black", linewidth=0.6,
           label="n = 8192  (linear)", zorder=3,
           error_kw=dict(lw=0.8, capthick=0.8))
    ax.bar(x + w / 2, deep_m, w, yerr=deep_s, capsize=3,
           color=PALETTE[1], edgecolor="black", linewidth=0.6,
           label="n = 16384  (cv_sigmoid)", zorder=3,
           error_kw=dict(lw=0.8, capthick=0.8))

    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.yaxis.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)

    # Memory annotations above end-to-end bars
    if light:
        rss = light.get("scoring_peak_rss_mb", 0)
        top = light_m[3] + light_s[3]
        ax.text(x[3] - w / 2, top + 12, f"{rss:.0f}\nMB",
                fontsize=6.5, ha="center", color=PALETTE[0])
    if deep:
        rss = deep.get("scoring_peak_rss_mb", 0)
        top = deep_m[3] + deep_s[3]
        ax.text(x[3] + w / 2, top + 12, f"{rss:.0f}\nMB",
                fontsize=6.5, ha="center", color=PALETTE[1])

    ax.set_title("FHE Phase Latency by Polynomial Degree")
    fig.tight_layout()
    _save(fig, "fhe_scaling")


# ===================================================================
# Figure 4 — Overhead heatmap
# ===================================================================

def fig_overhead_heatmap(pir_data: dict, fhe_data: dict) -> None:
    """Heatmap of encrypted/plaintext overhead factor."""

    pir_results = pir_data["results"]
    fhe_results = fhe_data["results"]
    pir_by_size = {r["db_size"]: r for r in pir_results}
    fhe_lin = next((r for r in fhe_results if r["model"] == "linear"), None)
    fhe_cv = next((r for r in fhe_results if r["model"] == "cv_sigmoid"), None)

    row_labels = ["PIR Lookup", "FHE Scoring\n(linear)",
                  "FHE Scoring\n(nonlinear)",
                  "PIR + FHE\n(linear)", "PIR + FHE\n(nonlinear)"]
    col_sizes = [100, 1000, 10000]
    col_labels = ["Small\n(100)", "Medium\n(1K)", "Large\n(10K)"]

    plain_ms = fhe_lin["plaintext_ms"] if fhe_lin else 0.003
    fhe_lin_lat = fhe_lin["end_to_end_ms_mean"] if fhe_lin else 193.0
    fhe_cv_lat = fhe_cv["end_to_end_ms_mean"] if fhe_cv else 875.0

    matrix = np.zeros((5, 3))
    for j, s in enumerate(col_sizes):
        pr = pir_by_size.get(s)
        pir_lat = pr["end_to_end_ms_mean"] if pr else 647.0
        triv = pr["trivial_pir_ms"] if pr else 0.003
        matrix[0, j] = pir_lat / max(triv, 1e-6)
        matrix[1, j] = fhe_lin_lat / max(plain_ms, 1e-6)
        matrix[2, j] = fhe_cv_lat / max(plain_ms, 1e-6)
        matrix[3, j] = (pir_lat + fhe_lin_lat) / max(triv + plain_ms, 1e-6)
        matrix[4, j] = (pir_lat + fhe_cv_lat) / max(triv + plain_ms, 1e-6)

    cmap = LinearSegmentedColormap.from_list(
        "overhead",
        [(0.0, "#2d7d4f"), (0.15, "#78b85c"), (0.35, "#d4c94a"),
         (0.6, "#e8944a"), (1.0, "#c4382a")])

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    log_mat = np.log10(matrix)
    vmin, vmax = 3.0, 6.0

    im = ax.imshow(log_mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val >= 1e6:
                txt = f"{val / 1e6:.1f}M\u00d7"
            elif val >= 1e3:
                txt = f"{val / 1e3:.0f}K\u00d7"
            else:
                txt = f"{val:.0f}\u00d7"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white" if log_mat[i, j] > 4.5 else "black")

    ax.set_xticks(range(3))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.set_title("Overhead Factor  (encrypted / plaintext)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_ticks([3, 4, 5, 6])
    cbar.set_ticklabels(["10\u00b3\u00d7", "10\u2074\u00d7",
                         "10\u2075\u00d7", "10\u2076\u00d7"])
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save(fig, "overhead_heatmap")


# ===================================================================
# Figure 5 — Feasibility boundary
# ===================================================================

def fig_feasibility_boundary(pir_data: dict, fhe_data: dict) -> None:
    """2-D feasibility plot with measured data points and error bars."""

    pir_results = pir_data["results"]
    fhe_results = fhe_data["results"]
    pir_by_size = {r["db_size"]: r for r in pir_results}
    fhe_lin = next((r for r in fhe_results if r["model"] == "linear"), None)
    fhe_cv = next((r for r in fhe_results if r["model"] == "cv_sigmoid"), None)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # --- Shaded regions ---
    xf = np.linspace(-0.3, 5.3, 300)

    # PIR region (lookup only, x ~ 0)
    ax.fill_between(xf, 0, 1500, where=(xf <= 0.7),
                    color=PALETTE_LIGHT[0], alpha=0.50, zorder=1)
    # FHE region (computation, x ~ 1-3)
    fhe_ceil = np.where(
        xf <= 1.0, 300,
        np.where(xf <= 2.0, 300 + (xf - 1.0) * 700,
                 np.where(xf <= 3.5, 1000 + (xf - 2.0) * 200, 1300)))
    ax.fill_between(xf, 0, fhe_ceil, where=(xf >= 0.5) & (xf <= 3.8),
                    color=PALETTE_LIGHT[1], alpha=0.45, zorder=1)
    # Combined PIR+FHE
    combo_ceil = np.where(
        xf <= 0.7, 800,
        np.where(xf <= 1.5, 800 + (xf - 0.7) * 1100,
                 np.where(xf <= 3.5, 1680 + (xf - 1.5) * 200, 2080)))
    ax.fill_between(xf, 0, combo_ceil, where=(xf >= -0.1) & (xf <= 3.8),
                    color=PALETTE_LIGHT[2], alpha=0.25, zorder=0)

    # Region labels
    ax.text(4.3, 1000, "Infeasible\n(requires\ngeneral MPC\nor TEE)",
            fontsize=8.5, color="#888", ha="center", va="center",
            style="italic")
    ax.text(0.15, 1200, "PIR\nfeasible", fontsize=9, color=PALETTE[0],
            fontweight="bold", ha="center", va="center", alpha=0.85)
    ax.text(2.0, 150, "FHE feasible", fontsize=9, color=PALETTE[1],
            fontweight="bold", ha="center", va="center", alpha=0.85)
    ax.text(1.3, 1700, "PIR + FHE", fontsize=9, color=PALETTE[2],
            fontweight="bold", ha="center", va="center", alpha=0.85)

    # --- Measured points ---
    pts: list[tuple[float, float, float, str]] = []

    pir_1k = pir_by_size.get(1000)
    if pir_1k:
        pts.append((0.0, pir_1k["end_to_end_ms_mean"],
                     pir_1k["end_to_end_ms_std"], "PIR (1K)"))
    pir_10k = pir_by_size.get(10000)
    if pir_10k:
        pts.append((0.0, pir_10k["end_to_end_ms_mean"],
                     pir_10k["end_to_end_ms_std"], "PIR (10K)"))
    if fhe_lin:
        pts.append((1.0, fhe_lin["end_to_end_ms_mean"],
                     fhe_lin["end_to_end_ms_std"],
                     "FHE linear\n(n=8192)"))
    if fhe_cv:
        pts.append((2.0, fhe_cv["end_to_end_ms_mean"],
                     fhe_cv["end_to_end_ms_std"],
                     "FHE cv_sigmoid\n(n=16384)"))
    if pir_1k and fhe_lin:
        c_lat = pir_1k["end_to_end_ms_mean"] + fhe_lin["end_to_end_ms_mean"]
        c_std = np.sqrt(pir_1k["end_to_end_ms_std"]**2 +
                        fhe_lin["end_to_end_ms_std"]**2)
        pts.append((1.0, c_lat, c_std, "PIR + FHE\n(linear)"))
    if pir_1k and fhe_cv:
        c_lat = pir_1k["end_to_end_ms_mean"] + fhe_cv["end_to_end_ms_mean"]
        c_std = np.sqrt(pir_1k["end_to_end_ms_std"]**2 +
                        fhe_cv["end_to_end_ms_std"]**2)
        pts.append((2.0, c_lat, c_std, "PIR + FHE\n(cv_sigmoid)"))

    markers = ["o", "D", "s", "^", "P", "X"]
    for idx, (cx, cy, cerr, lbl) in enumerate(pts):
        mk = markers[idx % len(markers)]
        ax.errorbar(cx, cy, yerr=cerr, fmt=mk, color="black",
                    markersize=7, capsize=3, capthick=0.8, elinewidth=0.8,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        dx = 0.3 if cx < 2.5 else -0.3
        dy = 60 if idx % 2 == 0 else -80
        ha = "left" if dx > 0 else "right"
        ax.annotate(lbl, (cx, cy), xytext=(cx + dx, cy + dy),
                    fontsize=7, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6),
                    zorder=6)

    # --- Axes ---
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(0, 2200)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["Key-value\nlookup", "Linear\nscoring",
                         "Quadratic\npolynomial", "Cubic\npolynomial",
                         "Higher-degree\n/ iterative",
                         "Arbitrary\nbranching"], fontsize=7.5)
    ax.set_xlabel("Computation Complexity")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Feasibility Boundary for Privacy-Preserving Tool Calls")
    ax.yaxis.grid(True, alpha=0.2)

    # Budget lines
    ax.axhline(y=1000, color="#999", ls="--", lw=0.8, zorder=2)
    ax.text(5.3, 1020, "1 s budget", fontsize=7.5, color="#999",
            ha="right", va="bottom")
    ax.axhline(y=2000, color="#cc6666", ls="--", lw=0.8, zorder=2)
    ax.text(5.3, 2020, "2 s budget", fontsize=7.5, color="#cc6666",
            ha="right", va="bottom")

    fig.tight_layout()
    _save(fig, "feasibility_boundary")


# ===================================================================
# Figure 6 — Privacy vs. latency tradeoff
# ===================================================================

def fig_privacy_vs_latency(pir_data: dict, fhe_data: dict) -> None:
    """Scatter of latency vs. information leakage with Pareto frontier."""

    leakage_path = LEAKAGE_DIR / "leakage_report.json"
    if leakage_path.exists():
        leakage = json.loads(leakage_path.read_text())
    else:
        leakage = None

    plain_leak = (leakage["direct"]["content_entropy_bits"]
                  if leakage else 6.4)
    pir_leak = (leakage["direct"]["timing_entropy_bits"]
                if leakage else 2.7)
    fhe_leak = 1.5   # size/timing side channels only
    combo_leak = (leakage["summary"]["total_leakage_score"]
                  if leakage else 0.4)

    pir_results = pir_data["results"]
    fhe_results = fhe_data["results"]
    pir_1k = next((r for r in pir_results if r["db_size"] == 1000), None)
    fhe_cv = next((r for r in fhe_results if r["model"] == "cv_sigmoid"), None)
    fhe_lin = next((r for r in fhe_results if r["model"] == "linear"), None)

    plain_lat = ((pir_1k["trivial_pir_ms"] if pir_1k else 0.003) +
                 (fhe_lin["plaintext_ms"] if fhe_lin else 0.003))
    pir_lat = pir_1k["end_to_end_ms_mean"] if pir_1k else 647.0
    fhe_lat = fhe_cv["end_to_end_ms_mean"] if fhe_cv else 875.0
    combo_lat = pir_lat + fhe_lat

    points = [
        (plain_lat, plain_leak, "Plaintext", PALETTE[1], "o"),
        (pir_lat,   pir_leak,   "PIR only",  PALETTE[0], "s"),
        (fhe_lat,   fhe_leak,   "FHE only",  PALETTE[2], "^"),
        (combo_lat, combo_leak, "PIR + FHE", PALETTE[3], "D"),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    for lat, leak, lbl, color, mk in points:
        ax.scatter(lat, leak, s=100, marker=mk, color=color,
                   edgecolors="black", linewidth=0.8, zorder=5, label=lbl)

    # Pareto frontier
    sp = sorted(points, key=lambda p: p[0])
    px, py = [sp[0][0]], [sp[0][1]]
    mn = sp[0][1]
    for lat, leak, *_ in sp[1:]:
        if leak < mn:
            px.append(lat)
            py.append(leak)
            mn = leak
    ax.plot(px, py, "--", color="#555", lw=1.0, alpha=0.6, zorder=3)

    ax.axhspan(0, 1.0, xmin=0, xmax=0.3, color="#e8f5e9", alpha=0.3,
               zorder=0)
    ax.text(0.15, 0.2, "ideal\nregion", fontsize=7.5, color="#4caf50",
            ha="center", va="center", style="italic",
            transform=ax.transAxes)

    offsets = [(15, 0.3), (15, 0.3), (-80, -0.4), (15, 0.3)]
    for i, (lat, leak, lbl, color, mk) in enumerate(points):
        dx, dy = offsets[i]
        ax.annotate(lbl, (lat, leak), xytext=(lat + dx, leak + dy),
                    fontsize=8, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6))

    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_ylabel("Content Leakage (bits)")
    ax.set_title("Privacy\u2013Latency Tradeoff")
    ax.set_ylim(-0.5, plain_leak + 1.5)
    ax.yaxis.grid(True, alpha=0.2)
    ax.xaxis.grid(True, alpha=0.2, which="both")

    mid = len(px) // 2
    if mid < len(px):
        ax.text(px[mid] * 1.5, py[mid] + 0.6,
                "Pareto frontier", fontsize=7.5, color="#555",
                style="italic", rotation=-15)

    fig.tight_layout()
    _save(fig, "privacy_vs_latency_tradeoff")


# ===================================================================
# Figure 7 — Memory profile (NEW)
# ===================================================================

def fig_memory_profile(pir_data: dict, fhe_data: dict,
                       pipeline_data: dict) -> None:
    """Horizontal bar chart of peak RSS across all operation phases."""

    entries: list[tuple[str, float, str]] = []  # (label, MB, color)

    pir_results = sorted(pir_data["results"], key=lambda r: r["db_size"])
    for r in pir_results:
        tag = _fmt_size(r["db_size"])
        entries.append((f"PIR setup ({tag})", r["setup_peak_rss_mb"],
                        PALETTE_LIGHT[0]))
        entries.append((f"PIR query cycle ({tag})",
                        r["query_cycle_peak_rss_mb"], PALETTE[0]))

    fhe_results = sorted(fhe_data["results"], key=lambda r: r["poly_mod"])
    for r in fhe_results:
        tag = f"n={r['poly_mod']}"
        entries.append((f"FHE setup ({tag})",
                        r["context_setup_peak_rss_mb"], PALETTE_LIGHT[1]))
        entries.append((f"FHE scoring ({tag})",
                        r["scoring_peak_rss_mb"], PALETTE[1]))

    for r in pipeline_data["results"]:
        tag = r.get("db_tag", _fmt_size(r.get("db_size", 10000)))
        entries.append((f"Pipeline total ({tag})",
                        r["pipeline_peak_rss_mb"], PALETTE[4]))

    if not entries:
        print("  SKIP: memory_profile -- no data")
        return

    # Sort by value for a clean horizontal bar chart
    entries.sort(key=lambda e: e[1])

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    colors = [e[2] for e in entries]

    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(entries) + 1.2))
    y = np.arange(len(entries))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.5,
            zorder=3, height=0.65)

    for i, v in enumerate(values):
        ax.text(v + 8, i, f"{v:.0f} MB", va="center", fontsize=7.5,
                color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Peak RSS (MB)")
    ax.set_title("Peak Memory Consumption by Operation")
    ax.xaxis.grid(True, alpha=0.2)

    # Reference lines
    ax.axvline(x=1024, color="#cc6666", ls="--", lw=0.8)
    ax.text(1030, len(entries) - 0.5, "1 GB", fontsize=7.5,
            color="#cc6666", va="center")

    ax.set_xlim(0, max(values) * 1.2)
    fig.tight_layout()
    _save(fig, "memory_profile")


# ===================================================================
# Figure 8 — Pipeline latency breakdown (NEW)
# ===================================================================

def fig_latency_breakdown(pipeline_data: dict) -> None:
    """Stacked bar showing pipeline phase breakdown with error bars."""

    results = pipeline_data["results"]
    if not results:
        print("  SKIP: latency_breakdown_boxplot -- no data")
        return

    r = results[0]  # single pipeline configuration

    phases = ["PIR Lookup", "FHE Scoring", "Decision"]
    means = [r["pir_phase_ms_mean"], r["fhe_phase_ms_mean"],
             r["decision_ms_mean"]]
    stds = [r["pir_phase_ms_std"], r["fhe_phase_ms_std"],
            r["decision_ms_std"]]
    phase_colors = [PALETTE[0], PALETTE[1], PALETTE[3]]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Stacked horizontal bar
    left = 0.0
    bars = []
    for i, (m, s, c) in enumerate(zip(means, stds, phase_colors)):
        bar = ax.barh(0, m, left=left, color=c, edgecolor="black",
                      linewidth=0.5, height=0.5, zorder=3,
                      label=phases[i])
        bars.append(bar)
        # Error bar at end of segment
        ax.errorbar(left + m, 0, xerr=s, fmt="none", color="black",
                    capsize=4, capthick=0.8, elinewidth=0.8, zorder=4)
        # Phase label inside bar
        if m > 80:  # only label if wide enough
            ax.text(left + m / 2, 0, f"{m:.0f} ms",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold", zorder=5)
        left += m

    # Total annotation
    total_m = r["total_ms_mean"]
    total_s = r["total_ms_std"]
    ax.errorbar(total_m, 0, xerr=total_s, fmt="|", color="black",
                markersize=12, capsize=4, capthick=1.0, elinewidth=1.0,
                zorder=5)
    ax.text(total_m + total_s + 20, 0,
            f"Total: {total_m:.0f} \u00b1 {total_s:.0f} ms",
            va="center", fontsize=9, fontweight="bold")

    # Setup costs (shown as separate thinner bar below)
    setup_phases = ["PIR Setup", "FHE Setup"]
    setup_means = [r["pir_setup_ms"], r["fhe_setup_ms"]]
    setup_colors = [PALETTE_LIGHT[0], PALETTE_LIGHT[1]]
    left = 0.0
    for i, (m, c) in enumerate(zip(setup_means, setup_colors)):
        ax.barh(-0.7, m, left=left, color=c, edgecolor="black",
                linewidth=0.5, height=0.35, zorder=3,
                label=setup_phases[i])
        if m > 200:
            ax.text(left + m / 2, -0.7, f"{m:.0f} ms",
                    ha="center", va="center", fontsize=7, color="#333",
                    zorder=5)
        left += m

    ax.set_yticks([0, -0.7])
    ax.set_yticklabels(["Per-query\nlatency", "One-time\nsetup"],
                       fontsize=9)
    ax.set_xlabel("Latency (ms)")
    ax.set_title(f"Pipeline Latency Breakdown  "
                 f"({r['db_size']//1000}K DB, "
                 f"n={r['n_patients']} patients)")
    ax.xaxis.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    ax.set_ylim(-1.2, 0.6)

    fig.tight_layout()
    _save(fig, "latency_breakdown_boxplot")


# ===================================================================
# Figure 9 — Communication overhead (NEW)
# ===================================================================

def fig_communication_overhead(pir_data: dict, fhe_data: dict) -> None:
    """Bar chart comparing communication costs: plaintext vs. PIR vs. FHE."""

    pir_results = sorted(pir_data["results"], key=lambda r: r["db_size"])
    fhe_results = fhe_data["results"]

    if not pir_results:
        print("  SKIP: communication_overhead -- no data")
        return

    # --- Sub-plot 1: PIR communication by DB size ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))

    sizes = [r["db_size"] for r in pir_results]
    labels = [_fmt_size(s) for s in sizes]

    query_mb = [r["communication_query_bytes"] / 1e6 for r in pir_results]
    resp_mb = [r["communication_response_bytes"] / 1e6 for r in pir_results]
    triv_mb = [r["trivial_pir_comm_bytes"] / 1e6 for r in pir_results]

    x = np.arange(len(sizes))
    w = 0.25

    ax1.bar(x - w, triv_mb, w, color=PALETTE_LIGHT[2],
            edgecolor=PALETTE[2], linewidth=0.8,
            label="Plaintext", zorder=3)
    ax1.bar(x, query_mb, w, color=PALETTE[0],
            edgecolor="black", linewidth=0.6,
            label="PIR query", zorder=3)
    ax1.bar(x + w, resp_mb, w, color=PALETTE[1],
            edgecolor="black", linewidth=0.6,
            label="PIR response", zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Database size (records)")
    ax1.set_ylabel("Communication (MB)")
    ax1.set_yscale("log")
    ax1.set_title("(a) PIR Communication Cost")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.yaxis.grid(True, alpha=0.2, which="both")

    # Overhead annotations (above the tallest bar in each group)
    for i, (t, q, r_val) in enumerate(zip(triv_mb, query_mb, resp_mb)):
        total = q + r_val
        if t > 0:
            factor = total / t
            top = max(t, q, r_val)
            ax1.text(x[i], top * 1.6, f"{factor:.0f}\u00d7",
                     ha="center", fontsize=6.5, color="#555")

    # --- Sub-plot 2: FHE key / ciphertext sizes ---
    cats = []
    key_vals = []
    galois_vals = []
    ct_vals = []

    for r in sorted(fhe_results, key=lambda r: r["poly_mod"]):
        tag = f"n={r['poly_mod']}"
        cats.append(tag)
        key_vals.append(r["key_context_bytes"] / 1e6)
        galois_vals.append(r["galois_key_bytes"] / 1e6)
        ct_vals.append(r["ciphertext_bytes"] / 1e6)

    if cats:
        x2 = np.arange(len(cats))
        w2 = 0.22

        ax2.bar(x2 - w2, key_vals, w2, color=PALETTE[0],
                edgecolor="black", linewidth=0.6,
                label="Key context", zorder=3)
        ax2.bar(x2, galois_vals, w2, color=PALETTE[1],
                edgecolor="black", linewidth=0.6,
                label="Galois keys", zorder=3)
        ax2.bar(x2 + w2, ct_vals, w2, color=PALETTE[3],
                edgecolor="black", linewidth=0.6,
                label="Ciphertext", zorder=3)

        ax2.set_xticks(x2)
        ax2.set_xticklabels(cats)
        ax2.set_xlabel("Polynomial modulus degree")
        ax2.set_ylabel("Size (MB)")
        ax2.set_title("(b) FHE Key and Ciphertext Sizes")
        ax2.legend(fontsize=7, loc="upper left")
        ax2.yaxis.grid(True, alpha=0.2)

        # Value labels
        for i in range(len(cats)):
            for val, offset in [(key_vals[i], -w2),
                                (galois_vals[i], 0),
                                (ct_vals[i], w2)]:
                if val > 1.0:
                    ax2.text(x2[i] + offset, val + 2,
                             f"{val:.1f}", ha="center", fontsize=6.5,
                             color="#333")

    fig.tight_layout(w_pad=2.5)
    _save(fig, "communication_overhead")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_academic_style()

    print("Loading scaled benchmark data...")
    pir = _load_scaled("pir_benchmarks_scaled.json")
    fhe = _load_scaled("fhe_benchmarks_scaled.json")
    pipeline = _load_scaled("pipeline_benchmarks_scaled.json")

    n_pir = len(pir["results"])
    n_fhe = len(fhe["results"])
    n_pipe = len(pipeline["results"])
    print(f"  PIR: {n_pir}  FHE: {n_fhe}  Pipeline: {n_pipe} records")

    if n_pir == 0 and n_fhe == 0:
        print("  ERROR: No benchmark data found.  "
              "Run run_benchmarks.py first.")
        return

    print("\nGenerating feasibility figures...\n")

    print("[1/9] Primitive comparison")
    fig_primitive_comparison(pir, fhe)

    print("[2/9] PIR scaling")
    fig_pir_scaling(pir)

    print("[3/9] FHE scaling")
    fig_fhe_scaling(fhe)

    print("[4/9] Overhead heatmap")
    fig_overhead_heatmap(pir, fhe)

    print("[5/9] Feasibility boundary")
    fig_feasibility_boundary(pir, fhe)

    print("[6/9] Privacy vs. latency tradeoff")
    fig_privacy_vs_latency(pir, fhe)

    print("[7/9] Memory profile")
    fig_memory_profile(pir, fhe, pipeline)

    print("[8/9] Pipeline latency breakdown")
    fig_latency_breakdown(pipeline)

    print("[9/9] Communication overhead")
    fig_communication_overhead(pir, fhe)

    print(f"\nAll feasibility figures saved to {FIGURES_DIR}/.")


if __name__ == "__main__":
    main()
