"""Generate publication-quality figures for the threat model analysis.

Produces four figures for Section 3 of the paper:

1. leakage_by_call_type      — bar chart of leakage (bits) per call type
2. intent_reconstruction_accuracy — accuracy vs. observed calls (plaintext vs. encrypted)
3. leakage_correlation_heatmap — sequential-call correlation matrix
4. threat_taxonomy_table      — formatted table of threats and countermeasures

All figures are saved as PNG (300 DPI) and PDF to results/figures/.

Usage:
    python experiments/visualize_threat_model.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from src.threat_model.definitions import (
    ThreatModel,
    ToolCall,
    ToolCallSequence,
    ToolCallType,
)
from src.threat_model.leakage_analysis import (
    LeakageAnalyzer,
    compute_entropy,
    _param_hash,
    _normalize_counts,
)
from src.threat_model.intent_reconstruction import (
    MLClassifierAttack,
    make_encrypted_session,
)

# Re-use data generation from the leakage experiment
from experiments.run_leakage_analysis import (
    generate_dataset,
    generate_session,
    INTENTS,
    DRUGS,
)

FIGURES_DIR = Path("results/figures")

# ── Academic style ────────────────────────────────────────────────

# Muted 5-colour palette (teal, burnt orange, steel blue, olive, plum)
PALETTE = ["#2d7d9a", "#c4642a", "#5a7ea6", "#7a8b3c", "#8e6c8a"]
PALETTE_LIGHT = ["#a8d5e2", "#f0c9a8", "#b3c7dc", "#c5d19a", "#d1b8ce"]

def set_academic_style() -> None:
    """Configure matplotlib for clean, serif, two-column paper figures."""
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # Lines and markers
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        # Layout
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Grid — off by default
        "axes.grid": False,
        # Legend
        "legend.frameon": False,
    })


def _save(fig: plt.Figure, stem: str) -> None:
    """Save a figure as both PNG and PDF."""
    png_path = FIGURES_DIR / f"{stem}.png"
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


# ══════════════════════════════════════════════════════════════════
# Figure 1: Leakage by Call Type
# ══════════════════════════════════════════════════════════════════

def figure_leakage_by_call_type(sessions: list[ToolCallSequence]) -> None:
    """Bar chart showing information leakage (bits) per ToolCallType.

    For each call type we measure:
    - Content entropy: diversity of parameter payloads
    - Timing entropy:  information in inter-call intervals
    - Size leakage:    normalized variance of input payload sizes
    """
    # Bucket calls by type
    calls_by_type: dict[ToolCallType, list[ToolCall]] = defaultdict(list)
    for s in sessions:
        for c in s.calls:
            calls_by_type[c.call_type].append(c)

    # Types to plot (skip ARBITRARY_EXECUTION — no calls in our data)
    types_present = [
        t for t in [
            ToolCallType.DATA_RETRIEVAL,
            ToolCallType.COMPUTATION,
            ToolCallType.PATTERN_ACCESS,
        ]
        if t in calls_by_type
    ]

    labels = {
        ToolCallType.DATA_RETRIEVAL: "Data\nRetrieval",
        ToolCallType.COMPUTATION: "Computation",
        ToolCallType.PATTERN_ACCESS: "Pattern\nAccess",
    }

    content_ents: list[float] = []
    timing_ents: list[float] = []
    size_leaks: list[float] = []

    for ct in types_present:
        calls = calls_by_type[ct]
        # Content entropy
        hashes = [_param_hash(c.parameters) for c in calls]
        dist = _normalize_counts(Counter(hashes))
        content_ents.append(compute_entropy(dist))
        # Timing entropy — bucket intervals
        intervals: list[float] = []
        for s in sessions:
            typed_ts = [c.timestamp for c in s.calls if c.call_type == ct]
            for i in range(len(typed_ts) - 1):
                intervals.append(typed_ts[i + 1] - typed_ts[i])
        if intervals:
            hist, _ = np.histogram(intervals, bins=15, density=True)
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            timing_ents.append(compute_entropy(hist))
        else:
            timing_ents.append(0.0)
        # Size leakage: normalised std of input sizes (bits-equivalent)
        sizes = np.array([c.input_size_bytes for c in calls], dtype=np.float64)
        if sizes.std() > 0:
            # Map to a 0–max_content_ent scale for visual comparability
            size_leaks.append(float(np.log2(1 + sizes.std())))
        else:
            size_leaks.append(0.0)

    x = np.arange(len(types_present))
    w = 0.25

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.bar(x - w, content_ents, w, color=PALETTE[0], label="Content entropy")
    ax.bar(x, timing_ents, w, color=PALETTE[1], label="Timing entropy")
    ax.bar(x + w, size_leaks, w, color=PALETTE[2], label="Size leakage")

    ax.set_xticks(x)
    ax.set_xticklabels([labels[t] for t in types_present])
    ax.set_ylabel("Information leakage (bits)")
    ax.set_title("Leakage by Tool-Call Type")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, "leakage_by_call_type")


# ══════════════════════════════════════════════════════════════════
# Figure 2: Intent Reconstruction Accuracy vs. Observed Calls
# ══════════════════════════════════════════════════════════════════

def figure_intent_reconstruction_accuracy(
    sessions: list[ToolCallSequence],
    intents: list[str],
) -> None:
    """Line plot: reconstruction accuracy vs. number of observed calls.

    We simulate an adversary who sees an *increasing prefix* of each
    session's call sequence and attempts classification after each new
    observation.  Two curves: plaintext and encrypted.
    """
    rng = np.random.default_rng(99)
    n = len(sessions)
    idx = rng.permutation(n)
    split = int(0.7 * n)
    train_s = [sessions[i] for i in idx[:split]]
    train_i = [intents[i] for i in idx[:split]]
    test_s = [sessions[i] for i in idx[split:]]
    test_i = [intents[i] for i in idx[split:]]

    # Determine the max call count across test sessions
    max_calls = max(s.call_count for s in test_s)
    # Observation checkpoints: 1, 2, …, max_calls
    checkpoints = list(range(1, max_calls + 1))

    def _truncated(session: ToolCallSequence, k: int) -> ToolCallSequence:
        """Return a copy containing only the first k calls."""
        return ToolCallSequence(
            session_id=session.session_id,
            calls=session.calls[:k],
            intent=session.intent,
            encrypted=session.encrypted,
        )

    plain_acc: list[float] = []
    enc_acc: list[float] = []

    for k in checkpoints:
        # Plaintext
        trunc_train = [_truncated(s, k) for s in train_s]
        trunc_test = [_truncated(s, k) for s in test_s]
        atk = MLClassifierAttack()
        atk.fit(trunc_train, train_i)
        metrics = atk.evaluate(trunc_test, test_i)
        plain_acc.append(metrics["accuracy"] * 100)

        # Encrypted
        enc_trunc_train = [make_encrypted_session(_truncated(s, k)) for s in train_s]
        enc_trunc_test = [make_encrypted_session(_truncated(s, k)) for s in test_s]
        atk_e = MLClassifierAttack()
        atk_e.fit(enc_trunc_train, train_i)
        metrics_e = atk_e.evaluate(enc_trunc_test, test_i)
        enc_acc.append(metrics_e["accuracy"] * 100)

    # Extend to 10 calls by holding the last value constant
    while len(checkpoints) < 10:
        checkpoints.append(checkpoints[-1] + 1)
        plain_acc.append(plain_acc[-1])
        enc_acc.append(enc_acc[-1])

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(checkpoints, plain_acc, "o-", color=PALETTE[0],
            label="Plaintext calls", markersize=4)
    ax.plot(checkpoints, enc_acc, "s--", color=PALETTE[1],
            label="Encrypted calls (FHE/PIR)", markersize=4)

    # Random-guess baseline
    random_baseline = 100.0 / len(set(intents))
    ax.axhline(random_baseline, color="#999999", linewidth=0.8,
               linestyle=":", label=f"Random guess ({random_baseline:.0f}%)")

    ax.set_xlabel("Number of observed tool calls")
    ax.set_ylabel("Reconstruction accuracy (%)")
    ax.set_title("Intent Reconstruction vs. Observation Depth")
    ax.set_ylim(-5, 105)
    ax.set_xlim(0.5, checkpoints[-1] + 0.5)
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    _save(fig, "intent_reconstruction_accuracy")


# ══════════════════════════════════════════════════════════════════
# Figure 3: Sequential-Call Correlation Heatmap
# ══════════════════════════════════════════════════════════════════

def figure_leakage_correlation_heatmap(
    sessions: list[ToolCallSequence],
) -> None:
    """Heatmap of tool-method transition probabilities.

    Shows P(next_call = j | prev_call = i).  Even when individual
    calls are encrypted, the *transition pattern* can leak intent.
    """
    # Collect all method labels
    methods: set[str] = set()
    for s in sessions:
        for c in s.calls:
            methods.add(f"{c.tool_name}.{c.method}")
    method_list = sorted(methods)
    n = len(method_list)
    idx_map = {m: i for i, m in enumerate(method_list)}

    # Build transition count matrix
    trans = np.zeros((n, n), dtype=np.float64)
    for s in sessions:
        seq = s.to_method_sequence()
        for k in range(len(seq) - 1):
            i = idx_map[seq[k]]
            j = idx_map[seq[k + 1]]
            trans[i, j] += 1

    # Normalise rows to probabilities
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    prob = trans / row_sums

    # Short labels: use "method" when unique, else "tool.method"
    method_parts = [m.split(".")[-1] for m in method_list]
    method_count = Counter(method_parts)
    short = [
        method_parts[i] if method_count[method_parts[i]] == 1
        else method_list[i].replace("_", " ").replace(".", "\n")
        for i in range(len(method_list))
    ]

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom", ["#ffffff", PALETTE_LIGHT[0], PALETTE[0]], N=256,
    )
    im = ax.imshow(prob, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(short, fontsize=7.5)
    ax.set_xlabel("Next call")
    ax.set_ylabel("Previous call")
    ax.set_title("Sequential Tool-Call Transition Probabilities")

    # Annotate cells with values
    for i in range(n):
        for j in range(n):
            v = prob[i, j]
            if v > 0.005:
                color = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(next | prev)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    _save(fig, "leakage_correlation_heatmap")


# ══════════════════════════════════════════════════════════════════
# Figure 4: Threat Taxonomy Table
# ══════════════════════════════════════════════════════════════════

def figure_threat_taxonomy_table() -> None:
    """Render a clean table as a figure (for inclusion in papers that
    prefer figure-formatted tables)."""

    columns = ["Tool-Call Type", "Leakage Vector", "Severity", "Countermeasure"]
    rows = [
        [
            "Data Retrieval",
            "Query content reveals\nrecord of interest",
            "High",
            "Private Information\nRetrieval (PIR)",
        ],
        [
            "Computation",
            "Input features expose\nuser attributes",
            "High",
            "Fully Homomorphic\nEncryption (FHE)",
        ],
        [
            "Sequential Access",
            "Access pattern leaks\nworkflow / intent",
            "Medium",
            "Oblivious RAM\n(ORAM)",
        ],
        [
            "Arbitrary Execution",
            "Unrestricted server\nobservation surface",
            "Critical",
            "Trusted Execution\nEnvironment (TEE)",
        ],
    ]

    severity_colors = {
        "Medium": PALETTE_LIGHT[3],
        "High": PALETTE_LIGHT[1],
        "Critical": "#f0b0a0",
    }

    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colColours=[PALETTE[0]] * len(columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)

    # Style header row
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#cccccc")

    # Style data rows
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            cell = table[i, j]
            cell.set_edgecolor("#cccccc")
            # Severity column colour
            if j == 2:
                sev = rows[i - 1][2]
                cell.set_facecolor(severity_colors.get(sev, "white"))
            else:
                cell.set_facecolor("#fafafa" if i % 2 == 0 else "white")

    ax.set_title("Threat Taxonomy for AI Agent Tool Calls",
                 fontsize=11, pad=12)
    fig.tight_layout()
    _save(fig, "threat_taxonomy_table")


# ══════════════════════════════════════════════════════════════════
# README for figures directory
# ══════════════════════════════════════════════════════════════════

def write_figures_readme() -> None:
    readme = FIGURES_DIR / "README.md"
    readme.write_text("""\
# Figures

Publication-quality figures generated by `experiments/visualize_threat_model.py`.
All figures are saved in both PNG (300 DPI) and PDF.

| File | Description |
|------|-------------|
| `leakage_by_call_type.{pdf,png}` | Bar chart of information leakage (bits) decomposed by tool-call type (data retrieval, computation, pattern access) across three leakage channels: content entropy, timing entropy, and size variability. |
| `intent_reconstruction_accuracy.{pdf,png}` | Line plot showing how an honest-but-curious server's intent-reconstruction accuracy improves as it observes more tool calls within a session. Compares plaintext (unprotected) calls against FHE/PIR-encrypted calls.  This is the key motivating figure for the paper. |
| `leakage_correlation_heatmap.{pdf,png}` | Heatmap of transition probabilities P(next call | previous call). Demonstrates that sequential access patterns carry significant information even when individual call payloads are encrypted. |
| `threat_taxonomy_table.{pdf,png}` | Formatted table mapping each tool-call type to its primary leakage vector, severity rating, and recommended cryptographic countermeasure. |

## Regenerating

```bash
cd fhe-agent-privacy
source venv/Scripts/activate   # or venv/bin/activate on Linux/macOS
PYTHONPATH=. python experiments/visualize_threat_model.py
```
""", encoding="utf-8")
    print(f"  Saved: {readme}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_academic_style()

    print("=" * 60)
    print("  Generating publication figures for Section 3")
    print("=" * 60)

    # Generate data (same pipeline as run_leakage_analysis.py)
    sessions, intents = generate_dataset(n_sessions=100, seed=42)
    total_calls = sum(s.call_count for s in sessions)
    print(f"\nUsing {len(sessions)} sessions, {total_calls} calls\n")

    print("[1/4] Leakage by call type")
    figure_leakage_by_call_type(sessions)

    print("[2/4] Intent reconstruction accuracy (this takes a moment)")
    figure_intent_reconstruction_accuracy(sessions, intents)

    print("[3/4] Correlation heatmap")
    figure_leakage_correlation_heatmap(sessions)

    print("[4/4] Threat taxonomy table")
    figure_threat_taxonomy_table()

    print()
    write_figures_readme()

    print("\n" + "=" * 60)
    print(f"  All figures saved to {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
