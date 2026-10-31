#!/usr/bin/env python3
"""Run all systems-level FHE analyses and produce a friction report.

Executes:
  1. CKKS precision loss analysis
  2. RAM spike profiling
  3. Galois key size measurement
  4. Noise budget estimation

Saves JSON results and a narrative report to results/.

Usage:
    python -m experiments.run_systems_analysis
"""

from __future__ import annotations

import json
import sys
import time
from io import StringIO
from pathlib import Path

# Allow running as script or module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feasibility.systems_analysis import (
    analyze_galois_keys,
    analyze_memory_spikes,
    analyze_noise_budget,
    analyze_precision_loss,
)

RESULTS_DIR = PROJECT_ROOT / "results"


# -- helpers -------------------------------------------------------

def _fmt_err(v: float) -> str:
    """Format an error value for human readability."""
    if v < 1e-10:
        return f"{v:.2e}"
    if v < 0.001:
        return f"{v:.6f}"
    if v < 1.0:
        return f"{v:.4f}"
    return f"{v:.2f}"


def _fmt_mb(v: float) -> str:
    if v >= 1000:
        return f"{v/1024:.2f} GB"
    return f"{v:.1f} MB"


class _Tee:
    """Write to both a StringIO buffer and stdout."""

    def __init__(self):
        self.buf = StringIO()

    def print(self, *args, **kwargs):
        line = StringIO()
        print(*args, file=line, **kwargs)
        text = line.getvalue()
        # Handle Windows cp1252 encoding by replacing non-encodable chars
        try:
            sys.stdout.write(text)
        except UnicodeEncodeError:
            sys.stdout.write(text.encode(sys.stdout.encoding or "utf-8",
                                         errors="replace").decode(
                                             sys.stdout.encoding or "utf-8",
                                             errors="replace"))
        self.buf.write(text)

    def getvalue(self) -> str:
        return self.buf.getvalue()


# -- report generation ---------------------------------------------

def _report_precision(out: _Tee, precision: dict) -> None:
    out.print("\nCKKS Precision:")
    for preset in precision["presets"]:
        out.print(f"\n  Preset: {preset['preset']} "
                  f"(n={preset['poly_modulus_degree']}, "
                  f"depth={preset['multiplicative_depth']})")
        for stage in preset["stages"]:
            name = stage["stage"]
            if "error" in stage:
                out.print(f"    - {name}: FAILED -- {stage.get('note', stage['error'])}")
                continue
            err = stage["max_abs_error"]
            bits = stage["bits_of_precision"]
            status = "EXCEEDS tolerance" if stage.get("exceeds_clinical_tolerance") else "within tolerance"
            out.print(f"    - {name}: error = {_fmt_err(err)}, "
                      f"{bits:.1f} bits precision ({status})")

        exceeded = preset.get("tolerance_first_exceeded")
        if exceeded:
            out.print(f"    >> Clinical tolerance (0.01) first exceeded at: {exceeded}")
        else:
            out.print(f"    >> All stages within clinical tolerance (0.01)")


def _report_memory(out: _Tee, memory: dict) -> None:
    out.print("\nMemory Spikes:")
    for run in memory["runs"]:
        out.print(f"\n  Preset: {run['preset']} "
                  f"(n={run['poly_modulus_degree']})")
        out.print(f"    - Baseline RSS: {_fmt_mb(run['baseline_rss_mb'])}")
        out.print(f"    - Peak RSS:     {_fmt_mb(run['peak_rss_mb'])} "
                  f"({run['spike_factor']}x spike)")
        out.print(f"    - Peak phase:   {run['peak_phase']}")
        if run.get("phase_peaks"):
            out.print("    - Per-phase peaks:")
            for pp in run["phase_peaks"]:
                out.print(f"        {pp['phase']:30s}  "
                          f"{_fmt_mb(pp['peak_rss_mb']):>10s}  "
                          f"(+{pp['delta_from_baseline_mb']:.0f} MB)")


def _report_galois(out: _Tee, galois: dict) -> None:
    out.print("\nGalois Keys:")
    for cfg in galois["configurations"]:
        out.print(f"\n  Preset: {cfg['preset']} "
                  f"(n={cfg['poly_modulus_degree']})")
        out.print(f"    - Context without Galois keys: "
                  f"{_fmt_mb(cfg['context_without_galois_mb'])}")
        out.print(f"    - Context with Galois keys:    "
                  f"{_fmt_mb(cfg['context_with_galois_mb'])}")
        out.print(f"    - Galois keys alone:           "
                  f"{_fmt_mb(cfg['galois_keys_only_mb'])} "
                  f"({cfg['galois_pct_of_context']:.0f}% of context)")
        out.print(f"    - Rotation keys:               "
                  f"{cfg['n_power_of_2_rotation_keys']} "
                  f"(~{_fmt_mb(cfg['estimated_per_key_mb'])} each)")
        out.print(f"    - Ciphertext size:             "
                  f"{cfg['ciphertext_kb']:.1f} KB")

    if "insight" in galois:
        out.print(f"\n  Insight: {galois['insight']}")


def _report_noise(out: _Tee, noise: dict) -> None:
    out.print("\nNoise Budget:")
    for preset in noise["presets"]:
        out.print(f"\n  Preset: {preset['preset']} "
                  f"(n={preset['poly_modulus_degree']}, "
                  f"depth={preset['multiplicative_depth']})")

        out.print("    Squaring test (x -> x^2 -> x^4 -> ...):")
        for step in preset["squaring_test"]:
            if step["status"] == "FAILED":
                out.print(f"      depth {step['depth']}: FAILED "
                          f"({step.get('error_type', 'Error')})")
            else:
                bits = step.get("bits_of_precision", 0)
                err = step.get("abs_error", 0)
                out.print(f"      depth {step['depth']}: error = {_fmt_err(err)}, "
                          f"{bits:.1f} bits -- {step['status']}")
        out.print(f"    >> Noise cliff at depth: {preset['noise_cliff_at_depth']}")

        out.print("    Pipeline error tracking:")
        for step in preset["pipeline_error_tracking"]:
            if "error" in step:
                out.print(f"      {step['step']:25s} FAILED -- "
                          f"{step.get('note', step['error'])}")
            else:
                err = step["max_abs_error"]
                bits = step["bits_of_precision"]
                margin = step.get("levels_remaining")
                extra = f", {margin} levels remaining" if margin is not None else ""
                out.print(f"      {step['step']:25s} error = {_fmt_err(err)}, "
                          f"{bits:.1f} bits{extra}")


# -- main ----------------------------------------------------------

def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out = _Tee()
    total_start = time.time()

    out.print("=" * 60)
    out.print("  SYSTEMS FRICTION REPORT")
    out.print("  FHE-Agent-Privacy -- Section VI Data")
    out.print("=" * 60)

    # -- 1. Precision --
    out.print("\n[1/4] Running CKKS precision loss analysis...")
    t0 = time.time()
    precision = analyze_precision_loss()
    out.print(f"  Done in {time.time() - t0:.1f}s")

    path = RESULTS_DIR / "precision_analysis.json"
    path.write_text(json.dumps(precision, indent=2))
    out.print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")

    _report_precision(out, precision)

    # -- 2. Memory --
    out.print(f"\n{'-'*60}")
    out.print("\n[2/4] Running RAM spike analysis...")
    t0 = time.time()
    memory = analyze_memory_spikes()
    out.print(f"  Done in {time.time() - t0:.1f}s")

    path = RESULTS_DIR / "memory_timeline.json"
    path.write_text(json.dumps(memory, indent=2, default=str))
    out.print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")

    _report_memory(out, memory)

    # -- 3. Galois keys --
    out.print(f"\n{'-'*60}")
    out.print("\n[3/4] Running Galois key size analysis...")
    t0 = time.time()
    galois = analyze_galois_keys()
    out.print(f"  Done in {time.time() - t0:.1f}s")

    path = RESULTS_DIR / "galois_key_analysis.json"
    path.write_text(json.dumps(galois, indent=2))
    out.print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")

    _report_galois(out, galois)

    # -- 4. Noise budget --
    out.print(f"\n{'-'*60}")
    out.print("\n[4/4] Running noise budget estimation...")
    t0 = time.time()
    noise = analyze_noise_budget()
    out.print(f"  Done in {time.time() - t0:.1f}s")

    # Noise data folds into precision_analysis.json
    path = RESULTS_DIR / "noise_budget_analysis.json"
    path.write_text(json.dumps(noise, indent=2))
    out.print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")

    _report_noise(out, noise)

    # -- Summary --
    total_elapsed = time.time() - total_start
    out.print(f"\n{'='*60}")
    out.print(f"  REPORT COMPLETE  ({total_elapsed:.0f}s)")
    out.print(f"{'='*60}")

    # Save the full report text
    report_path = RESULTS_DIR / "systems_friction_report.txt"
    report_path.write_text(out.getvalue())
    print(f"\n  Full report saved: {report_path.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
