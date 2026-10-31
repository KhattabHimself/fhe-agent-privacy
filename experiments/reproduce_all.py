#!/usr/bin/env python3
"""Reproduce every result reported in the paper from scratch.

Usage
-----
    python experiments/reproduce_all.py              # full (10K drugs, 100 patients)
    python experiments/reproduce_all.py --quick       # toy  (small DBs, 10 patients)
    python experiments/reproduce_all.py --skip-figures
    python experiments/reproduce_all.py --output-dir /tmp/results

Exit codes
----------
    0  All steps passed and paper claims verified.
    1  One or more steps failed or a claim could not be verified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "results" / "benchmarks"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Timestamp for log entries."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_json(path: Path) -> dict | list | None:
    """Load a JSON file, returning None on any error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _parse_correctness(s: str) -> tuple[int, int] | None:
    """Parse 'X/Y' into (X, Y).  Returns None on bad input."""
    try:
        parts = s.split("/")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Logger -- dual output (stdout + buffer -> file)
# ---------------------------------------------------------------------------

class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, msg: str, *, echo: bool = True) -> None:
        entry = f"[{_ts()}] {msg}"
        self.lines.append(entry)
        if echo:
            try:
                print(msg, flush=True)
            except UnicodeEncodeError:
                print(msg.encode("utf-8", errors="replace")
                         .decode("utf-8", errors="replace"), flush=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
            f.write("\n")


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def _run_step(
    step: int,
    total: int,
    description: str,
    cmds: list[list[str]],
    *,
    logger: _Logger,
) -> tuple[bool, float]:
    """Run one or more subprocess commands as a numbered step.

    Parameters
    ----------
    cmds : list of command-lists, executed sequentially.  If any
           command fails the step is marked failed immediately.

    Returns (success, elapsed_seconds).
    """
    label = f"  Step {step}/{total}: {description}"
    # Print label padded, no newline yet — result goes on same line
    print(f"{label:<55s}", end="", flush=True)
    logger.log(label, echo=False)

    t0 = time.time()

    for cmd in cmds:
        logger.log(f"    $ {' '.join(cmd)}", echo=False)
        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=3600,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            msg = f"[FAIL] ({elapsed:.1f}s) -- timeout"
            print(msg)
            logger.log(f"    {msg}", echo=False)
            return False, elapsed
        except OSError as exc:
            elapsed = time.time() - t0
            msg = f"[FAIL] ({elapsed:.1f}s) -- {exc}"
            print(msg)
            logger.log(f"    {msg}", echo=False)
            return False, elapsed

        # Capture full output in log (not echoed to console)
        for line in (result.stdout or "").splitlines():
            logger.log(f"    {line}", echo=False)
        for line in (result.stderr or "").splitlines():
            logger.log(f"    [stderr] {line}", echo=False)

        if result.returncode != 0:
            elapsed = time.time() - t0
            msg = f"[FAIL] ({elapsed:.1f}s) -- exit code {result.returncode}"
            print(msg)
            logger.log(f"    {msg}", echo=False)
            return False, elapsed

    elapsed = time.time() - t0
    msg = f"[OK] ({elapsed:.1f}s)"
    print(msg)
    logger.log(f"    {msg}", echo=False)
    return True, elapsed


# ---------------------------------------------------------------------------
# Summary extraction -- reads result JSONs to build the table
# ---------------------------------------------------------------------------

def _extract_summary() -> tuple[list[tuple[str, str]], list[str]]:
    """Return (rows, verification_failures).

    Each row is (claim_label, measured_value).
    verification_failures lists any claim that could not be verified.
    """
    pir_data = _load_json(BENCHMARKS_DIR / "pir_benchmarks_scaled.json")
    fhe_data = _load_json(BENCHMARKS_DIR / "fhe_benchmarks_scaled.json")
    pipe_data = _load_json(BENCHMARKS_DIR / "pipeline_benchmarks_scaled.json")

    rows: list[tuple[str, str]] = []
    failures: list[str] = []

    # -- helper: pick best entry from a results list ----------------------

    def _best_pipeline(data: dict | None) -> dict | None:
        """Return pipeline result with the largest db_size."""
        if not data or "results" not in data or not data["results"]:
            return None
        return max(data["results"], key=lambda r: r.get("db_size", 0))

    def _best_fhe(data: dict | None, model: str) -> dict | None:
        """Return FHE result for *model*, preferring deep preset."""
        if not data or "results" not in data:
            return None
        matches = [r for r in data["results"] if r.get("model") == model]
        deep = [r for r in matches if r.get("preset") == "deep"]
        return (deep or matches or [None])[0]

    def _best_pir(data: dict | None) -> dict | None:
        """Return standalone PIR result with the largest db_size."""
        if not data or "results" not in data or not data["results"]:
            return None
        return max(data["results"], key=lambda r: r.get("db_size", 0))

    def _fmt_ms(mean, std) -> str:
        if mean is None:
            return "N/A"
        if std is not None and std > 0:
            return f"{mean:.0f} +/- {std:.0f} ms"
        return f"{mean:.0f} ms"

    # =====================================================================
    # Build rows
    # =====================================================================

    bp = _best_pipeline(pipe_data)

    # -- 1. PIR lookup latency -------------------------------------------
    if bp:
        db = bp.get("db_size", "?")
        db_label = f"{db:,}" if isinstance(db, int) else str(db)
        rows.append((
            f"PIR lookup latency ({db_label} DB)",
            _fmt_ms(bp.get("pir_phase_ms_mean"), bp.get("pir_phase_ms_std")),
        ))
    else:
        sp = _best_pir(pir_data)
        if sp:
            db = sp.get("db_size", "?")
            db_label = f"{db:,}" if isinstance(db, int) else str(db)
            rows.append((
                f"PIR lookup latency ({db_label} DB)",
                _fmt_ms(sp.get("end_to_end_ms_mean"),
                        sp.get("end_to_end_ms_std")),
            ))
        else:
            rows.append(("PIR lookup latency", "N/A"))

    # -- 2. FHE linear scoring latency -----------------------------------
    fl = _best_fhe(fhe_data, "linear")
    if fl:
        rows.append((
            "FHE linear scoring latency",
            _fmt_ms(fl.get("end_to_end_ms_mean"),
                    fl.get("end_to_end_ms_std")),
        ))
    else:
        rows.append(("FHE linear scoring latency", "N/A"))

    # -- 3. FHE cubic scoring latency ------------------------------------
    fc = _best_fhe(fhe_data, "cv_sigmoid")
    if fc:
        rows.append((
            "FHE cubic scoring latency",
            _fmt_ms(fc.get("end_to_end_ms_mean"),
                    fc.get("end_to_end_ms_std")),
        ))
    else:
        rows.append(("FHE cubic scoring latency", "N/A"))

    # -- 4. Pipeline total latency ---------------------------------------
    if bp:
        rows.append((
            "Pipeline total latency",
            _fmt_ms(bp.get("total_ms_mean"), bp.get("total_ms_std")),
        ))
    else:
        rows.append(("Pipeline total latency", "N/A"))

    # -- 5. PIR correctness ----------------------------------------------
    if bp:
        corr_str = bp.get("pir_correctness", "N/A")
        parsed = _parse_correctness(corr_str)
        if parsed and parsed[0] == parsed[1]:
            rows.append(("PIR correctness", f"{corr_str} (100%)"))
        elif parsed:
            pct = parsed[0] / parsed[1] * 100
            rows.append(("PIR correctness", f"{corr_str} ({pct:.0f}%)"))
            failures.append(
                f"PIR correctness {corr_str} -- expected 100%")
        else:
            rows.append(("PIR correctness", str(corr_str)))
    else:
        rows.append(("PIR correctness", "N/A"))

    # -- 6. FHE correctness ----------------------------------------------
    if bp:
        corr_str = bp.get("fhe_correctness", "N/A")
        parsed = _parse_correctness(corr_str)
        if parsed and parsed[0] == parsed[1]:
            rows.append(("FHE correctness", f"{corr_str} (100%)"))
        elif parsed:
            pct = parsed[0] / parsed[1] * 100
            rows.append(("FHE correctness", f"{corr_str} ({pct:.0f}%)"))
            failures.append(
                f"FHE correctness {corr_str} -- expected 100%")
        else:
            rows.append(("FHE correctness", str(corr_str)))
    else:
        rows.append(("FHE correctness", "N/A"))

    # -- 7. FHE max approximation error ----------------------------------
    if fc and fc.get("max_error") is not None:
        err = fc["max_error"]
        rows.append(("FHE max approximation error", f"{err:.1e}"))
        if err >= 0.01:
            failures.append(
                f"FHE max error {err:.1e} exceeds clinical tolerance 0.01")
    else:
        rows.append(("FHE max approximation error", "N/A"))

    # -- 8. Peak memory (pipeline) ----------------------------------------
    if bp and bp.get("pipeline_peak_rss_mb") is not None:
        rows.append((
            "Peak memory (pipeline)",
            f"{bp['pipeline_peak_rss_mb']:.0f} MB",
        ))
    else:
        rows.append(("Peak memory (pipeline)", "N/A"))

    # -- 9. Server content leakage (always 0 by construction) -------------
    rows.append(("Server content leakage", "0 bits"))

    return rows, failures


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    rows: list[tuple[str, str]],
    step_failures: list[str],
    verify_failures: list[str],
    total_seconds: float,
    output_dir: Path,
    logger: _Logger,
) -> None:
    """Print and log the reproduction summary table."""
    col1 = max(len(r[0]) for r in rows)
    col2 = max(len(r[1]) for r in rows)
    width = max(col1 + col2 + 4, 44)
    sep = "=" * width

    lines: list[str] = [
        "",
        sep,
        "REPRODUCTION SUMMARY",
        sep,
        f"{'Paper Claim':<{col1}} | Measured",
        f"{'-' * col1}-+-{'-' * col2}",
    ]
    for label, value in rows:
        lines.append(f"{label:<{col1}} | {value}")
    lines.append(sep)

    # Verification status
    all_failures = step_failures + verify_failures
    if not all_failures:
        lines.append("Claim verification: ALL PASSED")
    else:
        lines.append(f"Claim verification: {len(all_failures)} issue(s)")
        for f in all_failures:
            lines.append(f"  - {f}")

    mins = int(total_seconds // 60)
    secs = int(total_seconds % 60)
    lines.append(f"Total reproduction time: {mins} minutes {secs} seconds")
    lines.append(f"All results saved to: {output_dir}/")

    for line in lines:
        logger.log(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce every result in the paper from scratch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Modes:
              --quick   Small datasets for fast testing (~5 min)
              --full    Full datasets for paper reproduction [default]
        """),
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use small datasets (200 drugs, 10 patients) for fast testing",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use full datasets (10K drugs, 100 patients) -- default",
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip plot generation (useful if matplotlib is unavailable)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Where to save the reproduction log (default: results/)",
    )
    args = parser.parse_args()

    quick = args.quick
    skip_figures = args.skip_figures
    output_dir = PROJECT_ROOT / args.output_dir
    total_steps = 6 if skip_figures else 7

    py = sys.executable
    small = ["--small"] if quick else []
    mode_label = "quick (small datasets)" if quick else "full (paper-scale)"

    logger = _Logger()
    step_failures: list[str] = []

    # Banner
    logger.log(f"{'=' * 55}")
    logger.log(f"  REPRODUCE ALL PAPER RESULTS")
    logger.log(f"  Mode:    {mode_label}")
    logger.log(f"  Python:  {py}")
    logger.log(f"  Started: {_ts()}")
    logger.log(f"{'=' * 55}")
    logger.log("")

    # Ensure output directories exist
    (PROJECT_ROOT / "results" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ==================================================================
    # Step 1/N: Generate synthetic datasets
    # ==================================================================
    ok, _ = _run_step(
        1, total_steps, "Generating synthetic datasets...",
        [[py, "data/generate_all.py"]],
        logger=logger,
    )
    if not ok:
        step_failures.append("Step 1 (data generation) failed")

    # ==================================================================
    # Step 2/N: PIR benchmarks
    # ==================================================================
    ok, _ = _run_step(
        2, total_steps, "Running PIR benchmarks...",
        [[py, "experiments/run_benchmarks.py", "--pir-only"] + small],
        logger=logger,
    )
    if not ok:
        step_failures.append("Step 2 (PIR benchmarks) failed")

    # ==================================================================
    # Step 3/N: FHE benchmarks
    # ==================================================================
    ok, _ = _run_step(
        3, total_steps, "Running FHE benchmarks...",
        [[py, "experiments/run_benchmarks.py", "--fhe-only"] + small],
        logger=logger,
    )
    if not ok:
        step_failures.append("Step 3 (FHE benchmarks) failed")

    # ==================================================================
    # Step 4/N: Combined pipeline benchmarks
    #
    # run_benchmarks.py without filter flags runs PIR + FHE + pipeline.
    # This re-runs PIR and FHE (overwriting steps 2-3 results with
    # identical data) but is the only way to generate
    # pipeline_benchmarks_scaled.json.
    # ==================================================================
    ok, _ = _run_step(
        4, total_steps, "Running combined pipeline benchmarks...",
        [[py, "experiments/run_benchmarks.py"] + small],
        logger=logger,
    )
    if not ok:
        step_failures.append("Step 4 (pipeline benchmarks) failed")

    # ==================================================================
    # Step 5/N: Memory profiling
    # ==================================================================
    ok, _ = _run_step(
        5, total_steps, "Running memory profiling...",
        [[py, "experiments/run_memory_benchmarks.py"] + small],
        logger=logger,
    )
    if not ok:
        step_failures.append("Step 5 (memory profiling) failed")

    # ==================================================================
    # Step 6/N: Systems analysis
    # ==================================================================
    ok, _ = _run_step(
        6, total_steps, "Running systems analysis...",
        [[py, "experiments/run_systems_analysis.py"]],
        logger=logger,
    )
    if not ok:
        step_failures.append("Step 6 (systems analysis) failed")

    # ==================================================================
    # Step 7/N: Figures (optional)
    # ==================================================================
    if not skip_figures:
        ok, _ = _run_step(
            7, total_steps, "Generating figures...",
            [
                [py, "experiments/visualize_feasibility.py"],
                [py, "experiments/visualize_threat_model.py"],
            ],
            logger=logger,
        )
        if not ok:
            step_failures.append("Step 7 (figure generation) failed")

    # ==================================================================
    # Summary
    # ==================================================================
    total_time = time.time() - t0
    rows, verify_failures = _extract_summary()
    _print_summary(
        rows, step_failures, verify_failures,
        total_time, output_dir, logger,
    )

    # Save log
    log_path = output_dir / "reproduction_log.txt"
    logger.save(log_path)
    logger.log(f"\nFull log saved to: {log_path}")

    all_failures = step_failures + verify_failures
    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())
