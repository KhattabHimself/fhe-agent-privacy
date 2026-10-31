#!/usr/bin/env python3
"""Regenerate all datasets from scratch with a fixed random seed.

Usage:
    python data/generate_all.py          # from project root
    python -m data.generate_all          # as module

Outputs:
    data/drug_interactions_10k.json      10,000 drug-pair interaction records
    data/drug_interactions_200.json      200-record subset for quick testing
    data/synthetic_patients_1000.json    1,000 patient profiles (50 edge cases)
    data/synthetic_patients_10.json      10-patient subset for quick testing

All outputs are deterministic (seed=42) and fully reproducible.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running from project root: `python data/generate_all.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_utils import generate_all_datasets, get_dataset_stats


def main() -> None:
    print("=" * 60)
    print("  FHE-Agent-Privacy: Dataset Generation")
    print("  Seed: 42 (deterministic, reproducible)")
    print("=" * 60)

    t0 = time.perf_counter()
    paths = generate_all_datasets(seed=42)
    elapsed = time.perf_counter() - t0

    print(f"\n--- Generation complete in {elapsed:.1f}s ---")
    print(f"\nFiles created:")
    for name, path in paths.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name:<40s}  {size_kb:>8.1f} KB")

    # Print summary statistics
    print()
    get_dataset_stats()

    print(f"\nAll datasets generated successfully.")


if __name__ == "__main__":
    main()
