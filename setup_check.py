"""Verify all project dependencies are importable and print a status table."""

import importlib
import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

DEPENDENCIES = [
    ("tenseal", "FHE backend (CKKS + BFV)"),
    ("Crypto", "PyCryptodome encryption"),
    ("numpy", "numerical computation"),
    ("pandas", "data manipulation"),
    ("sklearn", "machine learning"),
    ("matplotlib", "plotting"),
    ("tabulate", "table formatting"),
    ("memory_profiler", "memory benchmarking"),
    ("psutil", "system resource monitoring"),
    ("pytest", "testing framework"),
]


def check_imports():
    results = []
    for module_name, description in DEPENDENCIES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "n/a")
            results.append((True, module_name, version, description))
        except ImportError as e:
            results.append((False, module_name, str(e), description))
    return results


def main():
    results = check_imports()
    failed = [r for r in results if not r[0]]

    print(f"\n{'='*60}")
    print(f" fhe-agent-privacy Environment Check")
    print(f"{'='*60}")
    print(f" Python {sys.version}")
    print(f"{'='*60}\n")

    for ok, name, version, desc in results:
        icon = "\u2713" if ok else "\u2717"
        status = version if ok else "MISSING"
        print(f"  {icon} {name:<20} {status:<14} {desc}")

    print(f"\n{'='*60}")
    if not failed:
        print(" Environment ready!")
    else:
        print(f" {len(failed)} package(s) failed to import:")
        for _, name, err, _ in failed:
            print(f"   - {name}: {err}")
    print(f"{'='*60}\n")

    return len(failed) == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
