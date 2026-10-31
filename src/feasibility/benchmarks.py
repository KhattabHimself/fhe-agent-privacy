"""Benchmarking framework for cryptographic operations.

Measures latency, throughput, memory usage, and communication cost
for each privacy mechanism (PIR, FHE, ORAM) across varying parameters.
Results are stored as structured data for plotting and inclusion in
the paper.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np
import pandas as pd
import psutil
from tabulate import tabulate


# ===================================================================
# Memory tracking
# ===================================================================

@dataclass
class MemorySnapshot:
    """Result of a memory-tracked operation."""

    peak_rss_mb: float
    rss_delta_mb: float
    tracemalloc_peak_mb: float
    tracemalloc_delta_mb: float


@contextmanager
def memory_tracker() -> Generator[list[MemorySnapshot], None, None]:
    """Context manager that measures memory consumption of enclosed code.

    Uses both OS-level RSS (via psutil) and Python-level allocation
    tracking (via tracemalloc).  The result is stored in a single-element
    list so the caller can retrieve it after the ``with`` block.

    Usage::

        with memory_tracker() as mem:
            heavy_operation()
        snapshot = mem[0]
        print(f"Peak RSS: {snapshot.peak_rss_mb:.1f} MB")
    """
    result: list[MemorySnapshot] = []

    gc.collect()
    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.stop()
    tracemalloc.start()
    tm_before = tracemalloc.get_traced_memory()[0]

    yield result

    tm_current, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if was_tracing:
        tracemalloc.start()

    rss_after = proc.memory_info().rss
    peak_rss = rss_after  # conservative: rss_after >= rss during

    MB = 1024 * 1024
    result.append(MemorySnapshot(
        peak_rss_mb=peak_rss / MB,
        rss_delta_mb=(rss_after - rss_before) / MB,
        tracemalloc_peak_mb=tm_peak / MB,
        tracemalloc_delta_mb=(tm_current - tm_before) / MB,
    ))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes
    ----------
    mechanism : str
        Name of the mechanism ("PIR", "FHE", "ORAM", "plaintext").
    operation : str
        Specific operation measured ("setup", "query", "compute").
    params : dict
        Parameter configuration used for this run.
    latency_ms : float
        Wall-clock latency in milliseconds.
    throughput_ops_sec : float
        Operations per second (if applicable).
    memory_peak_mb : float
        Peak memory usage in megabytes (OS-level RSS).
    memory_delta_mb : float
        Memory increase during the operation (RSS after - before).
    key_size_mb : float
        Serialized size of cryptographic keys in megabytes.
    ciphertext_size_kb : float
        Serialized size of a single ciphertext in kilobytes.
    communication_bytes : int
        Total bytes transferred (if applicable).
    repetitions : int
        Number of repetitions averaged over.
    """

    mechanism: str
    operation: str
    params: dict[str, Any]
    latency_ms: float
    throughput_ops_sec: float = 0.0
    memory_peak_mb: float = 0.0
    memory_delta_mb: float = 0.0
    key_size_mb: float = 0.0
    ciphertext_size_kb: float = 0.0
    communication_bytes: int = 0
    repetitions: int = 1


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results with export utilities."""

    name: str
    results: list[BenchmarkResult] = field(default_factory=list)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the suite."""
        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis."""
        rows = []
        for r in self.results:
            row = {
                "mechanism": r.mechanism,
                "operation": r.operation,
                "latency_ms": r.latency_ms,
                "throughput_ops_sec": r.throughput_ops_sec,
                "memory_peak_mb": r.memory_peak_mb,
                "memory_delta_mb": r.memory_delta_mb,
                "key_size_mb": r.key_size_mb,
                "ciphertext_size_kb": r.ciphertext_size_kb,
                "communication_bytes": r.communication_bytes,
                "repetitions": r.repetitions,
            }
            for k, v in r.params.items():
                row[f"param_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def save_csv(self, path: Path) -> None:
        """Export results to CSV."""
        self.to_dataframe().to_csv(path, index=False)

    def save_json(self, path: Path) -> None:
        """Export results to JSON."""
        df = self.to_dataframe()
        df.to_json(path, orient="records", indent=2)

    def summary_table(self) -> str:
        """Generate a human-readable summary table."""
        headers = [
            "Mechanism", "Operation", "Params",
            "Latency (ms)", "Peak RSS (MB)", "Mem Delta (MB)",
            "Key (MB)", "CT (KB)", "Comm. (bytes)",
        ]
        rows = []
        for r in self.results:
            param_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
            rows.append([
                r.mechanism, r.operation, param_str,
                f"{r.latency_ms:.2f}",
                f"{r.memory_peak_mb:.1f}" if r.memory_peak_mb else "-",
                f"{r.memory_delta_mb:.1f}" if r.memory_delta_mb else "-",
                f"{r.key_size_mb:.2f}" if r.key_size_mb else "-",
                f"{r.ciphertext_size_kb:.1f}" if r.ciphertext_size_kb else "-",
                f"{r.communication_bytes:,}" if r.communication_bytes else "-",
            ])
        return tabulate(rows, headers=headers, tablefmt="grid")


def measure_latency(
    func: Callable[..., Any],
    *args: Any,
    repetitions: int = 10,
    warmup: int = 2,
    **kwargs: Any,
) -> dict[str, float]:
    """Measure function execution latency.

    Parameters
    ----------
    func : callable
        The function to benchmark.
    repetitions : int
        Number of timed executions.
    warmup : int
        Number of warmup executions (not timed).

    Returns
    -------
    dict
        Keys: mean_ms, median_ms, std_ms, min_ms, max_ms.
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def measure_memory(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> float:
    """Measure peak memory usage of a function call.

    Returns
    -------
    float
        Peak memory usage in megabytes.
    """
    try:
        from memory_profiler import memory_usage
        mem = memory_usage((func, args, kwargs), max_usage=True, retval=False)
        if isinstance(mem, (list, tuple)):
            return float(max(mem))
        return float(mem)
    except ImportError:
        return 0.0


def benchmark_scaling(
    func: Callable[..., Any],
    param_name: str,
    param_values: list[Any],
    repetitions: int = 5,
) -> list[BenchmarkResult]:
    """Benchmark a function across a range of parameter values.

    The function must accept ``param_name`` as a keyword argument.

    Parameters
    ----------
    func : callable
        Function to benchmark.
    param_name : str
        Name of the parameter to vary.
    param_values : list
        List of values to test.
    repetitions : int
        Repetitions per parameter value.

    Returns
    -------
    list[BenchmarkResult]
        One result per parameter value.
    """
    results = []
    for val in param_values:
        stats = measure_latency(func, repetitions=repetitions, warmup=1,
                                **{param_name: val})
        results.append(BenchmarkResult(
            mechanism="custom",
            operation=func.__name__,
            params={param_name: val},
            latency_ms=stats["mean_ms"],
            repetitions=repetitions,
        ))
    return results
