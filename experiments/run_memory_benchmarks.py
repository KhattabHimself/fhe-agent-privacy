"""Experiment: Memory and key-size benchmarks for FHE and PIR.

Measures OS-level RSS, Python-level allocations, serialized key sizes,
and ciphertext sizes for each cryptographic primitive across parameter
configurations relevant to the paper (RQ2 supplement).

Usage:
    python -m experiments.run_memory_benchmarks [--small]

    --small   Run only the smallest configurations for quick testing.

Results are saved to results/memory_benchmarks.json.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tenseal as ts

from src.feasibility.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    MemorySnapshot,
    memory_tracker,
)
from src.feasibility.fhe_computation import (
    EncryptedLinearModel,
    FHEContext,
    FHEParams,
)
from src.feasibility.pir_lookup import ComputationalPIR, PIRParams

RESULTS_DIR = Path("results/benchmarks")
MB = 1024 * 1024
KB = 1024


# ===================================================================
# Key / ciphertext size measurement helpers
# ===================================================================

def measure_bfv_sizes(poly_mod: int = 8192, plain_mod: int = 1_032_193,
                      db_size: int = 100) -> dict[str, float]:
    """Measure serialized sizes of BFV keys and a single ciphertext."""
    ctx = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=poly_mod,
        plain_modulus=plain_mod,
    )
    full_bytes = ctx.serialize(save_secret_key=True)
    full_size = len(full_bytes)

    ctx.make_context_public()
    pub_bytes = ctx.serialize()
    pub_size = len(pub_bytes)

    # Single ciphertext: encrypt a small vector
    ctx_full = ts.context_from(full_bytes)
    vec = [0] * min(db_size, poly_mod)
    enc = ts.bfv_vector(ctx_full, vec)
    ct_size = len(enc.serialize())

    return {
        "public_key_mb": pub_size / MB,
        "full_context_mb": full_size / MB,
        "ciphertext_kb": ct_size / KB,
    }


def measure_ckks_sizes(params: FHEParams) -> dict[str, float]:
    """Measure serialized sizes of CKKS keys (incl. Galois) and ciphertext."""
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=params.poly_modulus_degree,
        coeff_mod_bit_sizes=params.coeff_mod_bit_sizes,
    )
    ctx.global_scale = params.global_scale

    # Context WITHOUT Galois keys
    full_no_galois = ctx.serialize(save_secret_key=True)
    ctx.make_context_public()
    pub_no_galois = ctx.serialize()

    # Restore and add Galois keys
    ctx = ts.context_from(full_no_galois)
    ctx.generate_galois_keys()
    full_with_galois = ctx.serialize(save_secret_key=True)
    ctx.make_context_public()
    pub_with_galois = ctx.serialize()

    galois_key_size = len(pub_with_galois) - len(pub_no_galois)

    # Restore full context for encryption
    ctx = ts.context_from(full_with_galois)
    enc = ts.ckks_vector(ctx, [1.0, 2.0, 3.0, 4.0])
    ct_size = len(enc.serialize())

    return {
        "public_context_mb": len(pub_no_galois) / MB,
        "galois_key_mb": galois_key_size / MB,
        "full_context_mb": len(full_with_galois) / MB,
        "ciphertext_kb": ct_size / KB,
    }


# ===================================================================
# PIR memory benchmarks
# ===================================================================

def benchmark_pir_memory(suite: BenchmarkSuite,
                         db_sizes: list[int]) -> list[dict]:
    """Benchmark PIR memory usage across database sizes."""
    print("\n  PIR Memory Benchmarks")
    print("  " + "-" * 60)

    summary_rows = []

    for n in db_sizes:
        print(f"\n  DB size = {n:,}")
        record_size = 128
        records = [f"record-{i:06d}-".encode().ljust(record_size, b"\x00")
                   for i in range(n)]
        params = PIRParams(db_size=n, record_size_bytes=record_size)

        # --- Key sizes ---
        sizes = measure_bfv_sizes(
            poly_mod=params.poly_modulus_degree,
            plain_mod=params.plain_modulus,
            db_size=n,
        )
        print(f"    BFV context (pub+sk):  {sizes['full_context_mb']:.3f} MB")
        print(f"    BFV public only:       {sizes['public_key_mb']:.3f} MB")
        print(f"    BFV ciphertext:        {sizes['ciphertext_kb']:.1f} KB")

        suite.add_result(BenchmarkResult(
            mechanism="PIR", operation="key_sizes",
            params={"db_size": n, "poly_mod": params.poly_modulus_degree},
            latency_ms=0,
            key_size_mb=sizes["full_context_mb"],
            ciphertext_size_kb=sizes["ciphertext_kb"],
        ))

        # --- Setup memory ---
        gc.collect()
        with memory_tracker() as mem_setup:
            t0 = time.perf_counter()
            cpir = ComputationalPIR(params)
            cpir.setup()
            cpir.load_database(records)
            setup_ms = (time.perf_counter() - t0) * 1000
        ms = mem_setup[0]
        print(f"    Setup:  {setup_ms:.0f} ms | "
              f"RSS={ms.peak_rss_mb:.1f} MB, delta={ms.rss_delta_mb:+.1f} MB, "
              f"tracemalloc_peak={ms.tracemalloc_peak_mb:.1f} MB")

        suite.add_result(BenchmarkResult(
            mechanism="PIR", operation="setup",
            params={"db_size": n},
            latency_ms=setup_ms,
            memory_peak_mb=ms.peak_rss_mb,
            memory_delta_mb=ms.rss_delta_mb,
            key_size_mb=sizes["full_context_mb"],
        ))

        # --- Query encryption memory ---
        target = n // 2
        gc.collect()
        with memory_tracker() as mem_qgen:
            t0 = time.perf_counter()
            q = cpir.generate_query(target)
            qgen_ms = (time.perf_counter() - t0) * 1000
        mq = mem_qgen[0]
        print(f"    Query gen:  {qgen_ms:.1f} ms | "
              f"delta={mq.rss_delta_mb:+.1f} MB, "
              f"tracemalloc_peak={mq.tracemalloc_peak_mb:.1f} MB")

        suite.add_result(BenchmarkResult(
            mechanism="PIR", operation="query_encrypt",
            params={"db_size": n},
            latency_ms=qgen_ms,
            memory_peak_mb=mq.peak_rss_mb,
            memory_delta_mb=mq.rss_delta_mb,
        ))

        # --- Server processing memory ---
        gc.collect()
        with memory_tracker() as mem_proc:
            t0 = time.perf_counter()
            r = cpir.process_query(q)
            proc_ms = (time.perf_counter() - t0) * 1000
        mp = mem_proc[0]
        print(f"    Server proc:  {proc_ms:.1f} ms | "
              f"delta={mp.rss_delta_mb:+.1f} MB, "
              f"tracemalloc_peak={mp.tracemalloc_peak_mb:.1f} MB")

        suite.add_result(BenchmarkResult(
            mechanism="PIR", operation="server_process",
            params={"db_size": n},
            latency_ms=proc_ms,
            memory_peak_mb=mp.peak_rss_mb,
            memory_delta_mb=mp.rss_delta_mb,
        ))

        # --- Decryption memory ---
        gc.collect()
        with memory_tracker() as mem_dec:
            t0 = time.perf_counter()
            result = cpir.decrypt_result(r)
            dec_ms = (time.perf_counter() - t0) * 1000
        md = mem_dec[0]
        print(f"    Decrypt:  {dec_ms:.1f} ms | "
              f"delta={md.rss_delta_mb:+.1f} MB")

        suite.add_result(BenchmarkResult(
            mechanism="PIR", operation="decrypt",
            params={"db_size": n},
            latency_ms=dec_ms,
            memory_peak_mb=md.peak_rss_mb,
            memory_delta_mb=md.rss_delta_mb,
        ))

        # Verify correctness
        assert result.rstrip(b"\x00") == records[target].rstrip(b"\x00"), \
            f"PIR decryption mismatch at db_size={n}"

        summary_rows.append({
            "operation": f"PIR ({n:,} records)",
            "peak_rss_mb": ms.peak_rss_mb,
            "key_size_mb": sizes["full_context_mb"],
            "ciphertext_kb": sizes["ciphertext_kb"],
        })

        # Free memory before next iteration
        del cpir, records, q, r, result
        gc.collect()

    return summary_rows


# ===================================================================
# FHE memory benchmarks
# ===================================================================

def benchmark_fhe_memory(suite: BenchmarkSuite,
                         poly_mods: list[int]) -> list[dict]:
    """Benchmark FHE (CKKS) memory across polynomial modulus degrees."""
    print("\n  FHE Memory Benchmarks")
    print("  " + "-" * 60)

    summary_rows = []
    rng = np.random.default_rng(42)

    for poly_mod in poly_mods:
        print(f"\n  poly_modulus_degree = {poly_mod}")

        if poly_mod <= 8192:
            params = FHEParams.light()
        else:
            params = FHEParams.deep()

        # Override poly_mod if needed (the presets already match)
        assert params.poly_modulus_degree == poly_mod, \
            f"Preset mismatch: expected {poly_mod}, got {params.poly_modulus_degree}"

        # --- Key sizes ---
        print("    Measuring key sizes...")
        try:
            sizes = measure_ckks_sizes(params)
            print(f"    CKKS public context:  {sizes['public_context_mb']:.3f} MB")
            print(f"    CKKS Galois keys:    {sizes['galois_key_mb']:.2f} MB")
            print(f"    CKKS full context:   {sizes['full_context_mb']:.2f} MB")
            print(f"    CKKS ciphertext:     {sizes['ciphertext_kb']:.1f} KB")

            suite.add_result(BenchmarkResult(
                mechanism="FHE", operation="key_sizes",
                params={"poly_mod": poly_mod,
                        "galois_key_mb": sizes["galois_key_mb"]},
                latency_ms=0,
                key_size_mb=sizes["full_context_mb"],
                ciphertext_size_kb=sizes["ciphertext_kb"],
            ))
        except Exception as e:
            print(f"    KEY SIZE MEASUREMENT FAILED: {e}")
            sizes = {"full_context_mb": 0, "galois_key_mb": 0,
                     "ciphertext_kb": 0}

        gc.collect()

        # --- Context creation + Galois key generation ---
        print("    Measuring context creation (incl. Galois keys)...")
        try:
            gc.collect()
            with memory_tracker() as mem_ctx:
                t0 = time.perf_counter()
                ctx = FHEContext(params)
                ctx.setup()  # includes generate_galois_keys()
                ctx_ms = (time.perf_counter() - t0) * 1000
            mc = mem_ctx[0]
            print(f"    Context setup:  {ctx_ms:.0f} ms | "
                  f"RSS={mc.peak_rss_mb:.1f} MB, delta={mc.rss_delta_mb:+.1f} MB, "
                  f"tracemalloc_peak={mc.tracemalloc_peak_mb:.1f} MB")

            suite.add_result(BenchmarkResult(
                mechanism="FHE", operation="context_setup",
                params={"poly_mod": poly_mod},
                latency_ms=ctx_ms,
                memory_peak_mb=mc.peak_rss_mb,
                memory_delta_mb=mc.rss_delta_mb,
                key_size_mb=sizes["full_context_mb"],
            ))
        except MemoryError:
            print(f"    OOM during context setup for poly_mod={poly_mod}!")
            print(f"    This is expected for large poly_mod values.")
            suite.add_result(BenchmarkResult(
                mechanism="FHE", operation="context_setup",
                params={"poly_mod": poly_mod, "status": "OOM"},
                latency_ms=-1,
                memory_peak_mb=-1,
            ))
            summary_rows.append({
                "operation": f"FHE (n={poly_mod})",
                "peak_rss_mb": -1,
                "key_size_mb": sizes["full_context_mb"],
                "ciphertext_kb": sizes["ciphertext_kb"],
                "note": "OOM during context setup",
            })
            gc.collect()
            continue
        except Exception as e:
            print(f"    Context setup FAILED: {e}")
            summary_rows.append({
                "operation": f"FHE (n={poly_mod})",
                "peak_rss_mb": -1,
                "key_size_mb": sizes["full_context_mb"],
                "ciphertext_kb": sizes["ciphertext_kb"],
                "note": f"Failed: {e}",
            })
            gc.collect()
            continue

        # --- Encryption memory ---
        dim = 8
        features = rng.uniform(0, 1, size=dim).tolist()
        weights = rng.uniform(-1, 1, size=dim).tolist()

        gc.collect()
        with memory_tracker() as mem_enc:
            t0 = time.perf_counter()
            enc_x = ctx.encrypt_vector(features)
            enc_ms = (time.perf_counter() - t0) * 1000
        me = mem_enc[0]
        print(f"    Encryption:  {enc_ms:.2f} ms | "
              f"delta={me.rss_delta_mb:+.1f} MB")

        suite.add_result(BenchmarkResult(
            mechanism="FHE", operation="encrypt",
            params={"poly_mod": poly_mod, "dim": dim},
            latency_ms=enc_ms,
            memory_peak_mb=me.peak_rss_mb,
            memory_delta_mb=me.rss_delta_mb,
            ciphertext_size_kb=sizes["ciphertext_kb"],
        ))

        # --- Computation memory (dot product) ---
        model = EncryptedLinearModel(weights, 0.5)
        enc_x = ctx.encrypt_vector(features)

        gc.collect()
        with memory_tracker() as mem_comp:
            t0 = time.perf_counter()
            enc_result = model.predict_encrypted(enc_x)
            comp_ms = (time.perf_counter() - t0) * 1000
        mcomp = mem_comp[0]
        print(f"    Computation (dot):  {comp_ms:.2f} ms | "
              f"delta={mcomp.rss_delta_mb:+.1f} MB")

        suite.add_result(BenchmarkResult(
            mechanism="FHE", operation="compute_dot",
            params={"poly_mod": poly_mod, "dim": dim},
            latency_ms=comp_ms,
            memory_peak_mb=mcomp.peak_rss_mb,
            memory_delta_mb=mcomp.rss_delta_mb,
        ))

        # --- Computation memory (polynomial eval) ---
        if poly_mod >= 16384:
            coeffs = [0.5, 0.197, 0.0, -0.004]  # cubic sigmoid approx
            label = "cubic"
        else:
            coeffs = [0.5, 0.197]  # linear activation
            label = "linear"

        enc_x = ctx.encrypt_vector(features)
        enc_dot = model.predict_encrypted(enc_x)

        gc.collect()
        try:
            with memory_tracker() as mem_poly:
                t0 = time.perf_counter()
                enc_activated = enc_dot.polyval(coeffs)
                poly_ms = (time.perf_counter() - t0) * 1000
            mpoly = mem_poly[0]
            print(f"    Polynomial ({label}):  {poly_ms:.2f} ms | "
                  f"delta={mpoly.rss_delta_mb:+.1f} MB")

            suite.add_result(BenchmarkResult(
                mechanism="FHE", operation=f"compute_poly_{label}",
                params={"poly_mod": poly_mod, "degree": len(coeffs) - 1},
                latency_ms=poly_ms,
                memory_peak_mb=mpoly.peak_rss_mb,
                memory_delta_mb=mpoly.rss_delta_mb,
            ))
        except Exception as e:
            print(f"    Polynomial eval FAILED: {e}")

        # --- Decryption memory ---
        enc_x = ctx.encrypt_vector(features)
        enc_result = model.predict_encrypted(enc_x)

        gc.collect()
        with memory_tracker() as mem_dec:
            t0 = time.perf_counter()
            val = ctx.decrypt_result(enc_result)
            dec_ms = (time.perf_counter() - t0) * 1000
        mdec = mem_dec[0]
        print(f"    Decryption:  {dec_ms:.2f} ms | "
              f"delta={mdec.rss_delta_mb:+.1f} MB")

        suite.add_result(BenchmarkResult(
            mechanism="FHE", operation="decrypt",
            params={"poly_mod": poly_mod},
            latency_ms=dec_ms,
            memory_peak_mb=mdec.peak_rss_mb,
            memory_delta_mb=mdec.rss_delta_mb,
        ))

        # Verify correctness
        expected = model.predict_plaintext(features)
        error = abs(val - expected)
        print(f"    Correctness: enc={val:.6f}, plain={expected:.6f}, "
              f"error={error:.2e}")

        summary_rows.append({
            "operation": f"FHE {label} (n={poly_mod})",
            "peak_rss_mb": mc.peak_rss_mb,
            "key_size_mb": sizes["full_context_mb"],
            "galois_key_mb": sizes["galois_key_mb"],
            "ciphertext_kb": sizes["ciphertext_kb"],
        })

        # Free memory
        del ctx, enc_x, enc_result, model
        gc.collect()

    return summary_rows


# ===================================================================
# Summary table
# ===================================================================

def print_summary_table(pir_rows: list[dict], fhe_rows: list[dict]) -> None:
    """Print a formatted summary table."""
    from tabulate import tabulate

    headers = ["Operation", "Peak RSS (MB)", "Key Size (MB)",
               "Galois Keys (MB)", "Ciphertext (KB)", "Note"]
    rows = []

    for r in pir_rows:
        rows.append([
            r["operation"],
            f"{r['peak_rss_mb']:.1f}" if r["peak_rss_mb"] > 0 else "N/A",
            f"{r['key_size_mb']:.3f}",
            "-",
            f"{r['ciphertext_kb']:.1f}",
            r.get("note", ""),
        ])

    for r in fhe_rows:
        peak = r.get("peak_rss_mb", -1)
        rows.append([
            r["operation"],
            f"{peak:.1f}" if peak > 0 else "OOM/Failed",
            f"{r['key_size_mb']:.2f}" if r.get("key_size_mb") else "N/A",
            f"{r.get('galois_key_mb', 0):.2f}" if r.get("galois_key_mb") else "-",
            f"{r['ciphertext_kb']:.1f}" if r.get("ciphertext_kb") else "N/A",
            r.get("note", ""),
        ])

    print(f"\n{'='*80}")
    print("  MEMORY BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    """Run all memory benchmarks."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    small_mode = "--small" in sys.argv

    print("=" * 70)
    print("  Memory & Key-Size Benchmarks")
    print("=" * 70)

    suite = BenchmarkSuite(name="memory-benchmarks")

    if small_mode:
        pir_db_sizes = [100, 500]
        fhe_poly_mods = [8192]
        print("\n  (Running in --small mode)")
    else:
        pir_db_sizes = [100, 500, 1000, 5000, 10000]
        fhe_poly_mods = [8192, 16384]

    # --- PIR ---
    pir_rows = benchmark_pir_memory(suite, pir_db_sizes)

    # --- FHE ---
    fhe_rows = benchmark_fhe_memory(suite, fhe_poly_mods)

    # --- Summary ---
    print_summary_table(pir_rows, fhe_rows)

    # --- Save results ---
    out_path = RESULTS_DIR / "memory_benchmarks.json"
    df = suite.to_dataframe()
    records = json.loads(df.to_json(orient="records"))
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Also save summary
    summary_path = RESULTS_DIR / "memory_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"pir": pir_rows, "fhe": fhe_rows}, f, indent=2)
    print(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
