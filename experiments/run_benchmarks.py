"""Experiment: Scaled benchmarks for PIR, FHE, and the combined pipeline.

Replaces the old toy benchmarks with publication-quality measurements
using the full 10K drug-interaction database and 1000-patient dataset.

Benchmarks produced:
  1. PIR at scale — 100 to 10K records, 256-byte records, 20 queries each
  2. FHE at scale — Light (n=8192) and Deep (n=16384) on 100 real patients
  3. Combined pipeline — PIR lookup + FHE scoring on 100 patients vs 10K DB
  4. Hardware info header on every results JSON

Usage:
    python -m experiments.run_benchmarks [--small] [--pir-only] [--fhe-only]

    --small     Run reduced configurations for quick testing.
    --pir-only  Skip FHE and pipeline benchmarks.
    --fhe-only  Skip PIR and pipeline benchmarks.
"""

from __future__ import annotations

import datetime
import gc
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

import tenseal as ts

from src.feasibility.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    memory_tracker,
)
from src.feasibility.fhe_computation import (
    EncryptedLinearModel,
    EncryptedPolynomialModel,
    FHEContext,
    FHEParams,
)
from src.feasibility.pir_lookup import ComputationalPIR, PIRParams, TrivialPIR
from src.prototype.encrypted_scoring import (
    EncryptedScoringTool,
    PatientFeatures,
    cardiovascular_risk_model,
    simple_linear_model,
)
from src.utils.data_utils import load_drug_db, load_patients

RESULTS_DIR = Path("results/benchmarks")
MB = 1024 * 1024
KB = 1024


# ===================================================================
# Hardware info
# ===================================================================

def get_hardware_info() -> dict:
    """Collect hardware and software metadata."""
    mem = psutil.virtual_memory()
    info = {
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(mem.total / (1024**3), 1),
        "os": f"{platform.system()} {platform.version()}",
        "python": sys.version.split()[0],
        "tenseal": ts.__version__,
        "numpy": np.__version__,
        "platform_node": platform.node(),
    }
    return info


def wrap_results(results: dict | list, label: str) -> dict:
    """Wrap benchmark results with hardware info and timestamp."""
    return {
        "benchmark": label,
        "hardware": get_hardware_info(),
        "timestamp": datetime.datetime.now().isoformat(),
        "results": results,
    }


def save_json(data: dict, path: Path) -> None:
    """Save dict as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved {path}")


# ===================================================================
# PIR benchmarks at scale
# ===================================================================

def _make_records(n: int, record_size: int = 256) -> list[bytes]:
    """Generate n synthetic records of given size."""
    return [f"record-{i:06d}-data".encode().ljust(record_size, b"\x00")
            for i in range(n)]


def _measure_bfv_key_sizes(params: PIRParams, n: int) -> dict:
    """Measure BFV key and ciphertext serialization sizes."""
    ctx = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=params.poly_modulus_degree,
        plain_modulus=params.plain_modulus,
    )
    full_bytes = ctx.serialize(save_secret_key=True)
    ctx.make_context_public()
    pub_bytes = ctx.serialize()

    ctx_full = ts.context_from(full_bytes)
    vec = [0] * min(n, params.poly_modulus_degree)
    enc = ts.bfv_vector(ctx_full, vec)
    ct_bytes = enc.serialize()

    return {
        "full_context_bytes": len(full_bytes),
        "public_context_bytes": len(pub_bytes),
        "ciphertext_bytes": len(ct_bytes),
    }


def benchmark_pir_scaled(suite: BenchmarkSuite,
                         db_sizes: list[int],
                         n_queries: int = 20) -> list[dict]:
    """PIR benchmarks: 256-byte records, multiple random queries per DB size."""
    print("\n" + "=" * 70)
    print("  PART 1: PIR Benchmarks at Scale")
    print("=" * 70)

    rng = np.random.default_rng(42)
    record_size = 256
    all_results = []

    for n in db_sizes:
        print(f"\n  DB size = {n:,} records × {record_size} bytes")
        records = _make_records(n, record_size)
        params = PIRParams(db_size=n, record_size_bytes=record_size)

        # --- Key / ciphertext sizes (once per DB size) ---
        key_info = _measure_bfv_key_sizes(params, n)
        print(f"    Key sizes: context={key_info['full_context_bytes']/KB:.1f} KB, "
              f"ciphertext={key_info['ciphertext_bytes']/KB:.1f} KB")

        # --- Setup with memory tracking ---
        gc.collect()
        with memory_tracker() as mem_setup:
            t0 = time.perf_counter()
            cpir = ComputationalPIR(params)
            cpir.setup()
            cpir.load_database(records)
            setup_ms = (time.perf_counter() - t0) * 1000
        ms = mem_setup[0]
        print(f"    Setup: {setup_ms:.0f} ms | "
              f"RSS={ms.peak_rss_mb:.1f} MB, delta={ms.rss_delta_mb:+.1f} MB")

        suite.add_result(BenchmarkResult(
            mechanism="PIR", operation="setup",
            params={"db_size": n, "record_size": record_size},
            latency_ms=setup_ms,
            memory_peak_mb=ms.peak_rss_mb,
            memory_delta_mb=ms.rss_delta_mb,
            key_size_mb=key_info["full_context_bytes"] / MB,
            ciphertext_size_kb=key_info["ciphertext_bytes"] / KB,
        ))

        # --- Communication cost (once per DB size) ---
        costs = cpir.communication_cost()
        print(f"    Communication: query={costs['query_size']/KB:.1f} KB, "
              f"response={costs['response_size']/KB:.1f} KB, "
              f"total={costs['total']/KB:.1f} KB")

        suite.add_result(BenchmarkResult(
            mechanism="PIR", operation="communication",
            params={"db_size": n, "record_size": record_size},
            latency_ms=0,
            communication_bytes=costs["total"],
        ))

        # --- Per-query benchmarks (n_queries random indices) ---
        targets = rng.integers(0, n, size=n_queries).tolist()
        gen_times, proc_times, dec_times, e2e_times = [], [], [], []
        correctness_ok = 0

        for qi, target in enumerate(targets):
            # Query generation
            t0 = time.perf_counter()
            q = cpir.generate_query(target)
            t_gen = (time.perf_counter() - t0) * 1000
            gen_times.append(t_gen)

            # Server processing
            t0 = time.perf_counter()
            r = cpir.process_query(q)
            t_proc = (time.perf_counter() - t0) * 1000
            proc_times.append(t_proc)

            # Decryption
            t0 = time.perf_counter()
            result = cpir.decrypt_result(r)
            t_dec = (time.perf_counter() - t0) * 1000
            dec_times.append(t_dec)

            e2e_times.append(t_gen + t_proc + t_dec)

            # Correctness
            if result.rstrip(b"\x00") == records[target].rstrip(b"\x00"):
                correctness_ok += 1

        gen_arr = np.array(gen_times)
        proc_arr = np.array(proc_times)
        dec_arr = np.array(dec_times)
        e2e_arr = np.array(e2e_times)

        print(f"    Queries: {n_queries} random indices")
        print(f"      Query gen:    {gen_arr.mean():.1f} ± {gen_arr.std():.1f} ms")
        print(f"      Server proc:  {proc_arr.mean():.1f} ± {proc_arr.std():.1f} ms")
        print(f"      Decrypt:      {dec_arr.mean():.1f} ± {dec_arr.std():.1f} ms")
        print(f"      End-to-end:   {e2e_arr.mean():.1f} ± {e2e_arr.std():.1f} ms")
        print(f"      Correctness:  {correctness_ok}/{n_queries}")
        assert correctness_ok == n_queries, \
            f"PIR correctness failure at db_size={n}: {correctness_ok}/{n_queries}"

        for op, arr in [("query_gen", gen_arr), ("server_process", proc_arr),
                        ("decrypt", dec_arr), ("end_to_end", e2e_arr)]:
            suite.add_result(BenchmarkResult(
                mechanism="PIR", operation=op,
                params={"db_size": n, "record_size": record_size},
                latency_ms=float(arr.mean()),
                repetitions=n_queries,
            ))

        # --- Trivial PIR baseline ---
        tpir = TrivialPIR(records)
        t0 = time.perf_counter()
        _ = tpir.query(targets[0])
        trivial_ms = (time.perf_counter() - t0) * 1000
        trivial_cost = tpir.communication_cost()
        print(f"    Trivial PIR:    {trivial_ms:.3f} ms, "
              f"{trivial_cost/KB:.1f} KB comm")

        suite.add_result(BenchmarkResult(
            mechanism="TrivialPIR", operation="end_to_end",
            params={"db_size": n, "record_size": record_size},
            latency_ms=trivial_ms,
            communication_bytes=trivial_cost,
        ))

        # --- Memory during a single query cycle ---
        gc.collect()
        with memory_tracker() as mem_q:
            q = cpir.generate_query(targets[0])
            r = cpir.process_query(q)
            _ = cpir.decrypt_result(r)
        mq = mem_q[0]
        print(f"    Query cycle RSS: peak={mq.peak_rss_mb:.1f} MB, "
              f"delta={mq.rss_delta_mb:+.1f} MB")

        suite.add_result(BenchmarkResult(
            mechanism="PIR", operation="query_cycle_memory",
            params={"db_size": n},
            latency_ms=float(e2e_arr.mean()),
            memory_peak_mb=mq.peak_rss_mb,
            memory_delta_mb=mq.rss_delta_mb,
        ))

        all_results.append({
            "db_size": n,
            "record_size": record_size,
            "n_queries": n_queries,
            "setup_ms": setup_ms,
            "setup_peak_rss_mb": ms.peak_rss_mb,
            "setup_rss_delta_mb": ms.rss_delta_mb,
            "query_gen_ms_mean": float(gen_arr.mean()),
            "query_gen_ms_std": float(gen_arr.std()),
            "server_process_ms_mean": float(proc_arr.mean()),
            "server_process_ms_std": float(proc_arr.std()),
            "decrypt_ms_mean": float(dec_arr.mean()),
            "decrypt_ms_std": float(dec_arr.std()),
            "end_to_end_ms_mean": float(e2e_arr.mean()),
            "end_to_end_ms_std": float(e2e_arr.std()),
            "query_cycle_peak_rss_mb": mq.peak_rss_mb,
            "key_context_bytes": key_info["full_context_bytes"],
            "ciphertext_bytes": key_info["ciphertext_bytes"],
            "communication_query_bytes": costs["query_size"],
            "communication_response_bytes": costs["response_size"],
            "communication_total_bytes": costs["total"],
            "trivial_pir_ms": trivial_ms,
            "trivial_pir_comm_bytes": trivial_cost,
            "correctness": f"{correctness_ok}/{n_queries}",
        })

        del cpir, records
        gc.collect()

    return all_results


# ===================================================================
# FHE benchmarks at scale
# ===================================================================

def _load_patient_features(n: int = 100, seed: int = 42) -> list[PatientFeatures]:
    """Load real patient profiles and convert to PatientFeatures."""
    try:
        patients_raw = load_patients("1000")
    except FileNotFoundError:
        print("    WARNING: 1000-patient dataset not found, using random patients")
        rng = np.random.default_rng(seed)
        return [PatientFeatures.random(rng) for _ in range(n)]

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(patients_raw), size=min(n, len(patients_raw)),
                         replace=False)
    result = []
    for i in indices:
        p = patients_raw[i]
        result.append(PatientFeatures(
            age=p["age"],
            bmi=p["bmi"],
            blood_pressure_systolic=p["systolic_bp"],
            blood_pressure_diastolic=p["diastolic_bp"],
            cholesterol=p["cholesterol_total"],
            glucose=p["glucose_fasting"],
            num_medications=p["num_medications"],
            num_conditions=p["num_conditions"],
        ))
    return result


def _measure_ckks_key_sizes(params: FHEParams) -> dict:
    """Measure CKKS key and ciphertext serialization sizes."""
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=params.poly_modulus_degree,
        coeff_mod_bit_sizes=params.coeff_mod_bit_sizes,
    )
    ctx.global_scale = params.global_scale
    full_no_galois = ctx.serialize(save_secret_key=True)

    # Public context WITHOUT Galois keys (for size comparison)
    ctx.make_context_public()
    pub_no_galois = ctx.serialize()

    # Restore and add Galois keys
    ctx = ts.context_from(full_no_galois)
    ctx.generate_galois_keys()
    full_with_galois = ctx.serialize(save_secret_key=True)

    # Public context WITH Galois keys
    ctx.make_context_public()
    pub_with_galois = ctx.serialize()
    galois_size = len(pub_with_galois) - len(pub_no_galois)

    # Restore full context for ciphertext measurement
    ctx = ts.context_from(full_with_galois)
    enc = ts.ckks_vector(ctx, [1.0, 2.0, 3.0, 4.0])
    ct_size = len(enc.serialize())

    return {
        "full_context_bytes": len(full_with_galois),
        "context_no_galois_bytes": len(full_no_galois),
        "galois_key_bytes": galois_size,
        "ciphertext_bytes": ct_size,
    }


def benchmark_fhe_scaled(suite: BenchmarkSuite,
                         n_patients: int = 100,
                         small_mode: bool = False) -> list[dict]:
    """FHE benchmarks: Light and Deep presets on real patient profiles."""
    print("\n" + "=" * 70)
    print("  PART 2: FHE Benchmarks at Scale")
    print("=" * 70)

    patients = _load_patient_features(n_patients)
    print(f"  Loaded {len(patients)} patient profiles")

    configs = [
        {
            "label": "linear",
            "preset": "light",
            "params": FHEParams.light(),
            "model_fn": simple_linear_model,
            "has_activation": False,
            "tolerance": 0.01,
        },
        {
            "label": "cv_sigmoid",
            "preset": "deep",
            "params": FHEParams.deep(),
            "model_fn": cardiovascular_risk_model,
            "has_activation": True,
            "tolerance": 0.05,
        },
    ]

    if small_mode:
        n_scoring = min(10, len(patients))
    else:
        n_scoring = min(n_patients, len(patients))

    all_results = []

    for cfg in configs:
        label = cfg["label"]
        params = cfg["params"]
        poly_mod = params.poly_modulus_degree
        print(f"\n  Model: {label} | poly_mod={poly_mod} | "
              f"depth={params.multiplicative_depth}")

        # --- Key sizes ---
        key_info = _measure_ckks_key_sizes(params)
        print(f"    Key sizes: context={key_info['full_context_bytes']/MB:.2f} MB, "
              f"Galois={key_info['galois_key_bytes']/MB:.2f} MB, "
              f"ciphertext={key_info['ciphertext_bytes']/KB:.1f} KB")

        suite.add_result(BenchmarkResult(
            mechanism="FHE", operation="key_sizes",
            params={"model": label, "poly_mod": poly_mod},
            latency_ms=0,
            key_size_mb=key_info["full_context_bytes"] / MB,
            ciphertext_size_kb=key_info["ciphertext_bytes"] / KB,
        ))

        # --- Context creation ---
        gc.collect()
        with memory_tracker() as mem_ctx:
            t0 = time.perf_counter()
            if cfg["has_activation"]:
                w, b, act = cfg["model_fn"]()
                scorer = EncryptedScoringTool(params)
                scorer.setup(w, b, activation_coeffs=act)
            else:
                w, b = cfg["model_fn"]()
                scorer = EncryptedScoringTool(params)
                scorer.setup(w, b)
            ctx_ms = (time.perf_counter() - t0) * 1000
        mc = mem_ctx[0]
        print(f"    Context setup: {ctx_ms:.0f} ms | "
              f"RSS={mc.peak_rss_mb:.1f} MB, delta={mc.rss_delta_mb:+.1f} MB")

        suite.add_result(BenchmarkResult(
            mechanism="FHE", operation="context_setup",
            params={"model": label, "poly_mod": poly_mod},
            latency_ms=ctx_ms,
            memory_peak_mb=mc.peak_rss_mb,
            memory_delta_mb=mc.rss_delta_mb,
            key_size_mb=key_info["full_context_bytes"] / MB,
        ))

        # --- Per-patient scoring ---
        enc_times, comp_times, dec_times, e2e_times = [], [], [], []
        errors = []
        correctness_ok = 0
        scoring_patients = patients[:n_scoring]

        for pi, patient in enumerate(scoring_patients):
            # Encryption
            t0 = time.perf_counter()
            enc_feat = scorer.encrypt_features(patient)
            t_enc = (time.perf_counter() - t0) * 1000
            enc_times.append(t_enc)

            # Server computation
            t0 = time.perf_counter()
            enc_score = scorer.compute_score_encrypted(enc_feat)
            t_comp = (time.perf_counter() - t0) * 1000
            comp_times.append(t_comp)

            # Decryption
            t0 = time.perf_counter()
            dec_val = scorer.decrypt_score(enc_score)
            t_dec = (time.perf_counter() - t0) * 1000
            dec_times.append(t_dec)

            e2e_times.append(t_enc + t_comp + t_dec)

            # Correctness check
            plain_val = scorer.score_plaintext(patient)
            error = abs(dec_val - plain_val)
            errors.append(error)
            if error < cfg["tolerance"]:
                correctness_ok += 1

            if pi < 3 or pi == n_scoring - 1:
                print(f"      Patient {pi}: enc={t_enc:.1f}ms, "
                      f"comp={t_comp:.1f}ms, dec={t_dec:.1f}ms, "
                      f"error={error:.2e}")

        enc_arr = np.array(enc_times)
        comp_arr = np.array(comp_times)
        dec_arr = np.array(dec_times)
        e2e_arr = np.array(e2e_times)
        err_arr = np.array(errors)

        print(f"    Summary ({n_scoring} patients):")
        print(f"      Encrypt:     {enc_arr.mean():.2f} ± {enc_arr.std():.2f} ms")
        print(f"      Compute:     {comp_arr.mean():.2f} ± {comp_arr.std():.2f} ms")
        print(f"      Decrypt:     {dec_arr.mean():.2f} ± {dec_arr.std():.2f} ms")
        print(f"      End-to-end:  {e2e_arr.mean():.2f} ± {e2e_arr.std():.2f} ms")
        print(f"      Max error:   {err_arr.max():.2e}")
        print(f"      Correctness: {correctness_ok}/{n_scoring}")
        assert correctness_ok == n_scoring, (
            f"FHE correctness failure for {label}: "
            f"{correctness_ok}/{n_scoring}")

        for op, arr in [("encrypt", enc_arr), ("compute", comp_arr),
                        ("decrypt", dec_arr), ("end_to_end", e2e_arr)]:
            suite.add_result(BenchmarkResult(
                mechanism="FHE", operation=op,
                params={"model": label, "poly_mod": poly_mod},
                latency_ms=float(arr.mean()),
                repetitions=n_scoring,
            ))

        # Memory during scoring
        gc.collect()
        with memory_tracker() as mem_score:
            _ = scorer.score(scoring_patients[0])
        ms_score = mem_score[0]
        print(f"    Scoring RSS: peak={ms_score.peak_rss_mb:.1f} MB, "
              f"delta={ms_score.rss_delta_mb:+.1f} MB")

        suite.add_result(BenchmarkResult(
            mechanism="FHE", operation="scoring_memory",
            params={"model": label, "poly_mod": poly_mod},
            latency_ms=float(e2e_arr.mean()),
            memory_peak_mb=ms_score.peak_rss_mb,
            memory_delta_mb=ms_score.rss_delta_mb,
        ))

        # Plaintext baseline
        t0 = time.perf_counter()
        for _ in range(10000):
            scorer.score_plaintext(scoring_patients[0])
        plain_ms = (time.perf_counter() - t0) * 1000 / 10000
        overhead = float(e2e_arr.mean()) / max(plain_ms, 1e-6)
        print(f"    Plaintext: {plain_ms:.4f} ms (overhead: {overhead:.0f}x)")

        suite.add_result(BenchmarkResult(
            mechanism="plaintext", operation="patient_scoring",
            params={"model": label},
            latency_ms=plain_ms,
        ))
        suite.add_result(BenchmarkResult(
            mechanism="FHE", operation="overhead_factor",
            params={"model": label, "poly_mod": poly_mod},
            latency_ms=overhead,
        ))

        all_results.append({
            "model": label,
            "preset": cfg["preset"],
            "poly_mod": poly_mod,
            "depth": params.multiplicative_depth,
            "n_patients": n_scoring,
            "context_setup_ms": ctx_ms,
            "context_setup_peak_rss_mb": mc.peak_rss_mb,
            "context_setup_rss_delta_mb": mc.rss_delta_mb,
            "encrypt_ms_mean": float(enc_arr.mean()),
            "encrypt_ms_std": float(enc_arr.std()),
            "compute_ms_mean": float(comp_arr.mean()),
            "compute_ms_std": float(comp_arr.std()),
            "decrypt_ms_mean": float(dec_arr.mean()),
            "decrypt_ms_std": float(dec_arr.std()),
            "end_to_end_ms_mean": float(e2e_arr.mean()),
            "end_to_end_ms_std": float(e2e_arr.std()),
            "scoring_peak_rss_mb": ms_score.peak_rss_mb,
            "max_error": float(err_arr.max()),
            "mean_error": float(err_arr.mean()),
            "plaintext_ms": plain_ms,
            "overhead_factor": overhead,
            "key_context_bytes": key_info["full_context_bytes"],
            "galois_key_bytes": key_info["galois_key_bytes"],
            "ciphertext_bytes": key_info["ciphertext_bytes"],
            "correctness": f"{correctness_ok}/{n_scoring}",
        })

        del scorer
        gc.collect()

    return all_results


# ===================================================================
# Combined pipeline benchmarks (PIR + FHE)
# ===================================================================

def benchmark_pipeline(suite: BenchmarkSuite,
                       n_patients: int = 100,
                       small_mode: bool = False) -> list[dict]:
    """Full pipeline: PIR drug lookup + FHE risk scoring on real data."""
    print("\n" + "=" * 70)
    print("  PART 3: Combined Pipeline Benchmarks (PIR + FHE)")
    print("=" * 70)

    if small_mode:
        n_patients = min(5, n_patients)
        db_size_tag = "200"
    else:
        db_size_tag = "10k"

    # Load real drug database
    try:
        drug_db = load_drug_db(db_size_tag)
    except FileNotFoundError:
        print(f"  WARNING: Drug DB '{db_size_tag}' not found. "
              f"Run `python data/generate_all.py` first.")
        return []

    patients = _load_patient_features(n_patients)
    n_patients = len(patients)
    print(f"  Drug DB: {len(drug_db):,} records ({db_size_tag})")
    print(f"  Patients: {n_patients}")

    # Build serialized records for PIR (256-byte aligned)
    serialized_records = []
    for rec in drug_db:
        compact = json.dumps(rec, separators=(",", ":")).encode("utf-8")
        serialized_records.append(compact.ljust(256, b"\x00"))

    db_n = len(serialized_records)
    pir_params = PIRParams(db_size=db_n, record_size_bytes=256)

    # --- PIR setup ---
    print("\n  Setting up PIR...")
    gc.collect()
    with memory_tracker() as mem_pir:
        t0 = time.perf_counter()
        cpir = ComputationalPIR(pir_params)
        cpir.setup()
        cpir.load_database(serialized_records)
        pir_setup_ms = (time.perf_counter() - t0) * 1000
    mp = mem_pir[0]
    print(f"    PIR setup: {pir_setup_ms:.0f} ms | "
          f"RSS={mp.peak_rss_mb:.1f} MB")

    # --- FHE setup ---
    print("  Setting up FHE (CV model, deep params)...")
    gc.collect()
    with memory_tracker() as mem_fhe:
        t0 = time.perf_counter()
        w_cv, b_cv, act_cv = cardiovascular_risk_model()
        scorer = EncryptedScoringTool(FHEParams.deep())
        scorer.setup(w_cv, b_cv, activation_coeffs=act_cv)
        fhe_setup_ms = (time.perf_counter() - t0) * 1000
    mf = mem_fhe[0]
    print(f"    FHE setup: {fhe_setup_ms:.0f} ms | "
          f"RSS={mf.peak_rss_mb:.1f} MB")

    # --- Pipeline: for each patient, do PIR lookup + FHE scoring ---
    print(f"\n  Running pipeline on {n_patients} patients...")
    rng = np.random.default_rng(42)

    pir_times, fhe_times, decision_times, total_times = [], [], [], []
    pir_correct, fhe_correct = 0, 0
    all_run_results = []

    for pi in range(n_patients):
        patient = patients[pi]
        # Pick a random drug pair to look up
        target_idx = int(rng.integers(0, db_n))

        # --- PIR phase ---
        t0 = time.perf_counter()
        q = cpir.generate_query(target_idx)
        r = cpir.process_query(q)
        raw_result = cpir.decrypt_result(r)
        t_pir = (time.perf_counter() - t0) * 1000
        pir_times.append(t_pir)

        # Verify PIR correctness
        expected_raw = serialized_records[target_idx]
        if raw_result.rstrip(b"\x00") == expected_raw.rstrip(b"\x00"):
            pir_correct += 1

        # Parse the record
        try:
            record = json.loads(raw_result.rstrip(b"\x00").decode("utf-8"))
        except Exception:
            record = {"severity": "none"}

        # --- FHE phase ---
        t0 = time.perf_counter()
        enc_score = scorer.score(patient)
        t_fhe = (time.perf_counter() - t0) * 1000
        fhe_times.append(t_fhe)

        plain_score = scorer.score_plaintext(patient)
        if abs(enc_score - plain_score) < 0.05:
            fhe_correct += 1

        # --- Local decision phase ---
        t0 = time.perf_counter()
        # Simple decision logic: combine drug severity with risk score
        sev_str = record.get("severity", "none")
        sev_map = {"none": 0, "mild": 1, "moderate": 2,
                   "severe": 3, "contraindicated": 4}
        sev_int = sev_map.get(str(sev_str).lower(),
                              int(sev_str) if str(sev_str).isdigit() else 0)
        risk_level = "low" if enc_score < 0.3 else (
            "medium" if enc_score < 0.6 else "high")
        if sev_int >= 3:
            decision = "ALERT"
        elif risk_level == "high" and sev_int >= 2:
            decision = "WARNING"
        else:
            decision = "OK"
        t_decision = (time.perf_counter() - t0) * 1000
        decision_times.append(t_decision)

        t_total = t_pir + t_fhe + t_decision
        total_times.append(t_total)

        if pi < 3 or pi == n_patients - 1:
            print(f"    Patient {pi}: PIR={t_pir:.0f}ms, FHE={t_fhe:.0f}ms, "
                  f"decision={decision}, score={enc_score:.4f}")

    pir_arr = np.array(pir_times)
    fhe_arr = np.array(fhe_times)
    dec_arr = np.array(decision_times)
    tot_arr = np.array(total_times)

    print(f"\n  Pipeline Summary ({n_patients} patients):")
    print(f"    PIR phase:      {pir_arr.mean():.0f} ± {pir_arr.std():.0f} ms")
    print(f"    FHE phase:      {fhe_arr.mean():.0f} ± {fhe_arr.std():.0f} ms")
    print(f"    Decision:       {dec_arr.mean():.4f} ± {dec_arr.std():.4f} ms")
    print(f"    Total:          {tot_arr.mean():.0f} ± {tot_arr.std():.0f} ms")
    print(f"    PIR correct:    {pir_correct}/{n_patients}")
    print(f"    FHE correct:    {fhe_correct}/{n_patients}")

    # Overall pipeline memory
    gc.collect()
    with memory_tracker() as mem_pipeline:
        target = int(rng.integers(0, db_n))
        q = cpir.generate_query(target)
        r = cpir.process_query(q)
        raw = cpir.decrypt_result(r)
        _ = scorer.score(patients[0])
    mpl = mem_pipeline[0]
    print(f"    Pipeline RSS:   peak={mpl.peak_rss_mb:.1f} MB, "
          f"delta={mpl.rss_delta_mb:+.1f} MB")

    for op, arr in [("pir_phase", pir_arr), ("fhe_phase", fhe_arr),
                    ("decision", dec_arr), ("total", tot_arr)]:
        suite.add_result(BenchmarkResult(
            mechanism="pipeline", operation=op,
            params={"db_size": db_n, "n_patients": n_patients},
            latency_ms=float(arr.mean()),
            repetitions=n_patients,
        ))

    suite.add_result(BenchmarkResult(
        mechanism="pipeline", operation="memory",
        params={"db_size": db_n},
        latency_ms=float(tot_arr.mean()),
        memory_peak_mb=mpl.peak_rss_mb,
        memory_delta_mb=mpl.rss_delta_mb,
    ))

    pipeline_result = {
        "db_size": db_n,
        "db_tag": db_size_tag,
        "n_patients": n_patients,
        "pir_setup_ms": pir_setup_ms,
        "pir_setup_peak_rss_mb": mp.peak_rss_mb,
        "fhe_setup_ms": fhe_setup_ms,
        "fhe_setup_peak_rss_mb": mf.peak_rss_mb,
        "pir_phase_ms_mean": float(pir_arr.mean()),
        "pir_phase_ms_std": float(pir_arr.std()),
        "fhe_phase_ms_mean": float(fhe_arr.mean()),
        "fhe_phase_ms_std": float(fhe_arr.std()),
        "decision_ms_mean": float(dec_arr.mean()),
        "decision_ms_std": float(dec_arr.std()),
        "total_ms_mean": float(tot_arr.mean()),
        "total_ms_std": float(tot_arr.std()),
        "pipeline_peak_rss_mb": mpl.peak_rss_mb,
        "pir_correctness": f"{pir_correct}/{n_patients}",
        "fhe_correctness": f"{fhe_correct}/{n_patients}",
    }

    del cpir, scorer
    gc.collect()

    return [pipeline_result]


# ===================================================================
# Plots (updated for new data structure)
# ===================================================================

def generate_pir_plots(df, results_dir: Path) -> None:
    """Generate PIR comparison plots."""
    fig, ax = plt.subplots(figsize=(8, 5))
    pir_e2e = df[(df["mechanism"] == "PIR") & (df["operation"] == "end_to_end")]
    trivial_e2e = df[(df["mechanism"] == "TrivialPIR") &
                     (df["operation"] == "end_to_end")]
    if not pir_e2e.empty:
        ax.plot(pir_e2e["param_db_size"], pir_e2e["latency_ms"],
                "o-", color="#2D6A4F", label="Computational PIR", linewidth=2)
    if not trivial_e2e.empty:
        ax.plot(trivial_e2e["param_db_size"], trivial_e2e["latency_ms"],
                "s--", color="#95D5B2", label="Trivial PIR", linewidth=2)
    ax.set_xlabel("Database Size (records)", fontsize=12)
    ax.set_ylabel("End-to-End Latency (ms)", fontsize=12)
    ax.set_title("PIR Query Latency vs Database Size (256B records)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "pir_latency_vs_dbsize.png", dpi=300)
    fig.savefig(results_dir / "pir_latency_vs_dbsize.pdf")
    plt.close(fig)
    print("  Saved pir_latency_vs_dbsize.{png,pdf}")

    # Communication cost
    fig, ax = plt.subplots(figsize=(8, 5))
    pir_comm = df[(df["mechanism"] == "PIR") & (df["operation"] == "communication")]
    if not pir_comm.empty:
        ax.plot(pir_comm["param_db_size"],
                pir_comm["communication_bytes"] / 1024,
                "o-", color="#2D6A4F", label="Computational PIR", linewidth=2)
    if not trivial_e2e.empty:
        ax.plot(trivial_e2e["param_db_size"],
                trivial_e2e["communication_bytes"] / 1024,
                "s--", color="#95D5B2", label="Trivial PIR", linewidth=2)
    ax.set_xlabel("Database Size (records)", fontsize=12)
    ax.set_ylabel("Communication Cost (KB)", fontsize=12)
    ax.set_title("Communication Cost: Computational vs Trivial PIR", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "pir_communication_cost.png", dpi=300)
    fig.savefig(results_dir / "pir_communication_cost.pdf")
    plt.close(fig)
    print("  Saved pir_communication_cost.{png,pdf}")

    # Operation breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    ops = ["query_gen", "server_process", "decrypt"]
    colors = ["#2D6A4F", "#52B788", "#95D5B2"]
    for op, color in zip(ops, colors):
        subset = df[(df["mechanism"] == "PIR") & (df["operation"] == op)]
        if not subset.empty:
            ax.plot(subset["param_db_size"], subset["latency_ms"],
                    "o-", color=color, label=op.replace("_", " ").title(),
                    linewidth=2)
    ax.set_xlabel("Database Size (records)", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("PIR Operation Breakdown (256B records)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "pir_operation_breakdown.png", dpi=300)
    fig.savefig(results_dir / "pir_operation_breakdown.pdf")
    plt.close(fig)
    print("  Saved pir_operation_breakdown.{png,pdf}")


def generate_fhe_plots(df, results_dir: Path) -> None:
    """Generate FHE comparison plots."""
    # Patient scoring: FHE vs plaintext
    fig, ax = plt.subplots(figsize=(7, 5))
    patient_data = df[
        (df["mechanism"] == "FHE") & (df["operation"] == "end_to_end")]
    plain_data = df[
        (df["mechanism"] == "plaintext") & (df["operation"] == "patient_scoring")]

    if not patient_data.empty and "param_model" in patient_data.columns:
        labels = patient_data["param_model"].tolist()
        fhe_vals = patient_data["latency_ms"].tolist()

        if not plain_data.empty:
            labels.append("plaintext")
            fhe_vals.append(float(plain_data["latency_ms"].iloc[0]))

        colors = ["#264653", "#2A9D8F", "#95D5B2"][:len(labels)]
        bars = ax.bar(labels, fhe_vals, color=colors, width=0.5)
        ax.bar_label(bars, fmt="%.1f ms", fontsize=10)
        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.set_title("Patient Risk Scoring: FHE vs Plaintext", fontsize=14)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(results_dir / "fhe_patient_scoring.png", dpi=300)
    fig.savefig(results_dir / "fhe_patient_scoring.pdf")
    plt.close(fig)
    print("  Saved fhe_patient_scoring.{png,pdf}")

    # FHE operation breakdown (bar chart per model)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, model_label in zip(axes, ["linear", "cv_sigmoid"]):
        ops_data = []
        for op in ["encrypt", "compute", "decrypt"]:
            row = df[(df["mechanism"] == "FHE") &
                     (df["operation"] == op)]
            if "param_model" in df.columns:
                row = row[row["param_model"] == model_label]
            if not row.empty:
                ops_data.append((op.title(), float(row["latency_ms"].iloc[0])))
        if ops_data:
            labels_op, vals = zip(*ops_data)
            colors_op = ["#264653", "#2A9D8F", "#E9C46A"][:len(labels_op)]
            bars = ax.bar(labels_op, vals, color=colors_op, width=0.5)
            ax.bar_label(bars, fmt="%.1f ms", fontsize=10)
            ax.set_ylabel("Latency (ms)", fontsize=12)
            ax.set_title(f"FHE {model_label} Operation Breakdown", fontsize=13)
            ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(results_dir / "fhe_operation_breakdown.png", dpi=300)
    fig.savefig(results_dir / "fhe_operation_breakdown.pdf")
    plt.close(fig)
    print("  Saved fhe_operation_breakdown.{png,pdf}")


def generate_pipeline_plots(df, results_dir: Path) -> None:
    """Generate pipeline comparison plots."""
    fig, ax = plt.subplots(figsize=(8, 5))
    pipeline = df[df["mechanism"] == "pipeline"]
    if not pipeline.empty:
        ops = ["pir_phase", "fhe_phase", "decision"]
        colors = ["#264653", "#2A9D8F", "#E9C46A"]
        vals, labels = [], []
        for op, color in zip(ops, colors):
            row = pipeline[pipeline["operation"] == op]
            if not row.empty:
                vals.append(float(row["latency_ms"].iloc[0]))
                labels.append(op.replace("_", " ").title())
        if vals:
            bars = ax.bar(labels, vals, color=colors[:len(vals)], width=0.5)
            ax.bar_label(bars, fmt="%.0f ms", fontsize=10)
            ax.set_ylabel("Latency (ms)", fontsize=12)
            ax.set_title("Pipeline Phase Breakdown (PIR + FHE)", fontsize=14)
            ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(results_dir / "pipeline_breakdown.png", dpi=300)
    fig.savefig(results_dir / "pipeline_breakdown.pdf")
    plt.close(fig)
    print("  Saved pipeline_breakdown.{png,pdf}")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    """Run all scaled benchmarks."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    small_mode = "--small" in sys.argv
    pir_only = "--pir-only" in sys.argv
    fhe_only = "--fhe-only" in sys.argv
    run_pir = not fhe_only
    run_fhe = not pir_only
    run_pipeline = not pir_only and not fhe_only

    print("=" * 70)
    print("  RQ2: Scaled Cryptographic Mechanism Benchmarks")
    print("=" * 70)
    hw = get_hardware_info()
    print(f"  CPU: {hw['cpu']} ({hw['cpu_cores_logical']} threads)")
    print(f"  RAM: {hw['ram_total_gb']} GB")
    print(f"  OS:  {hw['os']}")
    print(f"  Python: {hw['python']}, TenSEAL: {hw['tenseal']}")

    suite = BenchmarkSuite(name="fhe-agent-privacy-scaled-benchmarks")

    if small_mode:
        print("\n  (Running in --small mode)")
        n_queries = 5
        n_patients = 10
    else:
        n_queries = 20
        n_patients = 100

    # ---------------------------------------------------------------
    # Part 1: PIR
    # ---------------------------------------------------------------
    pir_results = []
    if run_pir:
        db_sizes = ([100, 500] if small_mode
                    else [100, 500, 1000, 5000, 10000])
        pir_results = benchmark_pir_scaled(suite, db_sizes, n_queries)
        save_json(
            wrap_results(pir_results, "PIR benchmarks (scaled)"),
            RESULTS_DIR / "pir_benchmarks_scaled.json",
        )

    # ---------------------------------------------------------------
    # Part 2: FHE
    # ---------------------------------------------------------------
    fhe_results = []
    if run_fhe:
        fhe_results = benchmark_fhe_scaled(suite, n_patients, small_mode)
        save_json(
            wrap_results(fhe_results, "FHE benchmarks (scaled)"),
            RESULTS_DIR / "fhe_benchmarks_scaled.json",
        )

    # ---------------------------------------------------------------
    # Part 3: Pipeline
    # ---------------------------------------------------------------
    pipeline_results = []
    if run_pipeline:
        pipeline_results = benchmark_pipeline(suite, n_patients, small_mode)
        save_json(
            wrap_results(pipeline_results, "Pipeline benchmarks (scaled)"),
            RESULTS_DIR / "pipeline_benchmarks_scaled.json",
        )

    # ---------------------------------------------------------------
    # Combined output (backward compatible)
    # ---------------------------------------------------------------
    df = suite.to_dataframe()

    # Plots
    print("\nGenerating plots...")
    if run_pir:
        generate_pir_plots(df, RESULTS_DIR)
    if run_fhe:
        generate_fhe_plots(df, RESULTS_DIR)
    if run_pipeline:
        generate_pipeline_plots(df, RESULTS_DIR)

    # Save combined results (backward compatible with reproduce_all.py)
    all_records = json.loads(df.to_json(orient="records"))
    save_json(
        wrap_results(all_records, "All benchmarks (scaled)"),
        RESULTS_DIR / "benchmarks.json",
    )
    suite.save_csv(RESULTS_DIR / "benchmarks.csv")

    # Summary table
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(suite.summary_table())

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
