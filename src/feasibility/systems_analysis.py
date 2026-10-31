"""Systems-level analysis of FHE implementation realities.

Measures the practical friction that papers typically omit:
- CKKS precision loss accumulation across multiplicative depths
- RAM spike profiles during FHE operations
- Galois key size tradeoffs
- Effective noise budget estimation via error growth

These findings feed directly into Section VI of the paper.
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import tenseal as ts

from src.feasibility.fhe_computation import FHEParams

# ── Constants ──────────────────────────────────────────────────────

CLINICAL_TOLERANCE = 0.01  # max acceptable absolute error for risk scores
MB = 1024 * 1024


# ===================================================================
# 1. CKKS Precision Loss Analysis
# ===================================================================

def _measure_error(decrypted: list[float],
                   expected: list[float]) -> dict:
    """Compute absolute and relative error statistics."""
    dec = np.array(decrypted[: len(expected)])
    exp = np.array(expected)
    abs_err = np.abs(dec - exp)
    # avoid division by zero for near-zero expected values
    rel_err = np.where(np.abs(exp) > 1e-10,
                       abs_err / np.abs(exp), abs_err)
    return {
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "max_rel_error": float(np.max(rel_err)),
        "mean_rel_error": float(np.mean(rel_err)),
        "bits_of_precision": float(-np.log2(max(np.max(abs_err), 1e-50))),
        "exceeds_clinical_tolerance": bool(np.max(abs_err) > CLINICAL_TOLERANCE),
    }


def analyze_precision_loss(seed: int = 42) -> dict:
    """Measure CKKS error at increasing multiplicative depths.

    Uses the deep preset (n=16384, depth=4) so we can test the full
    pipeline including cubic polynomial activation.  Also tests the
    light preset for comparison.

    Returns a dict ready for JSON serialisation.
    """
    rng = np.random.default_rng(seed)

    # Values in the clinical range [0, 1] (normalised patient features)
    n_values = 100
    values = rng.uniform(0.0, 1.0, n_values).tolist()

    results: dict = {"presets": []}

    for preset_name, params in [("light", FHEParams.light()),
                                ("deep", FHEParams.deep())]:
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=params.poly_modulus_degree,
            coeff_mod_bit_sizes=params.coeff_mod_bit_sizes,
        )
        ctx.global_scale = params.global_scale
        ctx.generate_galois_keys()

        stages: list[dict] = []

        # ── Stage 0: encrypt / decrypt roundtrip ──
        enc = ts.ckks_vector(ctx, values)
        dec = enc.decrypt()
        s0 = _measure_error(dec, values)
        s0.update(stage="encrypt_decrypt_roundtrip", depth_consumed=0)
        stages.append(s0)

        # ── Stage 1: plaintext scalar multiply (× 2.5) ──
        enc = ts.ckks_vector(ctx, values)
        enc_scaled = enc * 2.5
        dec_scaled = enc_scaled.decrypt()
        expected_scaled = [v * 2.5 for v in values]
        s1 = _measure_error(dec_scaled, expected_scaled)
        s1.update(stage="plaintext_scalar_multiply", depth_consumed=0,
                  note="plaintext×ciphertext does not consume a level")
        stages.append(s1)

        # ── Stage 2: element-wise ciphertext squaring (x²) ──
        enc = ts.ckks_vector(ctx, values)
        enc_sq = enc * enc  # ct × ct → consumes 1 level
        dec_sq = enc_sq.decrypt()
        expected_sq = [v * v for v in values]
        s2 = _measure_error(dec_sq, expected_sq)
        s2.update(stage="element_wise_square", depth_consumed=1)
        stages.append(s2)

        # ── Stage 3: dot product with plaintext weights ──
        weights = rng.uniform(-1.0, 1.0, n_values).tolist()
        enc = ts.ckks_vector(ctx, values)
        enc_dot = enc.dot(weights)
        dec_dot = enc_dot.decrypt()
        expected_dot = [sum(v * w for v, w in zip(values, weights))]
        s3 = _measure_error(dec_dot, expected_dot)
        s3.update(stage="dot_product", depth_consumed=1)
        stages.append(s3)

        # ── Stage 4: cubic polynomial (sigmoid approx.) ──
        # Only the deep preset has enough depth for this
        coeffs = [0.5, 0.197, 0.0, -0.004]  # cubic sigmoid
        try:
            enc = ts.ckks_vector(ctx, values)
            enc_poly = enc.polyval(coeffs)
            dec_poly = enc_poly.decrypt()
            expected_poly = [sum(c * v**i for i, c in enumerate(coeffs))
                             for v in values]
            s4 = _measure_error(dec_poly, expected_poly)
            s4.update(stage="cubic_polynomial", depth_consumed=3)
        except Exception as e:
            s4 = {
                "stage": "cubic_polynomial",
                "depth_consumed": 3,
                "error": f"{type(e).__name__}: {e}",
                "note": ("failed -- insufficient multiplicative depth "
                         f"(preset has depth={params.multiplicative_depth})"),
            }
        stages.append(s4)

        # ── Stage 5: full pipeline (dot product + cubic activation) ──
        values_8 = rng.uniform(0.0, 1.0, 8).tolist()
        w8 = [0.25, 0.15, 0.20, 0.10, 0.15, 0.10, 0.03, 0.02]
        bias = -0.5
        try:
            enc_8 = ts.ckks_vector(ctx, values_8)
            enc_linear = enc_8.dot(w8) + bias
            enc_act = enc_linear.polyval(coeffs)
            dec_act = enc_act.decrypt()

            plain_linear = sum(v * w for v, w in zip(values_8, w8)) + bias
            plain_act = sum(c * plain_linear**i
                           for i, c in enumerate(coeffs))
            s5 = _measure_error(dec_act, [plain_act])
            s5.update(stage="full_pipeline_dot_plus_cubic",
                      depth_consumed=4)
        except Exception as e:
            s5 = {
                "stage": "full_pipeline_dot_plus_cubic",
                "depth_consumed": 4,
                "error": f"{type(e).__name__}: {e}",
                "note": ("failed -- this preset cannot support "
                         "the full pipeline"),
            }
        stages.append(s5)

        # Find first stage that exceeds clinical tolerance
        exceeded_at = None
        for s in stages:
            if s.get("exceeds_clinical_tolerance"):
                exceeded_at = s["stage"]
                break

        results["presets"].append({
            "preset": preset_name,
            "poly_modulus_degree": params.poly_modulus_degree,
            "coeff_mod_bit_sizes": params.coeff_mod_bit_sizes,
            "global_scale": params.global_scale,
            "multiplicative_depth": params.multiplicative_depth,
            "stages": stages,
            "clinical_tolerance": CLINICAL_TOLERANCE,
            "tolerance_first_exceeded": exceeded_at,
        })

    return results


# ===================================================================
# 2. RAM Spike Analysis
# ===================================================================

class MemoryMonitor:
    """Background thread that polls RSS at regular intervals."""

    def __init__(self, interval_ms: int = 50):
        self.interval = interval_ms / 1000.0
        self.timeline: list[dict] = []
        self.phases: list[dict] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process()
        self._t0 = 0.0

    def start(self) -> None:
        gc.collect()
        self._running = True
        self._t0 = time.perf_counter()
        self._record()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def mark_phase(self, name: str) -> None:
        ms = (time.perf_counter() - self._t0) * 1000
        self.phases.append({"timestamp_ms": round(ms, 1), "phase": name})

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._record()

    # ── internals ──

    def _record(self) -> None:
        ms = (time.perf_counter() - self._t0) * 1000
        rss = self._proc.memory_info().rss / MB
        self.timeline.append({
            "timestamp_ms": round(ms, 1),
            "rss_mb": round(rss, 1),
        })

    def _poll(self) -> None:
        while self._running:
            self._record()
            time.sleep(self.interval)

    def get_results(self) -> dict:
        if not self.timeline:
            return {}

        baseline = self.timeline[0]["rss_mb"]
        peak_entry = max(self.timeline, key=lambda t: t["rss_mb"])
        peak = peak_entry["rss_mb"]
        peak_time = peak_entry["timestamp_ms"]

        # Identify which phase the peak fell in
        peak_phase = "before_first_phase"
        for i, p in enumerate(self.phases):
            nxt = (self.phases[i + 1]["timestamp_ms"]
                   if i + 1 < len(self.phases) else float("inf"))
            if p["timestamp_ms"] <= peak_time < nxt:
                peak_phase = p["phase"]
                break

        # Per-phase peak
        phase_peaks: list[dict] = []
        for i, p in enumerate(self.phases):
            t_start = p["timestamp_ms"]
            t_end = (self.phases[i + 1]["timestamp_ms"]
                     if i + 1 < len(self.phases) else float("inf"))
            in_phase = [t for t in self.timeline
                        if t_start <= t["timestamp_ms"] < t_end]
            if in_phase:
                pp = max(in_phase, key=lambda t: t["rss_mb"])
                phase_peaks.append({
                    "phase": p["phase"],
                    "peak_rss_mb": pp["rss_mb"],
                    "delta_from_baseline_mb": round(pp["rss_mb"] - baseline, 1),
                })

        return {
            "baseline_rss_mb": baseline,
            "peak_rss_mb": peak,
            "peak_time_ms": peak_time,
            "peak_phase": peak_phase,
            "spike_factor": round(peak / max(baseline, 1), 2),
            "phase_peaks": phase_peaks,
            "timeline": self.timeline,
            "phases": self.phases,
        }


def analyze_memory_spikes(params: FHEParams | None = None) -> dict:
    """Profile memory during a full FHE pipeline with phase markers.

    Runs both the deep and light presets so we can compare.
    """
    results: dict = {"runs": []}

    for preset_name, p in [("deep", FHEParams.deep()),
                           ("light", FHEParams.light())]:
        run_params = p if params is None else params
        if preset_name == "light" and params is not None:
            continue  # user overrode -- run only once

        rng = np.random.default_rng(42)
        values = rng.uniform(0.0, 1.0, 8).tolist()
        weights = [0.25, 0.15, 0.20, 0.10, 0.15, 0.10, 0.03, 0.02]
        bias = -0.5
        activation = [0.5, 0.197, 0.0, -0.004]

        monitor = MemoryMonitor(interval_ms=50)
        monitor.start()

        try:
            monitor.mark_phase("context_creation")
            ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=run_params.poly_modulus_degree,
                coeff_mod_bit_sizes=run_params.coeff_mod_bit_sizes,
            )
            ctx.global_scale = run_params.global_scale

            monitor.mark_phase("galois_key_generation")
            ctx.generate_galois_keys()

            monitor.mark_phase("encryption")
            enc = ts.ckks_vector(ctx, values)

            monitor.mark_phase("dot_product")
            enc_result = enc.dot(weights) + bias

            if preset_name == "deep":
                monitor.mark_phase("polynomial_activation")
                enc_result = enc_result.polyval(activation)

            monitor.mark_phase("serialization")
            _ = ctx.serialize()

            monitor.mark_phase("decryption")
            _ = enc_result.decrypt()

            monitor.mark_phase("done")
        finally:
            monitor.stop()

        run_result = monitor.get_results()
        run_result["preset"] = preset_name
        run_result["poly_modulus_degree"] = run_params.poly_modulus_degree
        run_result["multiplicative_depth"] = run_params.multiplicative_depth
        results["runs"].append(run_result)

    return results


# ===================================================================
# 3. Galois Key Size Analysis
# ===================================================================

def analyze_galois_keys() -> dict:
    """Measure Galois key sizes and document the rotation-budget tradeoff."""
    configs: list[dict] = []

    for preset_name, params in [("light", FHEParams.light()),
                                ("deep", FHEParams.deep())]:
        # --- Without Galois keys ---
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=params.poly_modulus_degree,
            coeff_mod_bit_sizes=params.coeff_mod_bit_sizes,
        )
        ctx.global_scale = params.global_scale
        full_no_galois = ctx.serialize(save_secret_key=True)
        ctx.make_context_public()
        pub_no_galois = ctx.serialize()

        # --- With Galois keys ---
        ctx = ts.context_from(full_no_galois)
        ctx.generate_galois_keys()
        full_with_galois = ctx.serialize(save_secret_key=True)
        ctx.make_context_public()
        pub_with_galois = ctx.serialize()

        galois_bytes = len(pub_with_galois) - len(pub_no_galois)
        n_slots = params.poly_modulus_degree // 2
        # power-of-2 rotation keys: log2(n_slots) keys
        n_rotation_keys = int(np.log2(n_slots))

        configs.append({
            "preset": preset_name,
            "poly_modulus_degree": params.poly_modulus_degree,
            "n_slots": n_slots,
            "multiplicative_depth": params.multiplicative_depth,
            "context_without_galois_mb": round(len(pub_no_galois) / MB, 2),
            "context_with_galois_mb": round(len(pub_with_galois) / MB, 2),
            "galois_keys_only_mb": round(galois_bytes / MB, 2),
            "galois_pct_of_context": round(
                100 * galois_bytes / max(len(pub_with_galois), 1), 1),
            "full_context_with_sk_mb": round(len(full_with_galois) / MB, 2),
            "n_power_of_2_rotation_keys": n_rotation_keys,
            "estimated_per_key_mb": round(
                galois_bytes / max(n_rotation_keys, 1) / MB, 2),
        })

    # --- Ciphertext size for reference ---
    for cfg in configs:
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=cfg["poly_modulus_degree"],
            coeff_mod_bit_sizes=(
                FHEParams.light().coeff_mod_bit_sizes
                if cfg["preset"] == "light"
                else FHEParams.deep().coeff_mod_bit_sizes
            ),
        )
        ctx.global_scale = 2**40
        enc = ts.ckks_vector(ctx, [0.5] * 8)
        cfg["ciphertext_bytes"] = len(enc.serialize())
        cfg["ciphertext_kb"] = round(cfg["ciphertext_bytes"] / 1024, 1)

    result: dict = {"configurations": configs}

    if len(configs) >= 2:
        light_gk = configs[0]["galois_keys_only_mb"]
        deep_gk = configs[1]["galois_keys_only_mb"]
        result["scaling_factor"] = round(
            deep_gk / max(light_gk, 0.01), 1)
        result["insight"] = (
            f"Doubling poly_modulus_degree from "
            f"{configs[0]['poly_modulus_degree']} to "
            f"{configs[1]['poly_modulus_degree']} increases "
            f"Galois key size by {result['scaling_factor']}x "
            f"({light_gk:.1f} MB -> {deep_gk:.1f} MB). "
            f"TenSEAL always generates all power-of-2 rotation keys; "
            f"SEAL's C++ API allows selective generation but TenSEAL "
            f"Python does not expose this. You always pay the full cost."
        )

    return result


# ===================================================================
# 4. Noise Budget Estimation
# ===================================================================

def analyze_noise_budget(seed: int = 42) -> dict:
    """Estimate effective noise budget by tracking error growth.

    TenSEAL does not expose SEAL's ``invariant_noise_budget()`` to
    Python.  Instead we estimate the remaining precision by performing
    successive ciphertext-ciphertext multiplications (the most expensive
    operation in terms of noise) and measuring when decryption accuracy
    degrades past clinical tolerance or fails entirely.
    """
    rng = np.random.default_rng(seed)
    results: dict = {"presets": []}

    for preset_name, params in [("light", FHEParams.light()),
                                ("deep", FHEParams.deep())]:
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=params.poly_modulus_degree,
            coeff_mod_bit_sizes=params.coeff_mod_bit_sizes,
        )
        ctx.global_scale = params.global_scale
        ctx.generate_galois_keys()

        # ── Repeated squaring test ──
        # x → x² → x⁴ → x⁸ → ... each step consumes 1 level
        test_val = 0.6  # stay away from 0 and 1 to avoid trivial results
        squaring_steps: list[dict] = []

        enc = ts.ckks_vector(ctx, [test_val])
        expected = test_val

        for depth in range(params.multiplicative_depth + 2):
            if depth == 0:
                # just the initial encryption
                try:
                    dec_val = enc.decrypt()[0]
                    err = abs(dec_val - expected)
                    squaring_steps.append({
                        "depth": 0,
                        "operation": "encrypt",
                        "expected": float(expected),
                        "decrypted": float(dec_val),
                        "abs_error": float(err),
                        "bits_of_precision": float(
                            -np.log2(max(err, 1e-50))),
                        "status": "ok",
                    })
                except Exception as e:
                    squaring_steps.append({
                        "depth": 0, "operation": "encrypt",
                        "status": "error", "message": str(e)[:200],
                    })
                continue

            try:
                enc = enc * enc  # ct × ct → consumes 1 level
                expected = expected * expected
                dec_val = enc.decrypt()[0]
                err = abs(dec_val - expected)
                squaring_steps.append({
                    "depth": depth,
                    "operation": f"square (x^{2**depth})",
                    "expected": float(expected),
                    "decrypted": float(dec_val),
                    "abs_error": float(err),
                    "bits_of_precision": float(
                        -np.log2(max(err, 1e-50))),
                    "status": ("within_tolerance"
                               if err <= CLINICAL_TOLERANCE
                               else "exceeds_tolerance"),
                })
            except Exception as e:
                squaring_steps.append({
                    "depth": depth,
                    "operation": f"square (x^{2**depth})",
                    "status": "FAILED",
                    "error_type": type(e).__name__,
                    "message": str(e)[:200],
                    "note": ("exceeded available multiplicative depth "
                             f"({params.multiplicative_depth})"),
                })
                break

        # ── Pipeline error tracking ──
        values_8 = rng.uniform(0.0, 1.0, 8).tolist()
        w8 = [0.25, 0.15, 0.20, 0.10, 0.15, 0.10, 0.03, 0.02]
        bias = -0.5
        activation = [0.5, 0.197, 0.0, -0.004]
        pipeline_steps: list[dict] = []

        # Step 0: encrypt
        enc_8 = ts.ckks_vector(ctx, values_8)
        dec_8 = enc_8.decrypt()[:8]
        err0 = max(abs(d - v) for d, v in zip(dec_8, values_8))
        pipeline_steps.append({
            "step": "encrypt",
            "depth": 0,
            "max_abs_error": float(err0),
            "bits_of_precision": float(-np.log2(max(err0, 1e-50))),
        })

        # Step 1: dot product + bias
        try:
            enc_dot = enc_8.dot(w8) + bias
            plain_dot = sum(v * w for v, w in zip(values_8, w8)) + bias
            dec_dot = enc_dot.decrypt()[0]
            err1 = abs(dec_dot - plain_dot)
            pipeline_steps.append({
                "step": "dot_product_plus_bias",
                "depth": 1,
                "expected": float(plain_dot),
                "decrypted": float(dec_dot),
                "max_abs_error": float(err1),
                "bits_of_precision": float(-np.log2(max(err1, 1e-50))),
            })
        except Exception as e:
            pipeline_steps.append({
                "step": "dot_product_plus_bias", "depth": 1,
                "error": str(e)[:200],
            })
            enc_dot = None

        # Step 2: polynomial activation
        if enc_dot is not None:
            try:
                enc_poly = enc_dot.polyval(activation)
                plain_poly = sum(c * plain_dot**i
                                 for i, c in enumerate(activation))
                dec_poly = enc_poly.decrypt()[0]
                err2 = abs(dec_poly - plain_poly)
                pipeline_steps.append({
                    "step": "cubic_activation",
                    "depth": 4,
                    "expected": float(plain_poly),
                    "decrypted": float(dec_poly),
                    "max_abs_error": float(err2),
                    "bits_of_precision": float(
                        -np.log2(max(err2, 1e-50))),
                    "levels_remaining": params.multiplicative_depth - 4,
                })
            except Exception as e:
                pipeline_steps.append({
                    "step": "cubic_activation", "depth": 4,
                    "error": str(e)[:200],
                    "note": ("requires depth 4 but preset has "
                             f"depth {params.multiplicative_depth}"),
                })

        # Summarise
        last_ok = [s for s in squaring_steps if s["status"] != "FAILED"]
        max_ok_depth = last_ok[-1]["depth"] if last_ok else 0

        results["presets"].append({
            "preset": preset_name,
            "poly_modulus_degree": params.poly_modulus_degree,
            "multiplicative_depth": params.multiplicative_depth,
            "squaring_test": squaring_steps,
            "pipeline_error_tracking": pipeline_steps,
            "max_depth_before_failure": max_ok_depth,
            "noise_cliff_at_depth": max_ok_depth + 1,
        })

    results["clinical_tolerance"] = CLINICAL_TOLERANCE
    results["methodology"] = (
        "TenSEAL does not expose SEAL's invariant_noise_budget(). "
        "We estimate the effective budget by performing repeated "
        "ciphertext-ciphertext squaring and measuring when decryption "
        "returns errors exceeding the clinical tolerance (0.01) or "
        "when the operation throws an exception due to exhausted "
        "modulus chain."
    )

    return results
