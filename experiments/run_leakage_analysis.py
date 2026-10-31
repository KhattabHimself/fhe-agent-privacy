"""Experiment: Measure information leakage from tool-call sequences.

This script runs the threat-model analysis (RQ1):
1. Generate 100 synthetic medical-agent tool-call sessions.
2. Compute leakage metrics (direct, correlation, frequency).
3. Run intent-reconstruction attacks (frequency, sequence, ML).
4. Compare plaintext vs. encrypted reconstruction accuracy.
5. Output results to results/leakage_analysis/.

Usage:
    python experiments/run_leakage_analysis.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

# Ensure UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.threat_model.definitions import (
    ThreatModel,
    ToolCall,
    ToolCallSequence,
    ToolCallType,
)
from src.threat_model.leakage_analysis import LeakageAnalyzer
from src.threat_model.intent_reconstruction import (
    FrequencyAttack,
    MLClassifierAttack,
    SequenceAttack,
    compare_reconstruction_accuracy,
    make_encrypted_session,
)
from src.utils.data_utils import _build_drug_pool

RESULTS_DIR = Path("results/leakage_analysis")

# ── Synthetic data generation ─────────────────────────────────────

# Use the first 25 drugs from the shared pool for session generation.
# This keeps sessions diverse while drawing from the same pharmacopeia
# used by the 10K drug interaction database.
DRUGS = _build_drug_pool()[:25]

INTENTS = [
    "check_drug_interactions",
    "assess_cardiac_risk",
    "review_medication_safety",
    "evaluate_treatment_plan",
]


def _mc(tool: str, method: str, params: dict, t: float, sid: str,
        ct: ToolCallType = ToolCallType.DATA_RETRIEVAL) -> ToolCall:
    return ToolCall(tool_name=tool, method=method, parameters=params,
                    timestamp=t, session_id=sid, call_type=ct)


def generate_session(
    intent: str, sid: str, rng: np.random.Generator,
) -> ToolCallSequence:
    """Generate a realistic session for a given intent."""
    t = 0.0
    calls: list[ToolCall] = []

    if intent == "check_drug_interactions":
        d1, d2 = rng.choice(DRUGS, 2, replace=False).tolist()
        calls.append(_mc("drug_db", "lookup", {"drug": d1}, t, sid))
        t += rng.exponential(0.5)
        calls.append(_mc("drug_db", "lookup", {"drug": d2}, t, sid))
        t += rng.exponential(0.3)
        calls.append(_mc("drug_db", "check_interaction",
                         {"drug_a": d1, "drug_b": d2}, t, sid))
        if rng.random() > 0.4:
            t += rng.exponential(0.4)
            calls.append(_mc("drug_db", "get_alternatives",
                             {"drug": d1}, t, sid))

    elif intent == "assess_cardiac_risk":
        age = int(rng.integers(40, 80))
        bp = int(rng.integers(110, 180))
        chol = int(rng.integers(150, 300))
        calls.append(_mc("risk_scorer", "score",
                         {"age": age, "bp": bp, "cholesterol": chol},
                         t, sid, ToolCallType.COMPUTATION))
        t += rng.exponential(1.0)
        calls.append(_mc("risk_scorer", "get_risk_factors",
                         {"age": age}, t, sid, ToolCallType.COMPUTATION))
        if rng.random() > 0.3:
            t += rng.exponential(0.5)
            calls.append(_mc("guideline_db", "lookup",
                             {"condition": "cardiac"}, t, sid))
        if rng.random() > 0.6:
            t += rng.exponential(0.3)
            calls.append(_mc("symptom_checker", "check",
                             {"symptoms": ["chest_pain", "shortness_of_breath"]},
                             t, sid))

    elif intent == "review_medication_safety":
        n_meds = int(rng.integers(2, 6))
        meds = rng.choice(DRUGS, n_meds, replace=False).tolist()
        for med in meds:
            calls.append(_mc("drug_db", "lookup", {"drug": med}, t, sid))
            t += rng.exponential(0.3)
        calls.append(_mc("drug_db", "check_contraindications",
                         {"medications": meds}, t, sid))
        if rng.random() > 0.5:
            t += rng.exponential(0.4)
            calls.append(_mc("drug_db", "check_interaction",
                             {"drug_a": meds[0], "drug_b": meds[1]}, t, sid))

    elif intent == "evaluate_treatment_plan":
        d = rng.choice(DRUGS).item()
        calls.append(_mc("drug_db", "lookup", {"drug": d}, t, sid))
        t += rng.exponential(0.5)
        calls.append(_mc("risk_scorer", "score",
                         {"age": int(rng.integers(30, 75)),
                          "bp": int(rng.integers(100, 170)),
                          "cholesterol": int(rng.integers(140, 280))},
                         t, sid, ToolCallType.COMPUTATION))
        t += rng.exponential(0.4)
        calls.append(_mc("guideline_db", "lookup",
                         {"condition": "general"}, t, sid))
        t += rng.exponential(0.3)
        calls.append(_mc("drug_db", "get_dosage",
                         {"drug": d, "weight_kg": int(rng.integers(50, 100))},
                         t, sid))

    return ToolCallSequence(session_id=sid, calls=calls, intent=intent)


def generate_dataset(
    n_sessions: int = 100, seed: int = 42,
) -> tuple[list[ToolCallSequence], list[str]]:
    """Generate n_sessions balanced across intents."""
    rng = np.random.default_rng(seed)
    sessions: list[ToolCallSequence] = []
    intents: list[str] = []
    per_intent = n_sessions // len(INTENTS)
    for intent in INTENTS:
        for j in range(per_intent):
            sid = f"{intent[:8]}-{j:03d}"
            sessions.append(generate_session(intent, sid, rng))
            intents.append(intent)
    return sessions, intents


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  RQ1: Tool-Call Information Leakage Analysis")
    print("=" * 64)

    # ── Step 1: Generate data ─────────────────────────────────────
    sessions, intents = generate_dataset(n_sessions=100, seed=42)
    total_calls = sum(s.call_count for s in sessions)
    print(f"\nGenerated {len(sessions)} sessions with {total_calls} total tool calls")
    for intent in INTENTS:
        n = intents.count(intent)
        print(f"  {intent}: {n} sessions")

    # ── Step 2: Leakage analysis ──────────────────────────────────
    print("\n--- Leakage Metrics ---")
    analyzer = LeakageAnalyzer(threat_model=ThreatModel())
    analyzer.add_sessions(sessions)
    report = analyzer.generate_report()

    # Print summary
    rows = [
        ["Total sessions", report["summary"]["total_sessions"]],
        ["Total calls", report["summary"]["total_calls"]],
        ["Intent entropy (bits)", f'{report["summary"]["leakage_bits"]:.3f}'],
        ["Composite leakage score", f'{report["summary"]["total_leakage_score"]:.3f}'],
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))

    # Direct leakage
    print("\n  Direct Leakage:")
    for k, v in report["direct"].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Correlation leakage
    print("\n  Correlation Leakage:")
    for k, v in report["correlation"].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Frequency leakage
    print("\n  Frequency Leakage:")
    for k, v in report["frequency"].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Save report
    report_path = RESULTS_DIR / "leakage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {report_path}")

    # ── Step 3: Protected vs unprotected comparison ───────────────
    print("\n--- Protected vs. Unprotected Comparison ---")
    encrypted_sessions = [make_encrypted_session(s) for s in sessions]
    comparison = analyzer.compare_protected_vs_unprotected(encrypted_sessions)

    comp_rows = []
    for label, m in comparison.items():
        comp_rows.append([
            label,
            f"{m.direct.content_entropy:.3f}",
            f"{m.correlation.parameter_overlap_rate:.3f}",
            f"{m.frequency.query_distribution_entropy:.3f}",
            f"{m.total_leakage_score:.3f}",
        ])
    print(tabulate(
        comp_rows,
        headers=["Mode", "Content Entropy", "Param Overlap", "Freq Entropy", "Score"],
        tablefmt="simple",
    ))

    # ── Step 4: Intent-reconstruction attacks ─────────────────────
    print("\n--- Intent Reconstruction Attacks ---")
    n = len(sessions)
    rng = np.random.default_rng(99)
    idx = rng.permutation(n)
    split = int(0.7 * n)
    train_s = [sessions[i] for i in idx[:split]]
    train_i = [intents[i] for i in idx[:split]]
    test_s = [sessions[i] for i in idx[split:]]
    test_i = [intents[i] for i in idx[split:]]

    results = compare_reconstruction_accuracy(train_s, train_i, test_s, test_i)

    attack_rows = []
    for mode in ["plaintext", "encrypted"]:
        for attack_name in ["FrequencyAttack", "SequenceAttack", "MLClassifierAttack"]:
            m = results[mode][attack_name]
            attack_rows.append([
                mode, attack_name,
                f'{m["accuracy"]:.3f}',
                f'{m["top_3_accuracy"]:.3f}',
                f'{m["precision_macro"]:.3f}',
                f'{m["recall_macro"]:.3f}',
            ])
    print(tabulate(
        attack_rows,
        headers=["Mode", "Attack", "Accuracy", "Top-3 Acc", "Precision", "Recall"],
        tablefmt="simple",
    ))

    # Save reconstruction results
    recon_path = RESULTS_DIR / "reconstruction_results.json"
    with open(recon_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {recon_path}")

    # ── Step 5: Per-intent accuracy breakdown ─────────────────────
    print("\n--- Per-Intent ML Accuracy (Plaintext) ---")
    ml_attack = MLClassifierAttack()
    ml_attack.fit(train_s, train_i)
    intent_correct: dict[str, int] = {i: 0 for i in INTENTS}
    intent_total: dict[str, int] = {i: 0 for i in INTENTS}
    for sess, true_intent in zip(test_s, test_i):
        pred = ml_attack.predict(sess)
        intent_total[true_intent] += 1
        if pred.predicted_intent == true_intent:
            intent_correct[true_intent] += 1

    intent_rows = []
    for intent in INTENTS:
        total = intent_total[intent]
        correct = intent_correct[intent]
        acc = correct / total if total > 0 else 0.0
        intent_rows.append([intent, total, correct, f"{acc:.3f}"])
    print(tabulate(
        intent_rows,
        headers=["Intent", "Total", "Correct", "Accuracy"],
        tablefmt="simple",
    ))

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    plain_ml = results["plaintext"]["MLClassifierAttack"]["accuracy"]
    enc_ml = results["encrypted"]["MLClassifierAttack"]["accuracy"]
    reduction = plain_ml - enc_ml
    print(f"  Plaintext ML accuracy:  {plain_ml:.1%}")
    print(f"  Encrypted ML accuracy:  {enc_ml:.1%}")
    print(f"  Privacy gain:           {reduction:.1%} accuracy reduction")
    print(f"\n  Results saved to {RESULTS_DIR}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
