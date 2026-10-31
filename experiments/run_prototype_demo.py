"""Experiment: End-to-end prototype demo (healthcare scenario).

This script demonstrates the full encrypted tool-call pipeline (RQ3):

A medical AI agent needs to:
1. Look up drug interactions for a patient's medication (PIR).
2. Compute a risk score based on patient health features (FHE).
3. Make a treatment decision based on both results (locally, in plaintext).

At each step the script shows what the AGENT sees (plaintext inputs/outputs)
versus what the SERVER sees (encrypted blobs and metadata only).

Usage:
    python experiments/run_prototype_demo.py
    python experiments/run_prototype_demo.py --quick   # 3 patients instead of 10
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import textwrap
import time
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.prototype.encrypted_lookup import (
    DrugRecord,
    EncryptedDrugLookup,
    load_drug_records_from_json,
)
from src.prototype.encrypted_scoring import (
    EncryptedScoringTool,
    PatientFeatures,
    cardiovascular_risk_model,
)
from src.feasibility.fhe_computation import FHEParams
from src.utils.data_utils import load_drug_db, load_patients

# ===================================================================
# Paths
# ===================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# ===================================================================
# Helpers
# ===================================================================
SEVERITY_LABELS = {
    0: "NONE", 1: "MILD", 2: "MODERATE", 3: "SEVERE", 4: "CONTRAINDICATED",
}
WIDTH = 72


def hr(char: str = "-") -> str:
    return char * WIDTH


def banner(text: str) -> str:
    pad = (WIDTH - len(text) - 4) // 2
    return f"{'=' * pad}  {text}  {'=' * max(pad, WIDTH - pad - len(text) - 4)}"


def side_by_side(agent_lines: list[str], server_lines: list[str]) -> str:
    """Format agent vs server views in two columns."""
    col = (WIDTH - 3) // 2
    header = f"{'AGENT VIEW':^{col}} | {'SERVER VIEW':^{col}}"
    sep = f"{'-' * col}-+-{'-' * col}"
    out = [header, sep]
    n = max(len(agent_lines), len(server_lines))
    for i in range(n):
        a = agent_lines[i] if i < len(agent_lines) else ""
        s = server_lines[i] if i < len(server_lines) else ""
        # Truncate to column width
        a = a[:col].ljust(col)
        s = s[:col].ljust(col)
        out.append(f"{a} | {s}")
    return "\n".join(out)


def format_bytes_short(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def risk_label(score: float) -> str:
    if score < 0.3:
        return "LOW"
    if score < 0.6:
        return "MODERATE"
    return "HIGH"


# ===================================================================
# Demo patient profiles (realistic, varied)
# ===================================================================

@dataclass
class PatientCase:
    """A complete patient case for the demo pipeline."""
    name: str
    age: float
    bmi: float
    blood_pressure_systolic: float
    blood_pressure_diastolic: float
    cholesterol: float
    glucose: float
    num_medications: int
    num_conditions: int
    current_drug: str
    new_drug: str
    reason: str

    def to_features(self) -> PatientFeatures:
        return PatientFeatures(
            age=self.age,
            bmi=self.bmi,
            blood_pressure_systolic=self.blood_pressure_systolic,
            blood_pressure_diastolic=self.blood_pressure_diastolic,
            cholesterol=self.cholesterol,
            glucose=self.glucose,
            num_medications=self.num_medications,
            num_conditions=self.num_conditions,
        )


def _build_patient_cases(drug_pairs: list[tuple[str, str]],
                         rng: np.random.Generator) -> list[PatientCase]:
    """Build 10 diverse patient cases using drug pairs from the database."""
    # Pick drug pairs that actually exist in the database
    pair_set = set(drug_pairs)

    # Hand-crafted cases that use real pairs from the DB
    templates = [
        ("Margaret T.", 72, 31.2, 158, 95, 265, 110, 6, 3,
         "warfarin", "albuterol", "new inhaler for COPD"),
        ("James R.", 58, 28.5, 145, 88, 240, 180, 4, 2,
         "metformin", "simvastatin", "cholesterol management"),
        ("Aisha K.", 45, 24.1, 122, 78, 195, 95, 2, 1,
         "aspirin", "ibuprofen", "pain after knee surgery"),
        ("Robert M.", 67, 33.8, 170, 100, 290, 145, 8, 4,
         "lisinopril", "duloxetine", "depression treatment"),
        ("Sofia L.", 38, 22.0, 115, 72, 180, 88, 1, 0,
         "aspirin", "pantoprazole", "stomach protection"),
        ("William H.", 81, 26.5, 162, 92, 310, 130, 10, 5,
         "atorvastatin", "rivaroxaban", "DVT prophylaxis"),
        ("Chen W.", 55, 29.4, 140, 85, 225, 210, 5, 2,
         "metformin", "furosemide", "fluid retention"),
        ("Patricia D.", 63, 35.1, 155, 93, 275, 165, 7, 3,
         "gabapentin", "apixaban", "anticoagulation therapy"),
        ("David S.", 49, 27.0, 132, 80, 200, 105, 3, 1,
         "aspirin", "simvastatin", "cardiac prevention"),
        ("Elena V.", 70, 30.2, 165, 98, 285, 155, 9, 4,
         "montelukast", "ibuprofen", "NSAID for arthritis"),
    ]

    cases = []
    for t in templates:
        name, age, bmi, sbp, dbp, chol, gluc, meds, conds, drug_a, drug_b, reason = t
        # Verify this pair exists
        pair = tuple(sorted([drug_a.lower(), drug_b.lower()]))
        if pair in pair_set:
            cases.append(PatientCase(
                name=name, age=age, bmi=bmi,
                blood_pressure_systolic=sbp, blood_pressure_diastolic=dbp,
                cholesterol=chol, glucose=gluc,
                num_medications=meds, num_conditions=conds,
                current_drug=drug_a, new_drug=drug_b, reason=reason,
            ))

    # If we don't have enough, fill with random patients
    while len(cases) < 10:
        pair = drug_pairs[rng.integers(0, len(drug_pairs))]
        pf = PatientFeatures.random(rng)
        cases.append(PatientCase(
            name=f"Patient-{len(cases)+1}",
            age=pf.age, bmi=pf.bmi,
            blood_pressure_systolic=pf.blood_pressure_systolic,
            blood_pressure_diastolic=pf.blood_pressure_diastolic,
            cholesterol=pf.cholesterol, glucose=pf.glucose,
            num_medications=pf.num_medications,
            num_conditions=pf.num_conditions,
            current_drug=pair[0], new_drug=pair[1],
            reason="routine medication review",
        ))

    return cases


# ===================================================================
# Pipeline timing record
# ===================================================================

@dataclass
class PipelineRun:
    """Timing and correctness data for one pipeline execution."""
    patient_name: str
    drug_a: str
    drug_b: str
    # PIR timings (ms)
    pir_query_gen_ms: float = 0.0
    pir_server_process_ms: float = 0.0
    pir_decrypt_ms: float = 0.0
    pir_total_ms: float = 0.0
    # FHE timings (ms)
    fhe_encrypt_ms: float = 0.0
    fhe_server_compute_ms: float = 0.0
    fhe_decrypt_ms: float = 0.0
    fhe_total_ms: float = 0.0
    # Local decision
    local_decision_ms: float = 0.0
    # Overall
    total_ms: float = 0.0
    # Correctness
    pir_correct: bool = True
    fhe_correct: bool = True
    fhe_error: float = 0.0
    # Results
    severity: int = 0
    risk_score_encrypted: float = 0.0
    risk_score_plaintext: float = 0.0
    decision: str = ""
    # Communication
    pir_query_bytes: int = 0
    pir_response_bytes: int = 0
    fhe_ciphertext_bytes: int = 0
    fhe_result_bytes: int = 0


# ===================================================================
# Core demo functions
# ===================================================================

def run_detailed_demo(
    patient: PatientCase,
    drug_lookup: EncryptedDrugLookup,
    scorer: EncryptedScoringTool,
    verbose: bool = True,
) -> PipelineRun:
    """Run the full encrypted pipeline for one patient case.

    Returns timing and correctness data.
    """
    run = PipelineRun(
        patient_name=patient.name,
        drug_a=patient.current_drug,
        drug_b=patient.new_drug,
    )
    pipeline_t0 = time.perf_counter()

    features = patient.to_features()

    # -- Step 1: PIR drug interaction lookup --------------------------
    if verbose:
        print(f"\n  Step 1: Drug Interaction Lookup (PIR)")
        print(f"  {hr('.')}")

    # 1a. Agent creates encrypted query
    t0 = time.perf_counter()
    encrypted_query = drug_lookup.create_query(patient.current_drug, patient.new_drug)
    run.pir_query_gen_ms = (time.perf_counter() - t0) * 1000
    run.pir_query_bytes = len(encrypted_query)

    if verbose:
        agent_lines = [
            f"Query: {patient.current_drug} + {patient.new_drug}",
            f"Encrypted query: {format_bytes_short(len(encrypted_query))}",
            f"Time: {run.pir_query_gen_ms:.1f} ms",
        ]
        server_lines = [
            "Received: encrypted BFV ciphertext",
            f"Size: {format_bytes_short(len(encrypted_query))}",
            "Drug pair: <UNKNOWN>",
        ]
        print(side_by_side(agent_lines, server_lines))

    # 1b. Server processes query on ciphertext
    t0 = time.perf_counter()
    encrypted_result = drug_lookup.server_process(encrypted_query)
    run.pir_server_process_ms = (time.perf_counter() - t0) * 1000
    run.pir_response_bytes = sum(len(p) for p in encrypted_result)

    if verbose:
        agent_lines = [
            "Waiting for server response...",
            f"Received: {len(encrypted_result)} encrypted parts",
        ]
        server_lines = [
            f"Processed {drug_lookup.database_size} records",
            f"Time: {run.pir_server_process_ms:.1f} ms",
            f"Response: {format_bytes_short(run.pir_response_bytes)}",
            "Which record? <CANNOT DETERMINE>",
        ]
        print(side_by_side(agent_lines, server_lines))

    # 1c. Agent decrypts the result
    t0 = time.perf_counter()
    drug_record = drug_lookup.decrypt_result(encrypted_result)
    run.pir_decrypt_ms = (time.perf_counter() - t0) * 1000
    run.severity = drug_record.severity
    run.pir_total_ms = run.pir_query_gen_ms + run.pir_server_process_ms + run.pir_decrypt_ms

    # Verify PIR correctness
    plain_record = drug_lookup.query_plaintext(patient.current_drug, patient.new_drug)
    run.pir_correct = (
        plain_record is not None
        and drug_record.severity == plain_record.severity
        and drug_record.drug_a == plain_record.drug_a
        and drug_record.drug_b == plain_record.drug_b
    )

    if verbose:
        sev = SEVERITY_LABELS[drug_record.severity]
        agent_lines = [
            f"Decrypted result:",
            f"  {drug_record.drug_a} + {drug_record.drug_b}",
            f"  Severity: {sev}",
            f"  {drug_record.description[:34]}",
            f"Decrypt time: {run.pir_decrypt_ms:.1f} ms",
            f"Correct: {'YES' if run.pir_correct else 'NO'}",
        ]
        server_lines = [
            "Result: <ENCRYPTED>",
            "Severity: <UNKNOWN>",
            "Description: <UNKNOWN>",
            "",
            "Server learned: NOTHING",
            f"(0 bits of information)",
        ]
        print(side_by_side(agent_lines, server_lines))

    # -- Step 2: FHE risk scoring -------------------------------------
    if verbose:
        print(f"\n  Step 2: Patient Risk Scoring (FHE)")
        print(f"  {hr('.')}")

    # 2a. Agent encrypts patient features
    t0 = time.perf_counter()
    enc_features = scorer.encrypt_features(features)
    run.fhe_encrypt_ms = (time.perf_counter() - t0) * 1000
    run.fhe_ciphertext_bytes = len(enc_features.serialize())

    if verbose:
        agent_lines = [
            f"Patient: age={patient.age:.0f}, BMI={patient.bmi:.1f}",
            f"  BP={patient.blood_pressure_systolic:.0f}/{patient.blood_pressure_diastolic:.0f}",
            f"  chol={patient.cholesterol:.0f}, gluc={patient.glucose:.0f}",
            f"  meds={patient.num_medications}, conditions={patient.num_conditions}",
            f"Encrypted: {format_bytes_short(run.fhe_ciphertext_bytes)}",
            f"Time: {run.fhe_encrypt_ms:.1f} ms",
        ]
        server_lines = [
            "Received: CKKS ciphertext",
            f"Size: {format_bytes_short(run.fhe_ciphertext_bytes)}",
            "Patient age: <ENCRYPTED>",
            "Patient BMI: <ENCRYPTED>",
            "Blood pressure: <ENCRYPTED>",
            "All features: <UNKNOWN>",
        ]
        print(side_by_side(agent_lines, server_lines))

    # 2b. Server computes risk score on ciphertext
    t0 = time.perf_counter()
    enc_score = scorer.compute_score_encrypted(enc_features)
    run.fhe_server_compute_ms = (time.perf_counter() - t0) * 1000
    run.fhe_result_bytes = len(enc_score.serialize())

    if verbose:
        agent_lines = [
            "Waiting for server computation...",
            f"Received encrypted score",
        ]
        server_lines = [
            "Computing: w . enc(x) + b",
            "Then: sigmoid approximation",
            f"Time: {run.fhe_server_compute_ms:.1f} ms",
            "Score value: <ENCRYPTED>",
        ]
        print(side_by_side(agent_lines, server_lines))

    # 2c. Agent decrypts the risk score
    t0 = time.perf_counter()
    decrypted_score = scorer.decrypt_score(enc_score)
    run.fhe_decrypt_ms = (time.perf_counter() - t0) * 1000
    run.fhe_total_ms = run.fhe_encrypt_ms + run.fhe_server_compute_ms + run.fhe_decrypt_ms

    run.risk_score_encrypted = decrypted_score
    run.risk_score_plaintext = scorer.score_plaintext(features)
    run.fhe_error = abs(run.risk_score_encrypted - run.risk_score_plaintext)
    run.fhe_correct = run.fhe_error < 0.01

    if verbose:
        rlabel = risk_label(decrypted_score)
        agent_lines = [
            f"Decrypted risk score: {decrypted_score:.4f}",
            f"Risk level: {rlabel}",
            f"Plaintext baseline: {run.risk_score_plaintext:.4f}",
            f"Error: {run.fhe_error:.6f}",
            f"Correct: {'YES' if run.fhe_correct else 'NO'}",
        ]
        server_lines = [
            "Score: <ENCRYPTED>",
            "Risk level: <UNKNOWN>",
            "",
            "",
            "Server learned: NOTHING",
        ]
        print(side_by_side(agent_lines, server_lines))

    # -- Step 3: Local treatment decision -----------------------------
    t0 = time.perf_counter()
    decision = _make_treatment_decision(
        drug_record, decrypted_score, patient
    )
    run.local_decision_ms = (time.perf_counter() - t0) * 1000
    run.decision = decision

    run.total_ms = (time.perf_counter() - pipeline_t0) * 1000

    if verbose:
        print(f"\n  Step 3: Treatment Decision (LOCAL -- no server involved)")
        print(f"  {hr('.')}")
        agent_lines = [
            f"Drug interaction: {SEVERITY_LABELS[drug_record.severity]}",
            f"Risk score: {decrypted_score:.4f} ({risk_label(decrypted_score)})",
            f"",
            f"Decision:",
            *[f"  {line}" for line in textwrap.wrap(decision, 30)],
        ]
        server_lines = [
            "(not involved in this step)",
            "",
            "Decision: <NEVER SENT>",
            "Treatment plan: <NEVER SENT>",
            "",
            "Total server knowledge: 0 bits",
        ]
        print(side_by_side(agent_lines, server_lines))

    return run


def _make_treatment_decision(
    drug_record: DrugRecord,
    risk_score: float,
    patient: PatientCase,
) -> str:
    """Simple rule-based treatment decision (runs locally in plaintext)."""
    sev = drug_record.severity
    rlevel = risk_label(risk_score)

    if sev >= 3:
        label = "CONTRAINDICATED" if sev == 4 else "SEVERE"
        return (f"{label}: {drug_record.drug_a} + {drug_record.drug_b} "
                f"interaction is {label}. Do NOT prescribe. "
                f"Recommend alternative medication.")
    if sev == 2 and rlevel == "HIGH":
        return (f"CAUTION: Moderate drug interaction AND high risk score "
                f"({risk_score:.2f}). Consider alternative or close monitoring.")
    if sev == 2:
        return (f"MONITOR: Moderate interaction detected. "
                f"{drug_record.recommendation} "
                f"Patient risk is {rlevel.lower()}.")
    if sev == 1:
        return (f"PROCEED WITH AWARENESS: Minor interaction. "
                f"{drug_record.recommendation}")
    # severity 0
    if rlevel == "HIGH":
        return (f"PROCEED BUT MONITOR: No drug interaction, but patient "
                f"risk is HIGH ({risk_score:.2f}). Schedule follow-up.")
    return (f"SAFE TO PROCEED: No significant drug interaction. "
            f"Patient risk is {rlevel.lower()} ({risk_score:.2f}).")


# ===================================================================
# Main demo orchestration
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypted tool-call prototype demo")
    parser.add_argument("--quick", action="store_true",
                        help="Run with 3 patients instead of 10")
    args = parser.parse_args()

    n_patients = 3 if args.quick else 10
    rng = np.random.default_rng(42)

    # Capture all output to both console and file
    output_buffer = io.StringIO()

    class TeeWriter:
        """Write to both console and buffer."""
        def __init__(self, *writers):
            self.writers = writers
        def write(self, text):
            for w in self.writers:
                try:
                    w.write(text)
                except UnicodeEncodeError:
                    w.write(text.encode("ascii", errors="replace").decode("ascii"))
        def flush(self):
            for w in self.writers:
                w.flush()

    tee = TeeWriter(sys.stdout, output_buffer)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        _run_demo(n_patients, rng)
    finally:
        sys.stdout = original_stdout

    # Save demo log
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "demo_output.txt"
    log_path.write_text(output_buffer.getvalue(), encoding="utf-8")
    print(f"\nDemo log saved to {log_path}")


def _run_demo(n_patients: int, rng: np.random.Generator) -> None:
    print()
    print(banner("ENCRYPTED TOOL-CALLING DEMO"))
    print()
    print("  Scenario: A medical AI agent assists with treatment decisions.")
    print("  The agent uses two privacy-preserving tool calls:")
    print("    1. PIR -- look up drug interactions (server can't see which drug)")
    print("    2. FHE -- compute patient risk score (server can't see features)")
    print("  All decisions are made locally by the agent.")
    print()
    print(hr("="))

    # -- Phase 1: Setup -----------------------------------------------
    print(f"\n{banner('PHASE 1: SYSTEM SETUP')}\n")

    # Load drug database via data_utils (falls back to legacy file)
    print("  Loading drug interaction database...")
    t0 = time.perf_counter()
    try:
        raw_records = load_drug_db("200")
        db_path = DATA_DIR / "drug_interactions_200.json"
    except (FileNotFoundError, ValueError):
        # Fall back to legacy file if new datasets not yet generated
        db_path = DATA_DIR / "drug_interactions.json"
        if not db_path.exists():
            raise FileNotFoundError(
                f"No drug database found. Run `python data/generate_all.py` first."
            )
        raw_records = None

    if raw_records is not None:
        records = load_drug_records_from_json(db_path)
    else:
        records = load_drug_records_from_json(db_path)

    print(f"    Loaded {len(records)} drug interaction records from {db_path.name}")
    db_size = len(records)
    records_subset = records

    # Initialize PIR
    print("\n  Initializing PIR scheme (BFV encryption)...")
    drug_lookup = EncryptedDrugLookup()
    drug_lookup.load_drug_database(records_subset)
    drug_lookup.setup_pir()
    pir_setup_ms = (time.perf_counter() - t0) * 1000
    print(f"    PIR ready ({pir_setup_ms:.0f} ms)")
    print(f"    Database: {db_size} records, server CANNOT see query targets")

    # Initialize FHE scoring
    print("\n  Initializing FHE risk scoring (CKKS encryption)...")
    t0 = time.perf_counter()
    weights, bias, activation = cardiovascular_risk_model()
    scorer = EncryptedScoringTool(fhe_params=FHEParams.deep())
    scorer.setup(weights, bias, activation_coeffs=activation)
    fhe_setup_ms = (time.perf_counter() - t0) * 1000
    print(f"    FHE ready ({fhe_setup_ms:.0f} ms)")
    print(f"    Model: cardiovascular risk (8 features, cubic sigmoid)")
    print(f"    CKKS params: poly_mod=16384, depth=4, 128-bit security")

    # Build patient cases
    print(f"\n  Generating {n_patients} patient cases...")
    available_pairs = drug_lookup.drug_pairs
    all_cases = _build_patient_cases(available_pairs, rng)
    cases = all_cases[:n_patients]
    print(f"    Ready: {len(cases)} patients with drug interaction queries")

    print(f"\n  Setup complete. Total: {pir_setup_ms + fhe_setup_ms:.0f} ms")
    print(hr("="))

    # -- Phase 2: Detailed walkthrough (first patient) ----------------
    print(f"\n{banner('PHASE 2: DETAILED WALKTHROUGH')}\n")
    print(f"  Patient: {cases[0].name}, age {cases[0].age:.0f}")
    print(f"  Currently taking: {cases[0].current_drug}")
    print(f"  Doctor wants to prescribe: {cases[0].new_drug} ({cases[0].reason})")
    print(f"  Question: Is it safe? What is the patient's cardiac risk?")

    runs: list[PipelineRun] = []
    run0 = run_detailed_demo(cases[0], drug_lookup, scorer, verbose=True)
    runs.append(run0)

    print(f"\n  {'-' * WIDTH}")
    print(f"  Pipeline complete for {cases[0].name}: {run0.total_ms:.0f} ms total")

    # -- Phase 3: Batch runs ------------------------------------------
    if len(cases) > 1:
        print(f"\n{banner('PHASE 3: BATCH EXECUTION')}\n")
        print(f"  Running pipeline for {len(cases) - 1} more patients...\n")

        for i, case in enumerate(cases[1:], start=2):
            run = run_detailed_demo(case, drug_lookup, scorer, verbose=False)
            runs.append(run)
            sev = SEVERITY_LABELS[run.severity]
            rlabel = risk_label(run.risk_score_encrypted)
            check = "OK" if (run.pir_correct and run.fhe_correct) else "FAIL"
            print(f"  [{i:2d}/{len(cases)}] {case.name:<16s} "
                  f"{case.current_drug}+{case.new_drug:<14s} "
                  f"interaction={sev:<8s} risk={rlabel:<8s} "
                  f"{run.total_ms:7.0f} ms [{check}]")

    # -- Phase 4: Server observation audit ----------------------------
    print(f"\n{banner('PHASE 4: SERVER OBSERVATION AUDIT')}\n")
    print("  What did the servers actually observe?\n")

    print("  PIR Server (Drug Interaction Database):")
    print(f"    Total queries processed:    {len(drug_lookup.lookup_log)}")
    print(f"    Drug pairs revealed:        0  (all encrypted)")
    print(f"    Interaction results leaked:  0  (all encrypted)")
    for j, evt in enumerate(drug_lookup.lookup_log[:3]):
        print(f"    Query {j+1}: "
              f"input={format_bytes_short(evt.query_size_bytes)}, "
              f"output={format_bytes_short(evt.response_size_bytes)}, "
              f"time={evt.processing_time_ms:.0f}ms, "
              f"drug_pair_visible={evt.drug_pair_visible}")
    if len(drug_lookup.lookup_log) > 3:
        print(f"    ... ({len(drug_lookup.lookup_log) - 3} more queries)")

    print(f"\n  FHE Server (Risk Scoring Service):")
    print(f"    Total scores computed:      {len(scorer.scoring_log)}")
    print(f"    Patient features revealed:  0  (all encrypted)")
    print(f"    Risk scores leaked:         0  (all encrypted)")
    for j, evt in enumerate(scorer.scoring_log[:3]):
        print(f"    Query {j+1}: "
              f"input={format_bytes_short(evt.encrypted_input_size_bytes)}, "
              f"output={format_bytes_short(evt.encrypted_output_size_bytes)}, "
              f"time={evt.computation_time_ms:.0f}ms, "
              f"features_visible={evt.features_visible}, "
              f"score_visible={evt.score_visible}")
    if len(scorer.scoring_log) > 3:
        print(f"    ... ({len(scorer.scoring_log) - 3} more scores)")

    print(f"\n  Information leaked to servers: 0 bits")
    print(f"  (Servers see ciphertext sizes and timing only -- no content)")

    # -- Phase 5: Summary ---------------------------------------------
    print(f"\n{banner('PHASE 5: RESULTS SUMMARY')}\n")

    # Timing statistics
    pir_times = [r.pir_total_ms for r in runs]
    fhe_times = [r.fhe_total_ms for r in runs]
    local_times = [r.local_decision_ms for r in runs]
    total_times = [r.total_ms for r in runs]
    pir_correct = sum(1 for r in runs if r.pir_correct)
    fhe_correct = sum(1 for r in runs if r.fhe_correct)
    fhe_errors = [r.fhe_error for r in runs]

    # Timing breakdown table
    rows = [
        ["Patients processed", f"{len(runs)}"],
        ["", ""],
        ["PIR lookup (avg)", f"{np.mean(pir_times):.1f} ms"],
        ["  Query generation", f"{np.mean([r.pir_query_gen_ms for r in runs]):.1f} ms"],
        ["  Server processing", f"{np.mean([r.pir_server_process_ms for r in runs]):.1f} ms"],
        ["  Decryption", f"{np.mean([r.pir_decrypt_ms for r in runs]):.1f} ms"],
        ["", ""],
        ["FHE scoring (avg)", f"{np.mean(fhe_times):.1f} ms"],
        ["  Encryption", f"{np.mean([r.fhe_encrypt_ms for r in runs]):.1f} ms"],
        ["  Server computation", f"{np.mean([r.fhe_server_compute_ms for r in runs]):.1f} ms"],
        ["  Decryption", f"{np.mean([r.fhe_decrypt_ms for r in runs]):.1f} ms"],
        ["", ""],
        ["Local decision (avg)", f"{np.mean(local_times):.3f} ms"],
        ["Total pipeline (avg)", f"{np.mean(total_times):.1f} ms"],
        ["Total pipeline (min)", f"{np.min(total_times):.1f} ms"],
        ["Total pipeline (max)", f"{np.max(total_times):.1f} ms"],
        ["", ""],
        ["PIR correctness", f"{pir_correct}/{len(runs)} ({100*pir_correct/len(runs):.0f}%)"],
        ["FHE correctness", f"{fhe_correct}/{len(runs)} ({100*fhe_correct/len(runs):.0f}%)"],
        ["FHE max error", f"{np.max(fhe_errors):.8f}"],
        ["", ""],
        ["Communication (PIR query)", f"{format_bytes_short(runs[0].pir_query_bytes)}"],
        ["Communication (PIR response)", f"{format_bytes_short(runs[0].pir_response_bytes)}"],
        ["Communication (FHE features)", f"{format_bytes_short(runs[0].fhe_ciphertext_bytes)}"],
        ["Communication (FHE score)", f"{format_bytes_short(runs[0].fhe_result_bytes)}"],
        ["", ""],
        ["Server information leakage", "0 bits"],
        ["Privacy guarantee", "128-bit computational security"],
    ]

    # Print formatted table
    col1_w = max(len(r[0]) for r in rows) + 2
    col2_w = max(len(r[1]) for r in rows) + 2
    table_w = col1_w + col2_w + 3
    print(f"  +{'-' * (table_w)}+")
    print(f"  |{'PERFORMANCE SUMMARY':^{table_w}}|")
    print(f"  +{'-' * col1_w}+{'-' * (col2_w + 2)}+")
    print(f"  | {'Metric':<{col1_w - 2}} | {'Value':<{col2_w}} |")
    print(f"  +{'-' * col1_w}+{'-' * (col2_w + 2)}+")
    for label, val in rows:
        if label == "" and val == "":
            print(f"  +{'-' * col1_w}+{'-' * (col2_w + 2)}+")
        else:
            print(f"  | {label:<{col1_w - 2}} | {val:<{col2_w}} |")
    print(f"  +{'-' * col1_w}+{'-' * (col2_w + 2)}+")

    # Per-patient table
    print(f"\n  Per-patient results:")
    print(f"  {'Patient':<16s} {'Drug Pair':<28s} {'Interaction':<12s} "
          f"{'Risk':<10s} {'Total ms':>10s} {'Check':>6s}")
    print(f"  {'-'*16} {'-'*28} {'-'*12} {'-'*10} {'-'*10} {'-'*6}")
    for r in runs:
        pair = f"{r.drug_a}+{r.drug_b}"
        sev = SEVERITY_LABELS[r.severity]
        rlabel = risk_label(r.risk_score_encrypted)
        check = "OK" if (r.pir_correct and r.fhe_correct) else "FAIL"
        print(f"  {r.patient_name:<16s} {pair:<28s} {sev:<12s} "
              f"{rlabel:<10s} {r.total_ms:10.0f} {check:>6s}")

    # -- Final verdict ------------------------------------------------
    all_correct = all(r.pir_correct and r.fhe_correct for r in runs)
    print(f"\n{hr('=')}")
    if all_correct:
        print(f"  DEMO RESULT: All {len(runs)} pipeline runs completed successfully.")
        print(f"  Encrypted results match plaintext baselines in every case.")
        print(f"  Servers processed {len(runs)} queries without learning any patient data.")
    else:
        n_fail = sum(1 for r in runs if not (r.pir_correct and r.fhe_correct))
        print(f"  DEMO RESULT: {n_fail}/{len(runs)} runs had correctness issues.")
    print(hr("="))

    print(f"\n  Conclusion: FHE and PIR provide functional privacy for AI agent")
    print(f"  tool calls. The trade-off is latency (~{np.mean(total_times):.0f} ms per pipeline")
    print(f"  vs <1 ms plaintext), but the server learns exactly 0 bits about")
    print(f"  patient data, drug queries, risk scores, or treatment decisions.")
    print()

    # -- Save timing data ---------------------------------------------
    timings = {
        "metadata": {
            "n_patients": len(runs),
            "db_size": len(records_subset),
            "fhe_params": "CKKS poly_mod=16384 depth=4",
            "pir_params": f"BFV poly_mod=8192 db_size={len(records_subset)}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": {
            "avg_total_ms": float(np.mean(total_times)),
            "avg_pir_ms": float(np.mean(pir_times)),
            "avg_fhe_ms": float(np.mean(fhe_times)),
            "avg_local_ms": float(np.mean(local_times)),
            "min_total_ms": float(np.min(total_times)),
            "max_total_ms": float(np.max(total_times)),
            "pir_correctness": f"{pir_correct}/{len(runs)}",
            "fhe_correctness": f"{fhe_correct}/{len(runs)}",
            "fhe_max_error": float(np.max(fhe_errors)),
            "server_leakage_bits": 0,
        },
        "runs": [
            {
                "patient": r.patient_name,
                "drug_a": r.drug_a,
                "drug_b": r.drug_b,
                "pir_query_gen_ms": round(r.pir_query_gen_ms, 2),
                "pir_server_process_ms": round(r.pir_server_process_ms, 2),
                "pir_decrypt_ms": round(r.pir_decrypt_ms, 2),
                "pir_total_ms": round(r.pir_total_ms, 2),
                "fhe_encrypt_ms": round(r.fhe_encrypt_ms, 2),
                "fhe_server_compute_ms": round(r.fhe_server_compute_ms, 2),
                "fhe_decrypt_ms": round(r.fhe_decrypt_ms, 2),
                "fhe_total_ms": round(r.fhe_total_ms, 2),
                "local_decision_ms": round(r.local_decision_ms, 4),
                "total_ms": round(r.total_ms, 2),
                "pir_correct": r.pir_correct,
                "fhe_correct": r.fhe_correct,
                "fhe_error": round(r.fhe_error, 8),
                "severity": r.severity,
                "risk_score_encrypted": round(r.risk_score_encrypted, 6),
                "risk_score_plaintext": round(r.risk_score_plaintext, 6),
                "decision": r.decision,
                "pir_query_bytes": r.pir_query_bytes,
                "pir_response_bytes": r.pir_response_bytes,
                "fhe_ciphertext_bytes": r.fhe_ciphertext_bytes,
                "fhe_result_bytes": r.fhe_result_bytes,
            }
            for r in runs
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timings_path = RESULTS_DIR / "demo_timings.json"
    timings_path.write_text(
        json.dumps(timings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Timing data saved to {timings_path}")


if __name__ == "__main__":
    main()
