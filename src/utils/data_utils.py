"""Data loading and synthetic data generation utilities.

Provides functions to:
- Generate synthetic drug-interaction databases (200 and 10K records)
- Generate synthetic patient profiles (10 and 1000 records)
- Load datasets by size tag (``load_drug_db("10k")``, ``load_patients("1000")``)
- Print summary statistics for any generated dataset
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ===================================================================
# Project root (two levels up from src/utils/)
# ===================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ===================================================================
# Drug-name generation: combine real prefixes with real suffixes
# to produce a large, realistic-sounding pharmacopeia.
# ===================================================================

# Prefixes drawn from real drug stems (INN naming conventions)
_PREFIXES = [
    "Ator",  "Metfor",  "Amoxi",  "Lisi",   "Omepra",
    "Rosuva", "Cipro",  "Fluox",  "Gaba",   "Sertra",
    "Clopi",  "Losart", "Metopro","Amlo",   "Simva",
    "Escita", "Tramad", "Furo",   "Panto",  "Predni",
    "Napro",  "Ibupro", "Valsart","Rami",   "Enal",
    "Carve",  "Dilti",  "Vera",   "Doxo",   "Azithro",
    "Cefalo", "Levoflo","Duloxe", "Venla",  "Mirtaz",
    "Quetiap","Olanz",  "Rispe",  "Aripipr","Lanso",
    "Rabe",   "Dabi",   "Riva",   "Apixa",  "Warfa",
    "Edoxa",  "Tamsulo","Finas",  "Duta",   "Montel",
]

# Suffixes drawn from real pharmacological class stems
_SUFFIXES = [
    "statin",  "pril",   "olol",   "mab",    "zole",
    "mycin",   "sartan", "pine",   "oxetine","cillin",
    "dipine",  "semide", "azepam", "amine",  "dronate",
    "gliptin", "glutide","lukast", "fenac",  "afil",
    "rinol",   "triptan","vastatin","xaban",  "etine",
    "vudine",  "asone",  "tidine", "phylline","nazole",
]

# Pre-built list of 150 unique drug names (more than enough for 10K pairs)
_DRUG_POOL: list[str] = []


def _build_drug_pool() -> list[str]:
    """Generate deterministic pool of unique drug names from prefix+suffix."""
    global _DRUG_POOL
    if _DRUG_POOL:
        return _DRUG_POOL

    seen: set[str] = set()
    names: list[str] = []
    rng = random.Random(0)  # deterministic ordering

    # Combine every prefix with every suffix, then shuffle deterministically
    combos = [(p, s) for p in _PREFIXES for s in _SUFFIXES]
    rng.shuffle(combos)

    for prefix, suffix in combos:
        candidate = (prefix + suffix).lower()
        if candidate not in seen and len(candidate) >= 6:
            seen.add(candidate)
            names.append(candidate)
        if len(names) >= 150:
            break

    _DRUG_POOL = names
    return _DRUG_POOL


# ===================================================================
# Severity definitions (5 levels including contraindicated)
# ===================================================================

SEVERITY_LEVELS = ["none", "mild", "moderate", "severe", "contraindicated"]
SEVERITY_WEIGHTS = [0.40, 0.25, 0.20, 0.10, 0.05]
SEVERITY_TO_INT = {s: i for i, s in enumerate(SEVERITY_LEVELS)}

_MECHANISM_TEMPLATES = {
    "none": [
        "No documented pharmacological interaction.",
        "No clinically significant interaction expected.",
        "Independent metabolic pathways; no interaction.",
    ],
    "mild": [
        "Minor CYP3A4 competition may slightly alter plasma levels.",
        "Weak protein-binding displacement; usually not clinically significant.",
        "Additive mild CNS effects possible at high doses.",
    ],
    "moderate": [
        "CYP2D6 inhibition may increase plasma concentration of the co-administered drug.",
        "Moderate risk of QT prolongation with concurrent use.",
        "Potential for increased bleeding risk via platelet aggregation inhibition.",
        "Concurrent use may reduce renal clearance and elevate serum levels.",
    ],
    "severe": [
        "Strong CYP3A4 inhibition leading to dangerous plasma level elevation.",
        "High risk of serotonin syndrome with combined serotonergic agents.",
        "Synergistic hypotension risk; may cause hemodynamic collapse.",
        "Major bleeding risk due to combined anticoagulant and antiplatelet activity.",
    ],
    "contraindicated": [
        "Absolute contraindication: combined use may cause fatal cardiac arrhythmia.",
        "Contraindicated: irreversible hepatotoxicity reported in post-marketing surveillance.",
        "Contraindicated: synergistic nephrotoxicity with documented cases of acute renal failure.",
    ],
}

_RECOMMENDATION_TEMPLATES = {
    "none": "Standard monitoring. No dosage adjustment required.",
    "mild": "Monitor for side effects. Consider spacing administration times.",
    "moderate": "Assess risk-benefit before prescribing. Consider dose adjustment or alternatives.",
    "severe": "Avoid combination if possible. If essential, use lowest effective doses with close monitoring.",
    "contraindicated": "Do NOT co-prescribe. Use alternative therapy.",
}

# ===================================================================
# User intents for synthetic session generation
# ===================================================================

USER_INTENTS = [
    "check_drug_interactions",
    "assess_cardiac_risk",
    "review_medication_safety",
    "evaluate_treatment_plan",
    "screen_contraindications",
]


# ===================================================================
# Drug-interaction database generation
# ===================================================================

def generate_drug_db(
    num_records: int = 10_000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate a synthetic drug-interaction database.

    Each record is designed to serialize to <= 256 bytes (JSON compact
    form) so that it fits in a single BFV PIR slot chunk.  256 bytes is
    a power of 2 chosen for BFV slot alignment: the BFV plaintext
    modulus packs 2 bytes per slot, and powers-of-2 record sizes avoid
    wasted padding in the ciphertext packing.

    Parameters
    ----------
    num_records : int
        Target number of drug-pair interaction records.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Records with keys: drug_id, drug_a, drug_b, severity,
        mechanism, clinical_recommendation.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    drugs = _build_drug_pool()

    # We need C(n_drugs, 2) >= num_records.
    # C(n, 2) = n*(n-1)/2.  Solve: n >= (1 + sqrt(1 + 8*num_records)) / 2
    import math
    n_drugs_needed = math.ceil((1 + math.sqrt(1 + 8 * num_records)) / 2)
    n_drugs = min(n_drugs_needed, len(drugs))
    selected_drugs = sorted(drugs[:n_drugs])

    # Generate all unique pairs (alphabetically sorted)
    all_pairs = list(itertools.combinations(selected_drugs, 2))
    py_rng.shuffle(all_pairs)

    # Take exactly num_records pairs (or all if fewer available)
    pairs = all_pairs[:num_records]

    # Assign severities with the requested distribution
    severities = rng.choice(
        SEVERITY_LEVELS,
        size=len(pairs),
        p=SEVERITY_WEIGHTS,
    )

    records: list[dict[str, Any]] = []
    for i, ((drug_a, drug_b), sev) in enumerate(zip(pairs, severities)):
        mechanism = py_rng.choice(_MECHANISM_TEMPLATES[sev])
        recommendation = _RECOMMENDATION_TEMPLATES[sev]

        # Truncate mechanism/recommendation so the compact JSON serialization
        # of each record fits within 256 bytes.  This is a power of 2 chosen
        # for BFV slot alignment: the BFV plaintext modulus packs 2 bytes per
        # slot, and powers-of-2 record sizes avoid wasted padding.
        #
        # Budget: 256 bytes total.  Fixed overhead (keys, braces, quotes,
        # colons, commas, drug_id digits) ≈ 100 bytes + 2 × drug-name length.
        # Longest drug names are ~20 chars, so overhead ≈ 140 bytes.
        # That leaves ~116 bytes for mechanism + recommendation.
        max_mech_len = 68
        max_rec_len = 48
        mechanism = mechanism[:max_mech_len]
        recommendation = recommendation[:max_rec_len]

        record = {
            "drug_id": i,
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": sev,
            "mechanism": mechanism,
            "clinical_recommendation": recommendation,
        }

        # Verify the 256-byte constraint; trim further if needed
        compact = json.dumps(record, separators=(",", ":")).encode("utf-8")
        while len(compact) > 256 and max_mech_len > 20:
            max_mech_len -= 10
            record["mechanism"] = mechanism[:max_mech_len]
            compact = json.dumps(record, separators=(",", ":")).encode("utf-8")

        records.append(record)

    return records


def _save_drug_db(records: list[dict[str, Any]], path: Path) -> Path:
    """Save drug-interaction records to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def generate_drug_interactions(
    output_path: str | Path | None = None,
    num_records: int = 1000,
    seed: int = 42,
) -> Path:
    """Backward-compatible entry point used by ``make data``.

    Generates records in the new schema and saves them.
    """
    if output_path is None:
        output_path = DATA_DIR / "drug_interactions.json"
    output_path = Path(output_path)
    records = generate_drug_db(num_records=num_records, seed=seed)
    _save_drug_db(records, output_path)
    print(f"Generated {len(records)} drug interaction records -> {output_path}")
    return output_path


# ===================================================================
# Patient profile generation
# ===================================================================

def generate_patients(
    num_patients: int = 1000,
    seed: int = 42,
    include_edge_cases: bool = True,
) -> list[dict[str, Any]]:
    """Generate synthetic patient profiles with clinically plausible values.

    Parameters
    ----------
    num_patients : int
        Total patients to generate.
    seed : int
        Random seed for reproducibility.
    include_edge_cases : bool
        If True, the last 50 patients (or 5% if num_patients < 1000)
        are given extreme clinical values for stress-testing.

    Returns
    -------
    list[dict]
        Patient records with keys: patient_id, age, bmi, systolic_bp,
        diastolic_bp, cholesterol_total, cholesterol_hdl, glucose_fasting,
        num_medications, num_conditions, current_drugs.
    """
    rng = np.random.default_rng(seed)
    drugs = _build_drug_pool()

    n_edge = min(50, max(1, num_patients // 20)) if include_edge_cases else 0
    n_normal = num_patients - n_edge

    patients: list[dict[str, Any]] = []

    # --- Normal patients ---
    for i in range(n_normal):
        age = int(np.clip(rng.normal(55, 15), 18, 95))
        bmi = round(float(np.clip(rng.normal(27, 5), 15, 50)), 1)
        sbp = int(np.clip(rng.normal(130, 20), 80, 220))
        dbp = int(np.clip(rng.normal(80, 12), 50, 130))
        chol_total = int(np.clip(rng.normal(200, 40), 100, 350))
        chol_hdl = int(np.clip(rng.normal(55, 15), 20, 100))
        glucose = int(np.clip(rng.normal(100, 25), 60, 300))
        n_meds = int(rng.integers(0, 12))
        n_conds = int(rng.integers(0, 8))
        n_drugs = int(rng.integers(1, 6))  # 1-5 current drugs
        current_drugs = rng.choice(drugs, size=min(n_drugs, len(drugs)),
                                   replace=False).tolist()

        patients.append({
            "patient_id": f"P{i:04d}",
            "age": age,
            "bmi": bmi,
            "systolic_bp": sbp,
            "diastolic_bp": dbp,
            "cholesterol_total": chol_total,
            "cholesterol_hdl": chol_hdl,
            "glucose_fasting": glucose,
            "num_medications": n_meds,
            "num_conditions": n_conds,
            "current_drugs": current_drugs,
        })

    # --- Edge-case patients (extreme values) ---
    edge_profiles = [
        # Very elderly with many comorbidities
        {"age": 94, "bmi": 19.5, "systolic_bp": 195, "diastolic_bp": 55,
         "cholesterol_total": 340, "cholesterol_hdl": 28, "glucose_fasting": 280,
         "num_medications": 14, "num_conditions": 7},
        # Young, obese, metabolic syndrome
        {"age": 22, "bmi": 48.5, "systolic_bp": 155, "diastolic_bp": 100,
         "cholesterol_total": 310, "cholesterol_hdl": 25, "glucose_fasting": 295,
         "num_medications": 5, "num_conditions": 4},
        # Hypertensive crisis
        {"age": 60, "bmi": 32.0, "systolic_bp": 218, "diastolic_bp": 128,
         "cholesterol_total": 270, "cholesterol_hdl": 35, "glucose_fasting": 180,
         "num_medications": 8, "num_conditions": 5},
        # Healthy minimum
        {"age": 18, "bmi": 21.0, "systolic_bp": 110, "diastolic_bp": 70,
         "cholesterol_total": 155, "cholesterol_hdl": 75, "glucose_fasting": 78,
         "num_medications": 0, "num_conditions": 0},
        # Hypoglycemic
        {"age": 45, "bmi": 24.0, "systolic_bp": 120, "diastolic_bp": 75,
         "cholesterol_total": 190, "cholesterol_hdl": 60, "glucose_fasting": 62,
         "num_medications": 3, "num_conditions": 1},
    ]

    for j in range(n_edge):
        template = edge_profiles[j % len(edge_profiles)].copy()
        # Add slight randomization to edge cases so they're not all identical
        if j >= len(edge_profiles):
            template["age"] = int(np.clip(template["age"] + rng.integers(-3, 4), 18, 95))
            template["bmi"] = round(float(np.clip(
                template["bmi"] + rng.normal(0, 1), 15, 50)), 1)
            template["systolic_bp"] = int(np.clip(
                template["systolic_bp"] + rng.integers(-5, 6), 80, 220))

        n_drugs = int(rng.integers(1, 6))
        current_drugs = rng.choice(drugs, size=min(n_drugs, len(drugs)),
                                   replace=False).tolist()

        patients.append({
            "patient_id": f"P{n_normal + j:04d}",
            "age": template["age"],
            "bmi": template["bmi"],
            "systolic_bp": template["systolic_bp"],
            "diastolic_bp": template["diastolic_bp"],
            "cholesterol_total": template["cholesterol_total"],
            "cholesterol_hdl": template["cholesterol_hdl"],
            "glucose_fasting": template["glucose_fasting"],
            "num_medications": template["num_medications"],
            "num_conditions": template["num_conditions"],
            "current_drugs": current_drugs,
        })

    return patients


def _save_patients(patients: list[dict[str, Any]], path: Path) -> Path:
    """Save patient profiles to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(patients, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ===================================================================
# Unified loaders: load_drug_db() and load_patients()
# ===================================================================

_DRUG_DB_SIZES = {
    "200":  "drug_interactions_200.json",
    "10k":  "drug_interactions_10k.json",
    "10000":"drug_interactions_10k.json",
}

_PATIENT_SIZES = {
    "10":   "synthetic_patients_10.json",
    "1000": "synthetic_patients_1000.json",
}


def load_drug_db(
    size: str = "10k",
    data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load a drug-interaction database by size tag.

    Parameters
    ----------
    size : str
        One of ``"200"`` (quick testing) or ``"10k"`` / ``"10000"``
        (full evaluation).
    data_dir : Path, optional
        Override the default ``data/`` directory.

    Returns
    -------
    list[dict]
        Drug interaction records.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.  Run ``data/generate_all.py``
        to create it.
    ValueError
        If *size* is not a recognized tag.
    """
    ddir = data_dir or DATA_DIR
    fname = _DRUG_DB_SIZES.get(size)
    if fname is None:
        raise ValueError(
            f"Unknown drug-db size '{size}'. Choose from: {list(_DRUG_DB_SIZES)}"
        )
    path = ddir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_all.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_patients(
    size: str = "1000",
    data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load synthetic patient profiles by size tag.

    Parameters
    ----------
    size : str
        One of ``"10"`` (quick testing) or ``"1000"`` (full evaluation).
    data_dir : Path, optional
        Override the default ``data/`` directory.

    Returns
    -------
    list[dict]
        Patient profile records.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If *size* is not a recognized tag.
    """
    ddir = data_dir or DATA_DIR
    fname = _PATIENT_SIZES.get(size)
    if fname is None:
        raise ValueError(
            f"Unknown patient size '{size}'. Choose from: {list(_PATIENT_SIZES)}"
        )
    path = ddir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_all.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ===================================================================
# Dataset statistics
# ===================================================================

def get_dataset_stats(
    drug_records: list[dict[str, Any]] | None = None,
    patient_records: list[dict[str, Any]] | None = None,
) -> str:
    """Print and return summary statistics for the datasets.

    Parameters
    ----------
    drug_records : list[dict], optional
        Drug-interaction records.  If None, tries to load the 10k set.
    patient_records : list[dict], optional
        Patient records.  If None, tries to load the 1000 set.

    Returns
    -------
    str
        Formatted statistics string.
    """
    lines: list[str] = []

    # --- Drug DB stats ---
    if drug_records is None:
        try:
            drug_records = load_drug_db("10k")
        except FileNotFoundError:
            drug_records = []

    if drug_records:
        lines.append("=" * 60)
        lines.append("  Drug Interaction Database")
        lines.append("=" * 60)
        lines.append(f"  Record count: {len(drug_records)}")

        # Severity distribution
        from collections import Counter
        sev_counts = Counter(r["severity"] for r in drug_records)
        lines.append("  Severity distribution:")
        for sev in SEVERITY_LEVELS:
            count = sev_counts.get(sev, 0)
            pct = 100 * count / len(drug_records) if drug_records else 0
            lines.append(f"    {sev:<16s}: {count:>5d}  ({pct:5.1f}%)")

        # Average record size (compact JSON)
        sizes = [
            len(json.dumps(r, separators=(",", ":")).encode("utf-8"))
            for r in drug_records
        ]
        lines.append(f"  Avg record size (JSON bytes): {np.mean(sizes):.0f}")
        lines.append(f"  Max record size (JSON bytes): {max(sizes)}")
        lines.append(f"  Records <= 256 bytes: {sum(1 for s in sizes if s <= 256)}"
                      f" / {len(sizes)}")

        # Unique drugs
        all_drugs = set()
        for r in drug_records:
            all_drugs.add(r["drug_a"])
            all_drugs.add(r["drug_b"])
        lines.append(f"  Unique drugs: {len(all_drugs)}")

    # --- Patient stats ---
    if patient_records is None:
        try:
            patient_records = load_patients("1000")
        except FileNotFoundError:
            patient_records = []

    if patient_records:
        lines.append("")
        lines.append("=" * 60)
        lines.append("  Patient Profiles")
        lines.append("=" * 60)
        lines.append(f"  Patient count: {len(patient_records)}")

        numeric_fields = [
            "age", "bmi", "systolic_bp", "diastolic_bp",
            "cholesterol_total", "cholesterol_hdl", "glucose_fasting",
            "num_medications", "num_conditions",
        ]
        lines.append(f"  {'Feature':<22s} {'Mean':>8s} {'Std':>8s} "
                      f"{'Min':>8s} {'Max':>8s}")
        lines.append(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for field in numeric_fields:
            vals = np.array([p[field] for p in patient_records], dtype=float)
            lines.append(
                f"  {field:<22s} {np.mean(vals):8.1f} {np.std(vals):8.1f} "
                f"{np.min(vals):8.1f} {np.max(vals):8.1f}"
            )

        # Current drugs stats
        n_drugs_per_patient = [len(p["current_drugs"]) for p in patient_records]
        lines.append(f"  Drugs per patient: mean={np.mean(n_drugs_per_patient):.1f}, "
                      f"min={min(n_drugs_per_patient)}, max={max(n_drugs_per_patient)}")

    result = "\n".join(lines)
    print(result)
    return result


# ===================================================================
# Session data generation (for leakage analysis)
# ===================================================================

def generate_session_data(
    num_sessions: int = 200,
    num_drugs: int = 25,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate synthetic tool-call sessions with ground-truth intents.

    Each session represents one user interaction with the AI agent,
    consisting of a sequence of tool calls and a labeled intent.

    Args:
        num_sessions: Number of sessions to generate.
        num_drugs: Size of the drug pool to sample from.
        seed: Random seed.

    Returns:
        List of session dicts with keys: session_id, intent, calls.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    drugs = _build_drug_pool()[:num_drugs]

    intent_patterns: dict[str, list[list[str]]] = {
        "check_drug_interactions": [
            ["drug_interaction_db"],
            ["drug_interaction_db", "drug_interaction_db"],
        ],
        "assess_cardiac_risk": [
            ["risk_scoring_service"],
            ["risk_scoring_service", "drug_interaction_db"],
        ],
        "review_medication_safety": [
            ["drug_interaction_db", "drug_interaction_db", "risk_scoring_service"],
            ["drug_interaction_db", "risk_scoring_service"],
        ],
        "evaluate_treatment_plan": [
            ["drug_interaction_db", "risk_scoring_service", "drug_interaction_db"],
        ],
        "screen_contraindications": [
            ["drug_interaction_db", "drug_interaction_db", "drug_interaction_db"],
        ],
    }

    sessions: list[dict[str, Any]] = []
    for i in range(num_sessions):
        intent = rng.choice(USER_INTENTS)
        pattern = rng.choice(intent_patterns[intent])

        calls: list[dict[str, Any]] = []
        for tool_name in pattern:
            if tool_name == "drug_interaction_db":
                drug_a, drug_b = rng.sample(drugs, 2)
                calls.append({
                    "tool": tool_name,
                    "params": {"drug_a": drug_a, "drug_b": drug_b},
                })
            else:
                features = {
                    "age": int(np_rng.normal(55, 15)),
                    "bmi": round(float(np_rng.normal(27, 5)), 1),
                    "systolic_bp": int(np_rng.normal(130, 20)),
                    "diastolic_bp": int(np_rng.normal(80, 12)),
                    "cholesterol_total": int(np_rng.normal(200, 40)),
                    "glucose_fasting": int(np_rng.normal(100, 30)),
                    "num_medications": int(np_rng.integers(0, 15)),
                    "num_conditions": int(np_rng.integers(0, 10)),
                }
                calls.append({"tool": tool_name, "params": features})

        sessions.append({
            "session_id": f"session_{i:04d}",
            "intent": intent,
            "calls": calls,
        })

    return sessions


# ===================================================================
# Legacy loaders (kept for backward compatibility with encrypted_lookup)
# ===================================================================

def load_drug_database(path: Path) -> pd.DataFrame:
    """Load a drug-interaction database from CSV or JSON.

    Args:
        path: Path to the data file.

    Returns:
        DataFrame with drug interaction data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {path}")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def load_patient_data(path: Path) -> pd.DataFrame:
    """Load patient data from CSV or JSON.

    Args:
        path: Path to the data file.

    Returns:
        DataFrame with patient features.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Patient data file not found: {path}")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


# ===================================================================
# Full generation pipeline
# ===================================================================

def generate_all_datasets(seed: int = 42) -> dict[str, Path]:
    """Generate all datasets from scratch.  Called by data/generate_all.py.

    Returns a dict mapping dataset name to output path.
    """
    paths: dict[str, Path] = {}

    # Drug DBs
    print("\n--- Generating drug interaction databases ---")
    db_10k = generate_drug_db(num_records=10_000, seed=seed)
    p = _save_drug_db(db_10k, DATA_DIR / "drug_interactions_10k.json")
    paths["drug_interactions_10k"] = p
    print(f"  [OK] {p.name}: {len(db_10k)} records")

    db_200 = db_10k[:200]
    p = _save_drug_db(db_200, DATA_DIR / "drug_interactions_200.json")
    paths["drug_interactions_200"] = p
    print(f"  [OK] {p.name}: {len(db_200)} records (subset for quick testing)")

    # Patient profiles
    print("\n--- Generating patient profiles ---")
    pts_1000 = generate_patients(num_patients=1000, seed=seed, include_edge_cases=True)
    p = _save_patients(pts_1000, DATA_DIR / "synthetic_patients_1000.json")
    paths["synthetic_patients_1000"] = p
    print(f"  [OK] {p.name}: {len(pts_1000)} patients (incl. 50 edge cases)")

    pts_10 = pts_1000[:10]
    p = _save_patients(pts_10, DATA_DIR / "synthetic_patients_10.json")
    paths["synthetic_patients_10"] = p
    print(f"  [OK] {p.name}: {len(pts_10)} patients (subset for quick testing)")

    return paths
