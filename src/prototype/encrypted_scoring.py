"""FHE-based encrypted patient risk scoring.

End-to-end pipeline for privately computing a patient risk score:

1. Agent encrypts patient feature vector using CKKS.
2. Server evaluates a risk model on the ciphertext:
   - Linear scoring: ``score = w · x + b``
   - Optional polynomial activation: ``activated = poly(score)``
3. Agent decrypts the encrypted result.

The server computes the score without ever seeing the patient's
features or learning the resulting risk level.

Protocol (two-party)
--------------------
::

    Agent (holds secret key)              Server (public key only)
    ========================              =======================
    1. setup() → public context  ───────► load public context
    2. encrypt features          ───────► receive ciphertext
                                          compute on ciphertext
    3. receive encrypted result  ◄─────── send encrypted result
    4. decrypt → plaintext score

The server holds the model weights in plaintext (they are public /
shared).  Only the *patient data* and the *resulting score* are
private.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

import tenseal as ts

from src.feasibility.fhe_computation import (
    EncryptedLinearModel,
    EncryptedPolynomialModel,
    FHEContext,
    FHEParams,
)


# ===================================================================
# Feature normalization ranges (clinically meaningful)
# ===================================================================
# Each tuple is (min, max) for mapping to [0, 1].
_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "age":                      (0.0,   120.0),
    "bmi":                      (10.0,   60.0),
    "blood_pressure_systolic":  (60.0,  250.0),
    "blood_pressure_diastolic": (30.0,  150.0),
    "cholesterol":              (100.0, 400.0),
    "glucose":                  (50.0,  500.0),
    "num_medications":          (0.0,   30.0),
    "num_conditions":           (0.0,   20.0),
}

_FEATURE_ORDER = list(_FEATURE_RANGES.keys())
NUM_FEATURES = len(_FEATURE_ORDER)


# ===================================================================
# PatientFeatures
# ===================================================================

@dataclass
class PatientFeatures:
    """Structured patient features for risk scoring.

    Attributes
    ----------
    age : float
        Patient age in years.
    bmi : float
        Body mass index.
    blood_pressure_systolic : float
        Systolic BP in mmHg.
    blood_pressure_diastolic : float
        Diastolic BP in mmHg.
    cholesterol : float
        Total cholesterol in mg/dL.
    glucose : float
        Fasting glucose in mg/dL.
    num_medications : int
        Number of current medications.
    num_conditions : int
        Number of diagnosed conditions.
    """

    age: float
    bmi: float
    blood_pressure_systolic: float
    blood_pressure_diastolic: float
    cholesterol: float
    glucose: float
    num_medications: int
    num_conditions: int

    def to_vector(self) -> NDArray[np.float64]:
        """Convert to a normalized feature vector in [0, 1].

        Normalization uses clinically meaningful ranges so that all
        features contribute proportionally to the risk score.
        """
        raw = [
            self.age, self.bmi,
            self.blood_pressure_systolic, self.blood_pressure_diastolic,
            self.cholesterol, self.glucose,
            float(self.num_medications), float(self.num_conditions),
        ]
        normed = []
        for val, name in zip(raw, _FEATURE_ORDER):
            lo, hi = _FEATURE_RANGES[name]
            normed.append((val - lo) / (hi - lo))
        return np.array(normed, dtype=np.float64)

    @classmethod
    def from_vector(cls, vector: NDArray[np.float64]) -> PatientFeatures:
        """Reconstruct PatientFeatures from a normalized vector."""
        values: dict[str, float] = {}
        for i, name in enumerate(_FEATURE_ORDER):
            lo, hi = _FEATURE_RANGES[name]
            values[name] = vector[i] * (hi - lo) + lo
        return cls(
            age=values["age"],
            bmi=values["bmi"],
            blood_pressure_systolic=values["blood_pressure_systolic"],
            blood_pressure_diastolic=values["blood_pressure_diastolic"],
            cholesterol=values["cholesterol"],
            glucose=values["glucose"],
            num_medications=int(round(values["num_medications"])),
            num_conditions=int(round(values["num_conditions"])),
        )

    @classmethod
    def random(cls, rng: np.random.Generator | None = None) -> PatientFeatures:
        """Generate a random patient with realistic feature values."""
        rng = rng or np.random.default_rng()
        return cls(
            age=rng.uniform(18, 95),
            bmi=rng.uniform(16, 45),
            blood_pressure_systolic=rng.uniform(90, 200),
            blood_pressure_diastolic=rng.uniform(50, 120),
            cholesterol=rng.uniform(120, 350),
            glucose=rng.uniform(60, 300),
            num_medications=int(rng.integers(0, 15)),
            num_conditions=int(rng.integers(0, 10)),
        )


# ===================================================================
# Scoring event log
# ===================================================================

@dataclass
class ScoringEvent:
    """Metadata logged by the server for each scoring request."""
    timestamp: float
    encrypted_input_size_bytes: int
    encrypted_output_size_bytes: int
    computation_time_ms: float
    features_visible: bool = False   # always False for encrypted scoring
    score_visible: bool = False      # always False


# ===================================================================
# EncryptedRiskScorer (renamed from EncryptedScoringTool for clarity)
# ===================================================================

class EncryptedScoringTool:
    """Private risk scoring using FHE.

    The agent encrypts patient features, the server evaluates a risk
    model on the ciphertext, and the agent decrypts the result.  The
    server never sees the patient data or the resulting score.

    Parameters
    ----------
    fhe_params : FHEParams, optional
        CKKS parameters.  Defaults to ``FHEParams.deep()`` (poly_modulus
        16384, depth 4) which supports dot-product + cubic activation.
        Use ``FHEParams.light()`` if you only need a linear model
        (dot product, no polynomial activation).

    Example
    -------
    ::

        scorer = EncryptedScoringTool()
        scorer.setup(model_weights, model_bias)

        patient = PatientFeatures(age=65, bmi=28.5, ...)
        encrypted = scorer.encrypt_features(patient)       # agent
        enc_score = scorer.compute_score_encrypted(encrypted)  # server
        score = scorer.decrypt_score(enc_score)            # agent
    """

    def __init__(self, fhe_params: FHEParams | None = None) -> None:
        self._fhe_params = fhe_params or FHEParams.deep()
        self._fhe_ctx: FHEContext | None = None
        self._linear_model: EncryptedLinearModel | None = None
        self._poly_model: EncryptedPolynomialModel | None = None
        # Observability
        self.scoring_log: list[ScoringEvent] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(
        self,
        weights: list[float] | NDArray[np.float64],
        bias: float,
        activation_coeffs: list[float] | None = None,
    ) -> None:
        """Initialize FHE context and load the risk model.

        Parameters
        ----------
        weights : array-like
            Linear model weights (length = NUM_FEATURES = 8).
        bias : float
            Model bias / intercept.
        activation_coeffs : list[float], optional
            Polynomial coefficients for a nonlinear activation applied
            after the linear scoring.  E.g. a sigmoid approximation
            ``[0.5, 0.197, 0.0, -0.004]``.  If ``None``, no activation
            is applied and the raw linear score is returned.
        """
        self._fhe_ctx = FHEContext(self._fhe_params)
        self._fhe_ctx.setup()

        if isinstance(weights, np.ndarray):
            weights = weights.tolist()
        self._linear_model = EncryptedLinearModel(weights, bias)

        if activation_coeffs is not None:
            self._poly_model = EncryptedPolynomialModel(activation_coeffs)
        else:
            self._poly_model = None

    def _require_ctx(self) -> FHEContext:
        if self._fhe_ctx is None:
            raise RuntimeError("Call setup() first")
        return self._fhe_ctx

    # ------------------------------------------------------------------
    # Context transfer
    # ------------------------------------------------------------------

    def get_server_context(self) -> bytes:
        """Serialize the public FHE context for the server.

        The server uses this to perform computation on ciphertexts
        but cannot decrypt anything.
        """
        return self._require_ctx().public_context_bytes()

    # ------------------------------------------------------------------
    # Agent-side: encrypt / decrypt
    # ------------------------------------------------------------------

    def encrypt_features(self, patient: PatientFeatures) -> ts.CKKSVector:
        """Agent encrypts patient features.

        The features are normalized to [0, 1] and then encrypted.
        """
        ctx = self._require_ctx()
        vec = patient.to_vector()
        return ctx.encrypt_vector(vec)

    def encrypt_raw_vector(self, values: list[float]) -> ts.CKKSVector:
        """Encrypt a raw float vector (for benchmarking)."""
        return self._require_ctx().encrypt_vector(values)

    def decrypt_score(self, encrypted_score: ts.CKKSVector) -> float:
        """Agent decrypts the risk score."""
        return self._require_ctx().decrypt_result(encrypted_score)

    # ------------------------------------------------------------------
    # Server-side: compute on ciphertext
    # ------------------------------------------------------------------

    def compute_score_encrypted(
        self, encrypted_features: ts.CKKSVector,
    ) -> ts.CKKSVector:
        """Server computes risk score on encrypted features.

        This method runs on the server using only the public context.
        It cannot decrypt anything.

        Steps:
        1. Linear scoring: ``w · enc_x + b``
        2. Optional polynomial activation on the result.
        """
        if self._linear_model is None:
            raise RuntimeError("Call setup() first")

        t0 = time.perf_counter()

        linear_score = self._linear_model.predict_encrypted(encrypted_features)

        if self._poly_model is not None:
            result = self._poly_model.evaluate_encrypted(linear_score)
        else:
            result = linear_score

        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.scoring_log.append(ScoringEvent(
            timestamp=time.time(),
            encrypted_input_size_bytes=len(encrypted_features.serialize()),
            encrypted_output_size_bytes=len(result.serialize()),
            computation_time_ms=elapsed_ms,
        ))

        return result

    # ------------------------------------------------------------------
    # End-to-end convenience
    # ------------------------------------------------------------------

    def score(self, patient: PatientFeatures) -> float:
        """End-to-end encrypted scoring (agent → server → agent)."""
        enc_features = self.encrypt_features(patient)
        enc_score = self.compute_score_encrypted(enc_features)
        return self.decrypt_score(enc_score)

    # ------------------------------------------------------------------
    # Plaintext baseline
    # ------------------------------------------------------------------

    def score_plaintext(self, patient: PatientFeatures) -> float:
        """Compute risk score in plaintext (baseline, not private).

        Uses the same model weights but skips encryption entirely.
        """
        if self._linear_model is None:
            raise RuntimeError("Call setup() first")
        vec = patient.to_vector().tolist()
        raw = self._linear_model.predict_plaintext(vec)
        if self._poly_model is not None:
            return self._poly_model.evaluate_plaintext(raw)
        return raw

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_correctness(
        self,
        patient: PatientFeatures,
        tolerance: float = 0.01,
    ) -> tuple[bool, float, float]:
        """Verify that encrypted scoring matches plaintext scoring.

        Returns
        -------
        tuple[bool, float, float]
            (is_correct, encrypted_score, plaintext_score)
            where ``is_correct = |enc - plain| < tolerance``.
        """
        enc_score = self.score(patient)
        plain_score = self.score_plaintext(patient)
        return (abs(enc_score - plain_score) < tolerance, enc_score, plain_score)


# ===================================================================
# Pre-built risk scoring models
# ===================================================================

def cardiovascular_risk_model() -> tuple[NDArray[np.float64], float, list[float]]:
    """Return weights, bias, and activation coefficients for a
    simple cardiovascular risk scoring model.

    The model is a synthetic example designed for research demonstration.
    Features are assumed to be normalized to [0, 1].

    Returns
    -------
    weights : ndarray
        Weight vector of length 8.
    bias : float
        Intercept.
    activation : list[float]
        Cubic sigmoid approximation coefficients.
    """
    # Weights correspond to _FEATURE_ORDER:
    #   age, bmi, sys_bp, dia_bp, cholesterol, glucose, meds, conditions
    weights = np.array([
        0.25,   # age (strong positive)
        0.15,   # BMI
        0.20,   # systolic BP
        0.10,   # diastolic BP
        0.15,   # cholesterol
        0.10,   # glucose
        0.03,   # number of medications (weak)
        0.02,   # number of conditions (weak)
    ], dtype=np.float64)
    bias = -0.5  # shifts raw score so that average patient ≈ 0

    # Sigmoid approximation: maps (-∞, +∞) → (0, 1)
    # Good for |x| ≤ 5: σ(x) ≈ 0.5 + 0.197·x − 0.004·x³
    activation = [0.5, 0.197, 0.0, -0.004]

    return weights, bias, activation


def simple_linear_model() -> tuple[NDArray[np.float64], float]:
    """Return weights and bias for a simple linear risk model.

    No activation function — returns the raw dot-product score.
    Can run on lighter CKKS parameters (depth 2).
    """
    weights = np.array([
        0.20, 0.15, 0.20, 0.10, 0.15, 0.10, 0.05, 0.05,
    ], dtype=np.float64)
    bias = 0.0
    return weights, bias
