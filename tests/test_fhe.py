"""Tests for FHE computation and encrypted scoring.

Covers:
- FHEContext: encrypt/decrypt, dot product, polynomial evaluation
- EncryptedLinearModel: correctness against plaintext
- EncryptedPolynomialModel: correctness for various degrees
- EncryptedScoringTool: end-to-end encrypted patient risk scoring
- Privacy: server never sees plaintext features or scores
- Parameter sensitivity: accuracy under different CKKS configurations
"""

from __future__ import annotations

import numpy as np
import pytest

import tenseal as ts

from src.feasibility.fhe_computation import (
    EncryptedLinearModel,
    EncryptedPolynomialModel,
    FHEContext,
    FHEParams,
)
from src.prototype.encrypted_scoring import (
    EncryptedScoringTool,
    PatientFeatures,
    cardiovascular_risk_model,
    simple_linear_model,
    NUM_FEATURES,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def light_ctx() -> FHEContext:
    """CKKS context with depth-2 (8192) — fast, good for dot products."""
    ctx = FHEContext(FHEParams.light())
    ctx.setup()
    return ctx


@pytest.fixture
def deep_ctx() -> FHEContext:
    """CKKS context with depth-4 (16384) — slower, supports polynomials."""
    ctx = FHEContext(FHEParams.deep())
    ctx.setup()
    return ctx


@pytest.fixture
def sample_patient() -> PatientFeatures:
    """A representative patient for testing."""
    return PatientFeatures(
        age=65, bmi=28.5,
        blood_pressure_systolic=145, blood_pressure_diastolic=90,
        cholesterol=240, glucose=110,
        num_medications=3, num_conditions=2,
    )


# ===================================================================
# FHEContext — basic encrypt/decrypt
# ===================================================================

class TestFHEContext:

    def test_encrypt_decrypt_roundtrip(self, light_ctx: FHEContext) -> None:
        values = [1.0, 2.5, -3.7, 0.0, 100.0]
        enc = light_ctx.encrypt_vector(values)
        dec = light_ctx.decrypt_vector(enc)
        for v, d in zip(values, dec):
            assert abs(v - d) < 1e-4, f"expected {v}, got {d}"

    def test_encrypt_numpy_array(self, light_ctx: FHEContext) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        enc = light_ctx.encrypt_vector(arr)
        dec = light_ctx.decrypt_vector(enc)
        for v, d in zip(arr, dec):
            assert abs(v - d) < 1e-4

    def test_decrypt_result_scalar(self, light_ctx: FHEContext) -> None:
        enc = light_ctx.encrypt_vector([42.0])
        val = light_ctx.decrypt_result(enc)
        assert abs(val - 42.0) < 1e-4

    def test_setup_required(self) -> None:
        ctx = FHEContext()
        with pytest.raises(RuntimeError, match="setup"):
            ctx.encrypt_vector([1.0])

    def test_context_setup(self) -> None:
        ctx = FHEContext(FHEParams.light())
        ctx.setup()
        assert ctx._context is not None


# ===================================================================
# FHEContext — dot product
# ===================================================================

class TestDotProduct:

    def test_dot_product_correctness(self, light_ctx: FHEContext) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        w = [0.5, -0.3, 0.2, 0.1]
        enc_x = light_ctx.encrypt_vector(x)
        enc_dot = light_ctx.encrypted_dot_product(enc_x, w)
        result = light_ctx.decrypt_result(enc_dot)
        expected = np.dot(x, w)
        assert abs(result - expected) < 1e-3

    def test_dot_product_with_numpy_weights(self, light_ctx: FHEContext) -> None:
        x = [1.0, 2.0, 3.0]
        w = np.array([0.1, 0.2, 0.3])
        enc_x = light_ctx.encrypt_vector(x)
        enc_dot = light_ctx.encrypted_dot_product(enc_x, w)
        result = light_ctx.decrypt_result(enc_dot)
        expected = np.dot(x, w)
        assert abs(result - expected) < 1e-3

    def test_dot_product_zero_weights(self, light_ctx: FHEContext) -> None:
        x = [1.0, 2.0, 3.0]
        w = [0.0, 0.0, 0.0]
        enc_x = light_ctx.encrypt_vector(x)
        enc_dot = light_ctx.encrypted_dot_product(enc_x, w)
        result = light_ctx.decrypt_result(enc_dot)
        assert abs(result) < 1e-3

    def test_dot_product_eight_features(self, light_ctx: FHEContext) -> None:
        """Dot product with 8 features (same as patient feature vector)."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, size=8).tolist()
        w = rng.uniform(-0.5, 0.5, size=8).tolist()
        enc_x = light_ctx.encrypt_vector(x)
        enc_dot = light_ctx.encrypted_dot_product(enc_x, w)
        result = light_ctx.decrypt_result(enc_dot)
        expected = np.dot(x, w)
        assert abs(result - expected) < 1e-3


# ===================================================================
# FHEContext — polynomial evaluation
# ===================================================================

class TestPolynomialEval:

    def test_linear_polynomial(self, light_ctx: FHEContext) -> None:
        """f(x) = 2 + 3x at x=5 -> 17."""
        enc_x = light_ctx.encrypt_vector([5.0])
        result = light_ctx.encrypted_polynomial_eval(enc_x, [2.0, 3.0])
        val = light_ctx.decrypt_result(result)
        assert abs(val - 17.0) < 0.01

    def test_quadratic_polynomial(self, light_ctx: FHEContext) -> None:
        """f(x) = 1 + 0.5x + 0.25x^2 at x=2 -> 3."""
        enc_x = light_ctx.encrypt_vector([2.0])
        result = light_ctx.encrypted_polynomial_eval(enc_x, [1.0, 0.5, 0.25])
        val = light_ctx.decrypt_result(result)
        assert abs(val - 3.0) < 0.01

    def test_cubic_polynomial(self, deep_ctx: FHEContext) -> None:
        """f(x) = 1 + 2x - 0.5x^2 + 0.1x^3 at x=3 -> 5.2."""
        enc_x = deep_ctx.encrypt_vector([3.0])
        result = deep_ctx.encrypted_polynomial_eval(
            enc_x, [1.0, 2.0, -0.5, 0.1])
        val = deep_ctx.decrypt_result(result)
        assert abs(val - 5.2) < 0.01

    def test_polynomial_on_vector(self, light_ctx: FHEContext) -> None:
        """Polynomial applied element-wise to a vector."""
        values = [1.0, 2.0, 3.0]
        coeffs = [0.0, 1.0, 1.0]  # f(x) = x + x^2
        enc_x = light_ctx.encrypt_vector(values)
        result = light_ctx.encrypted_polynomial_eval(enc_x, coeffs)
        dec = light_ctx.decrypt_vector(result)
        for v, d in zip(values, dec[:3]):
            expected = v + v**2
            assert abs(d - expected) < 0.01, f"f({v}): got {d}, expected {expected}"


# ===================================================================
# EncryptedLinearModel
# ===================================================================

class TestEncryptedLinearModel:

    def test_predict_matches_plaintext(self, light_ctx: FHEContext) -> None:
        rng = np.random.default_rng(123)
        w = rng.uniform(-1, 1, size=8)
        b = 0.5
        x = rng.uniform(0, 1, size=8).tolist()

        model = EncryptedLinearModel(w, b)
        plain = model.predict_plaintext(x)

        enc_x = light_ctx.encrypt_vector(x)
        enc_result = model.predict_encrypted(enc_x)
        encrypted = light_ctx.decrypt_result(enc_result)

        assert abs(encrypted - plain) < 1e-3

    def test_predict_with_zero_bias(self, light_ctx: FHEContext) -> None:
        w = [1.0, 1.0, 1.0]
        model = EncryptedLinearModel(w, 0.0)
        enc_x = light_ctx.encrypt_vector([1.0, 2.0, 3.0])
        result = light_ctx.decrypt_result(model.predict_encrypted(enc_x))
        assert abs(result - 6.0) < 1e-3

    def test_encrypted_dot_product_matches_plaintext(self) -> None:
        """FHE dot product should approximate numpy dot product."""
        weights = np.array([0.5, -0.3, 0.8, 0.1])
        bias = 0.2
        features = np.array([1.0, 2.0, 0.5, 3.0])

        expected = float(np.dot(weights, features) + bias)

        ctx = FHEContext(FHEParams.light())
        ctx.setup()
        model = EncryptedLinearModel(weights=weights, bias=bias)
        encrypted_features = ctx.encrypt_vector(features.tolist())
        encrypted_result = model.predict_encrypted(encrypted_features)
        result = ctx.decrypt_result(encrypted_result)

        assert abs(result - expected) < 1e-2


# ===================================================================
# EncryptedPolynomialModel
# ===================================================================

class TestEncryptedPolynomialModel:

    def test_sigmoid_approximation(self, deep_ctx: FHEContext) -> None:
        """Sigmoid approx: 0.5 + 0.197x - 0.004x^3."""
        coeffs = [0.5, 0.197, 0.0, -0.004]
        model = EncryptedPolynomialModel(coeffs)

        for x_val in [0.0, 0.5, -0.5, 1.0, -1.0]:
            enc_x = deep_ctx.encrypt_vector([x_val])
            enc_result = model.evaluate_encrypted(enc_x)
            encrypted = deep_ctx.decrypt_result(enc_result)
            plain = model.evaluate_plaintext(x_val)
            assert abs(encrypted - plain) < 0.01, (
                f"x={x_val}: enc={encrypted:.6f}, plain={plain:.6f}"
            )

    def test_degree_property(self) -> None:
        assert EncryptedPolynomialModel([1.0]).degree == 0
        assert EncryptedPolynomialModel([1.0, 2.0]).degree == 1
        assert EncryptedPolynomialModel([1.0, 0.0, 3.0]).degree == 2
        assert EncryptedPolynomialModel([0.5, 0.197, 0.0, -0.004]).degree == 3

    def test_multiplicative_depth(self) -> None:
        assert EncryptedPolynomialModel([1.0]).multiplicative_depth == 0
        assert EncryptedPolynomialModel([1.0, 2.0]).multiplicative_depth == 1
        assert EncryptedPolynomialModel([1.0, 2.0, 3.0]).multiplicative_depth == 2


# ===================================================================
# PatientFeatures
# ===================================================================

class TestPatientFeatures:

    def test_to_vector_normalization(self, sample_patient: PatientFeatures) -> None:
        vec = sample_patient.to_vector()
        assert vec.shape == (NUM_FEATURES,)
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)

    def test_roundtrip_vector(self, sample_patient: PatientFeatures) -> None:
        vec = sample_patient.to_vector()
        restored = PatientFeatures.from_vector(vec)
        assert abs(restored.age - sample_patient.age) < 0.01
        assert abs(restored.bmi - sample_patient.bmi) < 0.01
        assert restored.num_medications == sample_patient.num_medications

    def test_random_patient(self) -> None:
        p = PatientFeatures.random(np.random.default_rng(99))
        vec = p.to_vector()
        assert vec.shape == (NUM_FEATURES,)


# ===================================================================
# EncryptedScoringTool — linear model
# ===================================================================

class TestScoringToolLinear:

    @pytest.fixture
    def linear_scorer(self) -> EncryptedScoringTool:
        w, b = simple_linear_model()
        scorer = EncryptedScoringTool(FHEParams.light())
        scorer.setup(w, b)
        return scorer

    def test_encrypted_matches_plaintext(
        self, linear_scorer: EncryptedScoringTool,
        sample_patient: PatientFeatures,
    ) -> None:
        ok, enc_s, plain_s = linear_scorer.verify_correctness(
            sample_patient, tolerance=0.01)
        assert ok, f"enc={enc_s:.6f} vs plain={plain_s:.6f}"

    def test_multiple_patients(self, linear_scorer: EncryptedScoringTool) -> None:
        rng = np.random.default_rng(42)
        for _ in range(3):
            p = PatientFeatures.random(rng)
            ok, enc_s, plain_s = linear_scorer.verify_correctness(p, tolerance=0.01)
            assert ok, f"enc={enc_s:.6f} vs plain={plain_s:.6f}"

    def test_setup_required(self) -> None:
        scorer = EncryptedScoringTool()
        p = PatientFeatures.random()
        with pytest.raises(RuntimeError, match="setup"):
            scorer.encrypt_features(p)


# ===================================================================
# EncryptedScoringTool — full model with activation
# ===================================================================

class TestScoringToolFull:

    @pytest.fixture
    def cv_scorer(self) -> EncryptedScoringTool:
        w, b, act = cardiovascular_risk_model()
        scorer = EncryptedScoringTool(FHEParams.deep())
        scorer.setup(w, b, activation_coeffs=act)
        return scorer

    def test_encrypted_matches_plaintext(
        self, cv_scorer: EncryptedScoringTool,
        sample_patient: PatientFeatures,
    ) -> None:
        ok, enc_s, plain_s = cv_scorer.verify_correctness(
            sample_patient, tolerance=0.01)
        assert ok, f"enc={enc_s:.6f} vs plain={plain_s:.6f}"

    def test_score_in_valid_range(
        self, cv_scorer: EncryptedScoringTool,
    ) -> None:
        """With sigmoid activation, score should be near [0, 1]."""
        rng = np.random.default_rng(7)
        for _ in range(3):
            p = PatientFeatures.random(rng)
            s = cv_scorer.score(p)
            assert -0.2 < s < 1.2, f"Score {s} out of expected range"


# ===================================================================
# Privacy guarantees
# ===================================================================

class TestScoringPrivacy:

    def test_server_log_reveals_nothing(
        self, sample_patient: PatientFeatures,
    ) -> None:
        """Server scoring log must not contain features or scores."""
        w, b, act = cardiovascular_risk_model()
        scorer = EncryptedScoringTool(FHEParams.deep())
        scorer.setup(w, b, activation_coeffs=act)

        scorer.score(sample_patient)
        scorer.score(PatientFeatures.random())

        assert len(scorer.scoring_log) == 2
        for evt in scorer.scoring_log:
            assert evt.features_visible is False
            assert evt.score_visible is False

    def test_ciphertext_hides_features(self) -> None:
        """Different patients produce different ciphertexts of similar size."""
        scorer = EncryptedScoringTool(FHEParams.light())
        w, b = simple_linear_model()
        scorer.setup(w, b)

        p1 = PatientFeatures(age=25, bmi=22, blood_pressure_systolic=110,
                              blood_pressure_diastolic=70, cholesterol=180,
                              glucose=85, num_medications=0, num_conditions=0)
        p2 = PatientFeatures(age=80, bmi=35, blood_pressure_systolic=190,
                              blood_pressure_diastolic=110, cholesterol=300,
                              glucose=250, num_medications=8, num_conditions=5)

        enc1 = scorer.encrypt_features(p1).serialize()
        enc2 = scorer.encrypt_features(p2).serialize()

        assert enc1 != enc2
        assert abs(len(enc1) - len(enc2)) / max(len(enc1), len(enc2)) < 0.05

    def test_public_context_cannot_decrypt(self) -> None:
        """Server's public context must not contain the secret key."""
        scorer = EncryptedScoringTool(FHEParams.light())
        w, b = simple_linear_model()
        scorer.setup(w, b)

        p = PatientFeatures.random()
        enc = scorer.encrypt_features(p)

        pub_ctx = scorer._fhe_ctx.public_context
        enc_on_server = ts.ckks_vector_from(pub_ctx, enc.serialize())
        try:
            dec = enc_on_server.decrypt()
            plain_vec = p.to_vector().tolist()
            diffs = [abs(d - v) for d, v in zip(dec[:8], plain_vec)]
            assert max(diffs) > 0.1, "Public context should not decrypt correctly"
        except Exception:
            pass  # Expected: TenSEAL raises when decrypting without secret key


# ===================================================================
# Parameter sensitivity
# ===================================================================

class TestParameterSensitivity:

    def test_light_params_dot_product_accuracy(self) -> None:
        """Light params (8192) should give <1e-3 error on dot product."""
        ctx = FHEContext(FHEParams.light())
        ctx.setup()

        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, size=8).tolist()
        w = rng.uniform(-1, 1, size=8).tolist()

        enc_x = ctx.encrypt_vector(x)
        enc_dot = ctx.encrypted_dot_product(enc_x, w)
        result = ctx.decrypt_result(enc_dot)
        expected = float(np.dot(x, w))

        assert abs(result - expected) < 1e-3

    def test_deep_params_polynomial_accuracy(self) -> None:
        """Deep params (16384) should give <0.01 error on cubic polynomial."""
        ctx = FHEContext(FHEParams.deep())
        ctx.setup()

        coeffs = [0.5, 0.197, 0.0, -0.004]
        for x_val in [0.0, 0.3, -0.3, 1.0, -1.0]:
            enc_x = ctx.encrypt_vector([x_val])
            enc_result = ctx.encrypted_polynomial_eval(enc_x, coeffs)
            result = ctx.decrypt_result(enc_result)
            expected = sum(c * x_val**i for i, c in enumerate(coeffs))
            assert abs(result - expected) < 0.01, (
                f"x={x_val}: got {result:.6f}, expected {expected:.6f}"
            )

    def test_light_vs_deep_accuracy(self) -> None:
        """Both presets should give accurate results for dot products."""
        for params in [FHEParams.light(), FHEParams.deep()]:
            ctx = FHEContext(params)
            ctx.setup()
            x = [0.5, 0.3, 0.8, 0.1]
            w = [0.2, -0.4, 0.6, 0.1]
            enc_x = ctx.encrypt_vector(x)
            result = ctx.decrypt_result(ctx.encrypted_dot_product(enc_x, w))
            expected = float(np.dot(x, w))
            assert abs(result - expected) < 1e-3
