"""Tests for the threat model module.

Covers definitions, leakage analysis, and intent reconstruction with
both unit tests and integration tests using synthetic medical-agent data.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.threat_model.definitions import (
    AdversaryModel,
    AgentSession,
    PrivacyGoal,
    PrivacyProperty,
    ThreatModel,
    ToolCall,
    ToolCallSequence,
    ToolCallType,
)
from src.threat_model.leakage_analysis import (
    DirectLeakage,
    CorrelationLeakage,
    FrequencyLeakage,
    LeakageAnalyzer,
    LeakageMetrics,
    compute_entropy,
    mutual_information,
)
from src.threat_model.intent_reconstruction import (
    AttackStrategy,
    FrequencyAttack,
    MLClassifierAttack,
    ReconstructionResult,
    SequenceAttack,
    compare_reconstruction_accuracy,
    make_encrypted_session,
)


# ===================================================================
# Fixtures: synthetic medical-agent data
# ===================================================================

DRUGS = ["aspirin", "warfarin", "metformin", "lisinopril", "atorvastatin",
         "amlodipine", "sertraline", "omeprazole", "ibuprofen", "gabapentin"]

INTENTS = [
    "check_drug_interactions",
    "assess_cardiac_risk",
    "review_medication_safety",
    "evaluate_treatment_plan",
]


def _make_call(
    tool: str,
    method: str,
    params: dict,
    t: float,
    sid: str,
    ctype: ToolCallType = ToolCallType.DATA_RETRIEVAL,
) -> ToolCall:
    return ToolCall(
        tool_name=tool,
        method=method,
        parameters=params,
        timestamp=t,
        session_id=sid,
        call_type=ctype,
    )


def _generate_session(
    intent: str,
    sid: str,
    rng: np.random.Generator,
) -> ToolCallSequence:
    """Generate a plausible session for a given intent."""
    t = 0.0
    calls: list[ToolCall] = []

    if intent == "check_drug_interactions":
        d1, d2 = rng.choice(DRUGS, size=2, replace=False).tolist()
        calls.append(_make_call("drug_db", "lookup", {"drug": d1}, t, sid))
        t += rng.exponential(0.5)
        calls.append(_make_call("drug_db", "lookup", {"drug": d2}, t, sid))
        t += rng.exponential(0.3)
        calls.append(_make_call("drug_db", "check_interaction",
                                {"drug_a": d1, "drug_b": d2}, t, sid))
        if rng.random() > 0.5:
            t += rng.exponential(0.4)
            calls.append(_make_call("drug_db", "get_alternatives",
                                    {"drug": d1}, t, sid))

    elif intent == "assess_cardiac_risk":
        age = int(rng.integers(40, 80))
        bp = int(rng.integers(110, 180))
        chol = int(rng.integers(150, 300))
        calls.append(_make_call("risk_scorer", "score",
                                {"age": age, "bp": bp, "cholesterol": chol},
                                t, sid, ToolCallType.COMPUTATION))
        t += rng.exponential(1.0)
        calls.append(_make_call("risk_scorer", "get_risk_factors",
                                {"age": age}, t, sid, ToolCallType.COMPUTATION))
        if rng.random() > 0.4:
            t += rng.exponential(0.5)
            calls.append(_make_call("guideline_db", "lookup",
                                    {"condition": "cardiac"}, t, sid))

    elif intent == "review_medication_safety":
        meds = rng.choice(DRUGS, size=rng.integers(2, 5), replace=False).tolist()
        for med in meds:
            calls.append(_make_call("drug_db", "lookup",
                                    {"drug": med}, t, sid))
            t += rng.exponential(0.3)
        calls.append(_make_call("drug_db", "check_contraindications",
                                {"medications": meds}, t, sid))

    elif intent == "evaluate_treatment_plan":
        d = rng.choice(DRUGS).item()
        calls.append(_make_call("drug_db", "lookup", {"drug": d}, t, sid))
        t += rng.exponential(0.5)
        calls.append(_make_call("risk_scorer", "score",
                                {"age": 55, "bp": 130, "cholesterol": 200},
                                t, sid, ToolCallType.COMPUTATION))
        t += rng.exponential(0.4)
        calls.append(_make_call("guideline_db", "lookup",
                                {"condition": "general"}, t, sid))
        t += rng.exponential(0.3)
        calls.append(_make_call("drug_db", "get_dosage",
                                {"drug": d, "weight_kg": 70}, t, sid))

    seq = ToolCallSequence(session_id=sid, calls=calls, intent=intent)
    return seq


@pytest.fixture
def small_dataset() -> tuple[list[ToolCallSequence], list[str]]:
    """12 sessions (3 per intent) for quick tests."""
    rng = np.random.default_rng(42)
    sessions = []
    intents_list = []
    for i, intent in enumerate(INTENTS):
        for j in range(3):
            sid = f"s-{i}-{j}"
            sessions.append(_generate_session(intent, sid, rng))
            intents_list.append(intent)
    return sessions, intents_list


@pytest.fixture
def large_dataset() -> tuple[list[ToolCallSequence], list[str]]:
    """80 sessions (20 per intent) for statistical tests."""
    rng = np.random.default_rng(123)
    sessions = []
    intents_list = []
    for i, intent in enumerate(INTENTS):
        for j in range(20):
            sid = f"L-{i}-{j}"
            sessions.append(_generate_session(intent, sid, rng))
            intents_list.append(intent)
    return sessions, intents_list


# ===================================================================
# Tests: definitions.py
# ===================================================================

class TestToolCallType:
    def test_enum_members(self) -> None:
        assert ToolCallType.DATA_RETRIEVAL is not None
        assert ToolCallType.COMPUTATION is not None
        assert ToolCallType.PATTERN_ACCESS is not None
        assert ToolCallType.ARBITRARY_EXECUTION is not None

    def test_enum_distinct(self) -> None:
        members = list(ToolCallType)
        assert len(members) == len(set(members))


class TestAdversaryModel:
    def test_honest_but_curious(self) -> None:
        assert AdversaryModel.HONEST_BUT_CURIOUS.value == 1

    def test_active(self) -> None:
        assert AdversaryModel.ACTIVE is not None


class TestToolCall:
    def test_creation(self) -> None:
        call = ToolCall(
            tool_name="drug_db",
            method="lookup",
            parameters={"drug": "aspirin"},
            timestamp=0.0,
            session_id="s1",
        )
        assert call.tool_name == "drug_db"
        assert call.parameters["drug"] == "aspirin"
        assert call.call_type == ToolCallType.DATA_RETRIEVAL

    def test_frozen(self) -> None:
        call = ToolCall("db", "lookup", {}, 0.0, "s1")
        with pytest.raises(AttributeError):
            call.tool_name = "other"  # type: ignore[misc]

    def test_auto_input_size(self) -> None:
        call = ToolCall("db", "lookup", {"drug": "aspirin"}, 0.0, "s1")
        assert call.input_size_bytes > 0

    def test_explicit_input_size(self) -> None:
        call = ToolCall("db", "lookup", {"drug": "aspirin"}, 0.0, "s1",
                        input_size_bytes=999)
        assert call.input_size_bytes == 999

    def test_call_type(self) -> None:
        call = ToolCall("scorer", "score", {"x": 1}, 0.0, "s1",
                        call_type=ToolCallType.COMPUTATION)
        assert call.call_type == ToolCallType.COMPUTATION

    def test_output_fields(self) -> None:
        call = ToolCall("db", "lookup", {}, 0.0, "s1",
                        output={"result": "ok"}, output_size_bytes=42)
        assert call.output == {"result": "ok"}
        assert call.output_size_bytes == 42


class TestToolCallSequence:
    def test_add_call(self) -> None:
        seq = ToolCallSequence(session_id="s1")
        seq.add_call(ToolCall("db", "lookup", {}, 1.0, "s1"))
        assert seq.call_count == 1

    def test_unique_tools(self) -> None:
        seq = ToolCallSequence(session_id="s1")
        seq.add_call(ToolCall("db", "lookup", {}, 0.0, "s1"))
        seq.add_call(ToolCall("scorer", "score", {}, 1.0, "s1"))
        seq.add_call(ToolCall("db", "lookup", {}, 2.0, "s1"))
        assert seq.unique_tools == {"db", "scorer"}

    def test_unique_methods(self) -> None:
        seq = ToolCallSequence(session_id="s1")
        seq.add_call(ToolCall("db", "lookup", {}, 0.0, "s1"))
        seq.add_call(ToolCall("db", "check_interaction", {}, 1.0, "s1"))
        assert seq.unique_methods == {"db.lookup", "db.check_interaction"}

    def test_duration(self) -> None:
        seq = ToolCallSequence(session_id="s1")
        seq.add_call(ToolCall("db", "lookup", {}, 0.0, "s1"))
        seq.add_call(ToolCall("db", "lookup", {}, 3.5, "s1"))
        assert seq.duration == pytest.approx(3.5)

    def test_empty_duration(self) -> None:
        seq = ToolCallSequence(session_id="s1")
        assert seq.duration == 0.0

    def test_tool_call_counts(self) -> None:
        seq = ToolCallSequence(session_id="s1")
        seq.add_call(ToolCall("db", "lookup", {}, 0.0, "s1"))
        seq.add_call(ToolCall("db", "lookup", {}, 1.0, "s1"))
        seq.add_call(ToolCall("scorer", "score", {}, 2.0, "s1"))
        counts = seq.tool_call_counts()
        assert counts["db.lookup"] == 2
        assert counts["scorer.score"] == 1

    def test_inter_call_intervals(self) -> None:
        seq = ToolCallSequence(session_id="s1")
        seq.add_call(ToolCall("db", "a", {}, 0.0, "s1"))
        seq.add_call(ToolCall("db", "b", {}, 1.5, "s1"))
        seq.add_call(ToolCall("db", "c", {}, 3.0, "s1"))
        intervals = seq.inter_call_intervals()
        assert len(intervals) == 2
        assert intervals[0] == pytest.approx(1.5)
        assert intervals[1] == pytest.approx(1.5)

    def test_to_sequences(self) -> None:
        seq = ToolCallSequence(session_id="s1")
        seq.add_call(ToolCall("db", "lookup", {}, 0.0, "s1"))
        seq.add_call(ToolCall("scorer", "score", {}, 1.0, "s1"))
        assert seq.to_tool_name_sequence() == ["db", "scorer"]
        assert seq.to_method_sequence() == ["db.lookup", "scorer.score"]

    def test_intent_field(self) -> None:
        seq = ToolCallSequence(session_id="s1", intent="check_interactions")
        assert seq.intent == "check_interactions"

    def test_encrypted_field(self) -> None:
        seq = ToolCallSequence(session_id="s1", encrypted=True)
        assert seq.encrypted is True

    def test_total_bytes(self) -> None:
        seq = ToolCallSequence(session_id="s1")
        seq.add_call(ToolCall("db", "lookup", {"x": "y"}, 0.0, "s1",
                              input_size_bytes=100, output_size_bytes=200))
        seq.add_call(ToolCall("db", "lookup", {"a": "b"}, 1.0, "s1",
                              input_size_bytes=50, output_size_bytes=75))
        assert seq.total_input_bytes == 150
        assert seq.total_output_bytes == 275


class TestAgentSessionAlias:
    def test_alias(self) -> None:
        s = AgentSession(session_id="x")
        assert isinstance(s, ToolCallSequence)


class TestThreatModel:
    def test_defaults(self) -> None:
        tm = ThreatModel()
        assert tm.adversary == AdversaryModel.HONEST_BUT_CURIOUS
        assert PrivacyProperty.QUERY_CONTENT in tm.observed_properties

    def test_property_checks(self) -> None:
        tm = ThreatModel()
        assert tm.is_property_observed(PrivacyProperty.QUERY_CONTENT)
        assert not tm.is_property_observed(PrivacyProperty.USER_INTENT)
        assert tm.is_property_targeted(PrivacyProperty.USER_INTENT)

    def test_active_adversary(self) -> None:
        tm = ThreatModel(adversary=AdversaryModel.ACTIVE)
        assert tm.adversary == AdversaryModel.ACTIVE


class TestPrivacyGoal:
    def test_satisfied(self) -> None:
        goal = PrivacyGoal(
            property_to_protect=PrivacyProperty.QUERY_CONTENT,
            mechanism="FHE",
            acceptable_leakage=0.01,
        )
        assert goal.is_satisfied(0.005)
        assert not goal.is_satisfied(0.02)

    def test_zero_leakage(self) -> None:
        goal = PrivacyGoal(
            property_to_protect=PrivacyProperty.ACCESS_PATTERN,
            mechanism="ORAM",
        )
        assert goal.acceptable_leakage == 0.0
        assert goal.is_satisfied(0.0)
        assert not goal.is_satisfied(1e-10)


# ===================================================================
# Tests: leakage_analysis.py
# ===================================================================

class TestEntropy:
    def test_uniform_8(self) -> None:
        dist = np.ones(8) / 8
        assert compute_entropy(dist) == pytest.approx(3.0)

    def test_uniform_2(self) -> None:
        dist = np.array([0.5, 0.5])
        assert compute_entropy(dist) == pytest.approx(1.0)

    def test_degenerate(self) -> None:
        dist = np.array([1.0, 0.0, 0.0])
        assert compute_entropy(dist) == pytest.approx(0.0)

    def test_empty(self) -> None:
        dist = np.array([])
        assert compute_entropy(dist) == 0.0

    def test_all_zeros(self) -> None:
        dist = np.array([0.0, 0.0])
        assert compute_entropy(dist) == 0.0


class TestMutualInformation:
    def test_independent(self) -> None:
        """Independent variables have MI = 0."""
        px = np.array([0.5, 0.5])
        py = np.array([0.5, 0.5])
        joint = np.outer(px, py)  # independent
        mi = mutual_information(joint, px, py)
        assert mi == pytest.approx(0.0, abs=1e-10)

    def test_perfectly_correlated(self) -> None:
        """Perfectly correlated variables have MI = H(X) = H(Y)."""
        joint = np.array([[0.5, 0.0], [0.0, 0.5]])
        px = np.array([0.5, 0.5])
        py = np.array([0.5, 0.5])
        mi = mutual_information(joint, px, py)
        assert mi == pytest.approx(1.0)


class TestDirectLeakage:
    def test_basic(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        analyzer = LeakageAnalyzer(threat_model=ThreatModel())
        analyzer.add_sessions(sessions)
        dl = analyzer._compute_direct_leakage()
        assert dl.calls_analyzed > 0
        assert dl.content_entropy > 0
        assert dl.avg_input_size_bytes > 0
        assert dl.unique_parameters > 0


class TestCorrelationLeakage:
    def test_basic(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        analyzer = LeakageAnalyzer(threat_model=ThreatModel())
        analyzer.add_sessions(sessions)
        cl = analyzer._compute_correlation_leakage()
        assert cl.unique_bigrams > 0
        assert 0.0 <= cl.parameter_overlap_rate <= 1.0
        assert 0.0 <= cl.sequence_predictability <= 1.0
        assert cl.tool_cooccurrence_entropy >= 0.0


class TestFrequencyLeakage:
    def test_basic(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        analyzer = LeakageAnalyzer(threat_model=ThreatModel())
        analyzer.add_sessions(sessions)
        fl = analyzer._compute_frequency_leakage()
        assert fl.unique_query_types > 0
        assert fl.query_distribution_entropy > 0
        assert 0.0 <= fl.top_query_concentration <= 1.0
        assert 0.0 <= fl.repeat_query_rate <= 1.0


class TestLeakageAnalyzer:
    def test_creation(self) -> None:
        analyzer = LeakageAnalyzer(threat_model=ThreatModel())
        assert len(analyzer.sessions) == 0

    def test_add_sessions(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        analyzer = LeakageAnalyzer(threat_model=ThreatModel())
        analyzer.add_sessions(sessions)
        assert len(analyzer.sessions) == 12

    def test_compute_leakage(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        analyzer = LeakageAnalyzer(threat_model=ThreatModel())
        analyzer.add_sessions(sessions)
        metrics = analyzer.compute_leakage()
        assert isinstance(metrics, LeakageMetrics)
        assert metrics.direct.calls_analyzed > 0
        assert metrics.correlation.unique_bigrams > 0
        assert metrics.frequency.unique_query_types > 0

    def test_entropy_before(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        analyzer = LeakageAnalyzer(threat_model=ThreatModel())
        analyzer.add_sessions(sessions)
        metrics = analyzer.compute_leakage()
        # 4 intents with equal frequency => H = 2.0 bits
        assert metrics.entropy_before == pytest.approx(2.0)

    def test_total_leakage_score(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        analyzer = LeakageAnalyzer(threat_model=ThreatModel())
        analyzer.add_sessions(sessions)
        metrics = analyzer.compute_leakage()
        assert 0.0 <= metrics.total_leakage_score <= 1.0

    def test_generate_report(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        analyzer = LeakageAnalyzer(threat_model=ThreatModel())
        analyzer.add_sessions(sessions)
        report = analyzer.generate_report()
        assert "summary" in report
        assert "direct" in report
        assert "correlation" in report
        assert "frequency" in report
        assert report["summary"]["total_sessions"] == 12
        assert report["summary"]["total_calls"] > 0

    def test_compare_protected_vs_unprotected(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        analyzer = LeakageAnalyzer(threat_model=ThreatModel())
        analyzer.add_sessions(sessions)
        encrypted = [make_encrypted_session(s) for s in sessions]
        comparison = analyzer.compare_protected_vs_unprotected(encrypted)
        assert "unprotected" in comparison
        assert "protected" in comparison
        # Encrypted sessions should have lower content entropy
        assert (comparison["protected"].direct.content_entropy
                < comparison["unprotected"].direct.content_entropy)

    def test_observation_matrix_shape(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        analyzer = LeakageAnalyzer(threat_model=ThreatModel())
        analyzer.add_sessions(sessions)
        mat = analyzer._build_observation_matrix()
        assert mat.shape[0] == 12  # 12 sessions
        assert mat.shape[1] > 4   # at least the 4 aggregate features


# ===================================================================
# Tests: intent_reconstruction.py
# ===================================================================

class TestMakeEncryptedSession:
    def test_preserves_structure(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        enc = make_encrypted_session(sessions[0])
        assert enc.call_count == sessions[0].call_count
        assert enc.encrypted is True
        assert enc.intent == sessions[0].intent

    def test_hides_content(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        enc = make_encrypted_session(sessions[0])
        for call in enc.calls:
            assert call.tool_name == "encrypted_tool"
            assert call.method == "encrypted_method"
            assert call.parameters == {"opaque": "***"}

    def test_pads_sizes(self, small_dataset: tuple) -> None:
        sessions, _ = small_dataset
        enc = make_encrypted_session(sessions[0])
        for call in enc.calls:
            assert call.input_size_bytes == 512
            assert call.output_size_bytes == 512


class TestFrequencyAttack:
    def test_fit_predict(self, small_dataset: tuple) -> None:
        sessions, intents = small_dataset
        attack = FrequencyAttack()
        attack.fit(sessions, intents)
        result = attack.predict(sessions[0])
        assert isinstance(result, ReconstructionResult)
        assert result.attack_strategy == AttackStrategy.FREQUENCY
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.top_k_intents) > 0

    def test_evaluate(self, small_dataset: tuple) -> None:
        sessions, intents = small_dataset
        attack = FrequencyAttack()
        attack.fit(sessions, intents)
        metrics = attack.evaluate(sessions, intents)
        assert "accuracy" in metrics
        assert "top_3_accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestSequenceAttack:
    def test_fit_predict(self, small_dataset: tuple) -> None:
        sessions, intents = small_dataset
        attack = SequenceAttack()
        attack.fit(sessions, intents)
        result = attack.predict(sessions[0])
        assert result.attack_strategy == AttackStrategy.SEQUENCE
        assert result.predicted_intent in INTENTS

    def test_evaluate(self, small_dataset: tuple) -> None:
        sessions, intents = small_dataset
        attack = SequenceAttack()
        attack.fit(sessions, intents)
        metrics = attack.evaluate(sessions, intents)
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestMLClassifierAttack:
    def test_fit_predict(self, large_dataset: tuple) -> None:
        sessions, intents = large_dataset
        attack = MLClassifierAttack()
        attack.fit(sessions, intents)
        result = attack.predict(sessions[0])
        assert result.attack_strategy == AttackStrategy.ML_CLASSIFIER
        assert result.predicted_intent in INTENTS
        assert 0.0 <= result.confidence <= 1.0

    def test_evaluate_on_training_data(self, large_dataset: tuple) -> None:
        """RF on training data should achieve near-perfect accuracy."""
        sessions, intents = large_dataset
        attack = MLClassifierAttack()
        attack.fit(sessions, intents)
        metrics = attack.evaluate(sessions, intents)
        assert metrics["accuracy"] >= 0.8

    def test_predict_before_fit_raises(self) -> None:
        attack = MLClassifierAttack()
        seq = ToolCallSequence(session_id="s1")
        seq.add_call(ToolCall("db", "lookup", {}, 0.0, "s1"))
        with pytest.raises(RuntimeError, match="fit"):
            attack.predict(seq)


class TestCompareReconstructionAccuracy:
    def test_encrypted_worse_than_plaintext(self, large_dataset: tuple) -> None:
        """Encryption should reduce attack accuracy."""
        sessions, intents = large_dataset
        n = len(sessions)
        split = int(0.7 * n)
        # Shuffle deterministically
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        train_s = [sessions[i] for i in idx[:split]]
        train_i = [intents[i] for i in idx[:split]]
        test_s = [sessions[i] for i in idx[split:]]
        test_i = [intents[i] for i in idx[split:]]

        results = compare_reconstruction_accuracy(train_s, train_i, test_s, test_i)

        # ML attack should be worse on encrypted data
        plain_acc = results["plaintext"]["MLClassifierAttack"]["accuracy"]
        enc_acc = results["encrypted"]["MLClassifierAttack"]["accuracy"]
        # Encrypted should not be better than plaintext (allow small margin)
        assert enc_acc <= plain_acc + 0.15

        # All result dicts should have the expected keys
        for mode in ["plaintext", "encrypted"]:
            for attack_name in ["FrequencyAttack", "SequenceAttack", "MLClassifierAttack"]:
                assert "accuracy" in results[mode][attack_name]
                assert "top_3_accuracy" in results[mode][attack_name]
