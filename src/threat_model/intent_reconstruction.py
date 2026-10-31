"""Simulate intent-reconstruction attacks by an honest-but-curious tool server.

An honest-but-curious server observes a sequence of tool calls and attempts
to infer the user's high-level intent (e.g., "the user is checking drug
interactions for a patient on antidepressants and blood thinners").

This module implements three attack strategies of increasing power:

1. **FrequencyAttack** — build a per-intent frequency profile over
   (tool, method) invocations and match new sessions via cosine
   similarity.
2. **SequenceAttack** — additionally leverage n-gram transition
   patterns in the call ordering.
3. **MLClassifierAttack** — train a Random Forest (or other sklearn
   classifier) on the full feature vector for maximum accuracy.

Each attack also supports an *encrypted* evaluation mode where the
server can only observe metadata (timing, size, call count) but not
parameter content.  Comparing plaintext vs. encrypted accuracy
quantifies the privacy gain from cryptographic protection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from src.threat_model.definitions import ToolCallSequence


# ===================================================================
# Enums & result types
# ===================================================================

class AttackStrategy(Enum):
    """Attack strategies for intent reconstruction."""

    FREQUENCY = auto()
    SEQUENCE = auto()
    ML_CLASSIFIER = auto()


@dataclass
class ReconstructionResult:
    """Result of an intent-reconstruction attack on a single session.

    Attributes
    ----------
    predicted_intent : str
        The server's best guess at the user's goal.
    confidence : float
        Confidence score in [0, 1].
    top_k_intents : list[tuple[str, float]]
        Top-k predictions with associated scores.
    attack_strategy : AttackStrategy
        Which strategy produced this result.
    """

    predicted_intent: str
    confidence: float
    top_k_intents: list[tuple[str, float]]
    attack_strategy: AttackStrategy


# ===================================================================
# Feature extraction helpers
# ===================================================================

def _extract_frequency_features(
    session: ToolCallSequence,
    vocabulary: list[str],
) -> NDArray[np.float64]:
    """Extract a frequency-count feature vector from a session.

    One dimension per element in *vocabulary* (each element is a
    ``"tool.method"`` string), plus aggregate features.
    """
    counts = session.tool_call_counts()
    vec = np.zeros(len(vocabulary) + 4, dtype=np.float64)
    for i, method in enumerate(vocabulary):
        vec[i] = counts.get(method, 0)
    vec[len(vocabulary)] = session.call_count
    vec[len(vocabulary) + 1] = session.duration
    intervals = session.inter_call_intervals()
    vec[len(vocabulary) + 2] = float(np.mean(intervals)) if intervals else 0.0
    vec[len(vocabulary) + 3] = session.total_input_bytes
    return vec


def _extract_ngram_features(
    session: ToolCallSequence,
    vocabulary: list[str],
    ngram_vocab: list[tuple[str, ...]],
    n: int = 2,
) -> NDArray[np.float64]:
    """Extract frequency + n-gram features from a session."""
    freq = _extract_frequency_features(session, vocabulary)
    methods = session.to_method_sequence()
    ngram_counts = Counter(
        tuple(methods[i : i + n]) for i in range(len(methods) - n + 1)
    )
    ngram_vec = np.zeros(len(ngram_vocab), dtype=np.float64)
    for i, ng in enumerate(ngram_vocab):
        ngram_vec[i] = ngram_counts.get(ng, 0)
    return np.concatenate([freq, ngram_vec])


def _build_vocabulary(sessions: list[ToolCallSequence]) -> list[str]:
    """Collect all ``"tool.method"`` strings seen in training data."""
    methods: set[str] = set()
    for s in sessions:
        for c in s.calls:
            methods.add(f"{c.tool_name}.{c.method}")
    return sorted(methods)


def _build_ngram_vocabulary(
    sessions: list[ToolCallSequence],
    n: int = 2,
) -> list[tuple[str, ...]]:
    """Collect all n-gram tuples seen in training data."""
    ngrams: set[tuple[str, ...]] = set()
    for s in sessions:
        methods = s.to_method_sequence()
        for i in range(len(methods) - n + 1):
            ngrams.add(tuple(methods[i : i + n]))
    return sorted(ngrams)


def _cosine_similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ===================================================================
# Base class
# ===================================================================

class IntentReconstructor(ABC):
    """Abstract base for intent-reconstruction attacks."""

    @abstractmethod
    def fit(
        self,
        sessions: list[ToolCallSequence],
        intents: list[str],
    ) -> None:
        """Train on labelled (session, intent) pairs."""
        ...

    @abstractmethod
    def predict(self, session: ToolCallSequence) -> ReconstructionResult:
        """Predict the intent of a single session."""
        ...

    def evaluate(
        self,
        test_sessions: list[ToolCallSequence],
        true_intents: list[str],
    ) -> dict[str, float]:
        """Evaluate reconstruction accuracy on a test set.

        Returns
        -------
        dict[str, float]
            ``accuracy``, ``top_3_accuracy``, ``precision_macro``,
            ``recall_macro``.
        """
        predictions = [self.predict(s) for s in test_sessions]
        pred_intents = [r.predicted_intent for r in predictions]

        accuracy = accuracy_score(true_intents, pred_intents)

        # Top-3 accuracy
        top3_correct = 0
        for result, true in zip(predictions, true_intents):
            top3_labels = [label for label, _ in result.top_k_intents[:3]]
            if true in top3_labels:
                top3_correct += 1
        top3_acc = top3_correct / len(true_intents) if true_intents else 0.0

        labels = sorted(set(true_intents) | set(pred_intents))
        precision = precision_score(
            true_intents, pred_intents, labels=labels,
            average="macro", zero_division=0,
        )
        recall = recall_score(
            true_intents, pred_intents, labels=labels,
            average="macro", zero_division=0,
        )

        return {
            "accuracy": float(accuracy),
            "top_3_accuracy": float(top3_acc),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
        }


# ===================================================================
# FrequencyAttack
# ===================================================================

class FrequencyAttack(IntentReconstructor):
    """Infer intent from tool/method usage frequencies.

    The simplest attack: build a mean frequency profile for each
    intent class, then match new sessions via cosine similarity.
    """

    def __init__(self) -> None:
        self._intent_profiles: dict[str, NDArray[np.float64]] = {}
        self._vocabulary: list[str] = []

    def fit(
        self,
        sessions: list[ToolCallSequence],
        intents: list[str],
    ) -> None:
        self._vocabulary = _build_vocabulary(sessions)
        groups: dict[str, list[NDArray[np.float64]]] = defaultdict(list)
        for session, intent in zip(sessions, intents):
            vec = _extract_frequency_features(session, self._vocabulary)
            groups[intent].append(vec)
        for intent, vecs in groups.items():
            self._intent_profiles[intent] = np.mean(vecs, axis=0)

    def predict(self, session: ToolCallSequence) -> ReconstructionResult:
        vec = _extract_frequency_features(session, self._vocabulary)
        scores: list[tuple[str, float]] = []
        for intent, profile in self._intent_profiles.items():
            sim = _cosine_similarity(vec, profile)
            scores.append((intent, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        best_intent, best_score = scores[0] if scores else ("unknown", 0.0)
        return ReconstructionResult(
            predicted_intent=best_intent,
            confidence=best_score,
            top_k_intents=scores[:5],
            attack_strategy=AttackStrategy.FREQUENCY,
        )


# ===================================================================
# SequenceAttack
# ===================================================================

class SequenceAttack(IntentReconstructor):
    """Infer intent from call ordering via bigram patterns.

    Combines frequency features with bigram transition counts for a
    richer representation.
    """

    def __init__(self, n_gram_size: int = 2) -> None:
        self.n_gram_size = n_gram_size
        self._vocabulary: list[str] = []
        self._ngram_vocab: list[tuple[str, ...]] = []
        self._intent_profiles: dict[str, NDArray[np.float64]] = {}

    def fit(
        self,
        sessions: list[ToolCallSequence],
        intents: list[str],
    ) -> None:
        self._vocabulary = _build_vocabulary(sessions)
        self._ngram_vocab = _build_ngram_vocabulary(sessions, self.n_gram_size)
        groups: dict[str, list[NDArray[np.float64]]] = defaultdict(list)
        for session, intent in zip(sessions, intents):
            vec = _extract_ngram_features(
                session, self._vocabulary, self._ngram_vocab, self.n_gram_size,
            )
            groups[intent].append(vec)
        for intent, vecs in groups.items():
            self._intent_profiles[intent] = np.mean(vecs, axis=0)

    def predict(self, session: ToolCallSequence) -> ReconstructionResult:
        vec = _extract_ngram_features(
            session, self._vocabulary, self._ngram_vocab, self.n_gram_size,
        )
        scores: list[tuple[str, float]] = []
        for intent, profile in self._intent_profiles.items():
            sim = _cosine_similarity(vec, profile)
            scores.append((intent, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        best_intent, best_score = scores[0] if scores else ("unknown", 0.0)
        return ReconstructionResult(
            predicted_intent=best_intent,
            confidence=best_score,
            top_k_intents=scores[:5],
            attack_strategy=AttackStrategy.SEQUENCE,
        )


# ===================================================================
# MLClassifierAttack
# ===================================================================

class MLClassifierAttack(IntentReconstructor):
    """Train an sklearn classifier for maximum reconstruction accuracy.

    Uses a Random Forest by default, operating on the combined
    frequency + n-gram feature vector.  Represents the strongest
    realistic attack a server could mount with standard ML tools.
    """

    def __init__(self, classifier_type: str = "random_forest", n_gram_size: int = 2) -> None:
        self.classifier_type = classifier_type
        self.n_gram_size = n_gram_size
        self._model: RandomForestClassifier | None = None
        self._label_encoder = LabelEncoder()
        self._vocabulary: list[str] = []
        self._ngram_vocab: list[tuple[str, ...]] = []

    def fit(
        self,
        sessions: list[ToolCallSequence],
        intents: list[str],
    ) -> None:
        self._vocabulary = _build_vocabulary(sessions)
        self._ngram_vocab = _build_ngram_vocabulary(sessions, self.n_gram_size)
        X = np.array([
            _extract_ngram_features(s, self._vocabulary, self._ngram_vocab, self.n_gram_size)
            for s in sessions
        ])
        y = self._label_encoder.fit_transform(intents)
        self._model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
        )
        self._model.fit(X, y)

    def predict(self, session: ToolCallSequence) -> ReconstructionResult:
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        vec = _extract_ngram_features(
            session, self._vocabulary, self._ngram_vocab, self.n_gram_size,
        )
        proba = self._model.predict_proba(vec.reshape(1, -1))[0]
        class_names = self._label_encoder.classes_
        scored = sorted(
            zip(class_names, proba), key=lambda x: x[1], reverse=True,
        )
        best_intent = str(scored[0][0])
        best_conf = float(scored[0][1])
        return ReconstructionResult(
            predicted_intent=best_intent,
            confidence=best_conf,
            top_k_intents=[(str(n), float(p)) for n, p in scored[:5]],
            attack_strategy=AttackStrategy.ML_CLASSIFIER,
        )


# ===================================================================
# Encrypted-mode comparison helper
# ===================================================================

def make_encrypted_session(session: ToolCallSequence) -> ToolCallSequence:
    """Create an encrypted version of a session.

    Simulates what the server observes when the agent uses FHE/PIR:
    - Tool name and method are replaced with generic labels.
    - Parameter values are replaced with opaque hashes of fixed size.
    - Input/output sizes are padded to a common bucket size.
    - Timestamps are preserved (timing side-channel not yet mitigated).

    Parameters
    ----------
    session : ToolCallSequence
        Original plaintext session.

    Returns
    -------
    ToolCallSequence
        Encrypted-equivalent session with metadata-only observables.
    """
    from src.threat_model.definitions import ToolCall, ToolCallType

    BUCKET_SIZE = 512  # pad all sizes to this
    encrypted_calls = []
    for call in session.calls:
        encrypted_calls.append(ToolCall(
            tool_name="encrypted_tool",
            method="encrypted_method",
            parameters={"opaque": "***"},
            timestamp=call.timestamp,
            session_id=call.session_id,
            call_type=call.call_type,
            input_size_bytes=BUCKET_SIZE,
            output=None,
            output_size_bytes=BUCKET_SIZE,
        ))
    return ToolCallSequence(
        session_id=session.session_id,
        calls=encrypted_calls,
        intent=session.intent,
        encrypted=True,
    )


def compare_reconstruction_accuracy(
    train_sessions: list[ToolCallSequence],
    train_intents: list[str],
    test_sessions: list[ToolCallSequence],
    test_intents: list[str],
) -> dict[str, dict[str, float]]:
    """Run all three attacks on plaintext and encrypted sessions.

    Returns a nested dict::

        {
            "plaintext": {
                "FrequencyAttack": {"accuracy": ..., ...},
                "SequenceAttack": {"accuracy": ..., ...},
                "MLClassifierAttack": {"accuracy": ..., ...},
            },
            "encrypted": { ... same structure ... },
        }
    """
    enc_train = [make_encrypted_session(s) for s in train_sessions]
    enc_test = [make_encrypted_session(s) for s in test_sessions]

    results: dict[str, dict[str, dict[str, float]]] = {
        "plaintext": {},
        "encrypted": {},
    }

    attacks: list[tuple[str, IntentReconstructor]] = [
        ("FrequencyAttack", FrequencyAttack()),
        ("SequenceAttack", SequenceAttack()),
        ("MLClassifierAttack", MLClassifierAttack()),
    ]

    for name, attack in attacks:
        attack.fit(train_sessions, train_intents)
        results["plaintext"][name] = attack.evaluate(test_sessions, test_intents)

    for name, attack in [
        ("FrequencyAttack", FrequencyAttack()),
        ("SequenceAttack", SequenceAttack()),
        ("MLClassifierAttack", MLClassifierAttack()),
    ]:
        attack.fit(enc_train, train_intents)
        results["encrypted"][name] = attack.evaluate(enc_test, test_intents)

    return results
