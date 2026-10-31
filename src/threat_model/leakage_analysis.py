"""Quantify information leakage from tool-call sequences.

This module models what an honest-but-curious tool server learns by
observing plaintext (unprotected) and ciphertext (protected) tool calls.
It computes three complementary leakage metrics:

1. **Direct leakage** — information revealed by each individual call
   (input content, timing, payload size).
2. **Correlation leakage** — information revealed by *patterns across*
   calls within a session (tool co-occurrence, parameter overlap).
3. **Frequency leakage** — information revealed by repeated query
   patterns over multiple sessions (query-type distribution skew).

The module also provides entropy primitives (Shannon entropy, mutual
information) and a ``LeakageAnalyzer`` class that combines all three
channels into a single ``LeakageReport``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.threat_model.definitions import (
    PrivacyProperty,
    ThreatModel,
    ToolCall,
    ToolCallSequence,
)


# ===================================================================
# Entropy primitives
# ===================================================================

def compute_entropy(distribution: NDArray[np.float64]) -> float:
    """Compute Shannon entropy *H(X)* in bits.

    Parameters
    ----------
    distribution : array-like
        Probability distribution over discrete outcomes.  Must be
        non-negative and sum to 1 (within floating-point tolerance).

    Returns
    -------
    float
        Entropy in bits.  Returns 0.0 for a degenerate (single-
        outcome) distribution.
    """
    p = np.asarray(distribution, dtype=np.float64).ravel()
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return -float(np.sum(p * np.log2(p)))


def mutual_information(
    joint: NDArray[np.float64],
    marginal_x: NDArray[np.float64],
    marginal_y: NDArray[np.float64],
) -> float:
    """Compute mutual information *I(X; Y)* in bits.

    Uses the identity  ``I(X;Y) = H(X) + H(Y) - H(X,Y)``.

    Parameters
    ----------
    joint : ndarray, shape (|X|, |Y|)
        Joint probability table  ``P(X, Y)``.
    marginal_x : ndarray, shape (|X|,)
        Marginal distribution ``P(X)``.
    marginal_y : ndarray, shape (|Y|,)
        Marginal distribution ``P(Y)``.

    Returns
    -------
    float
        Non-negative mutual information in bits.
    """
    h_x = compute_entropy(marginal_x)
    h_y = compute_entropy(marginal_y)
    h_xy = compute_entropy(joint.ravel())
    return max(0.0, h_x + h_y - h_xy)


def _normalize_counts(counts: dict[str, int]) -> NDArray[np.float64]:
    """Convert a counter dict to a probability vector."""
    vals = np.array(list(counts.values()), dtype=np.float64)
    total = vals.sum()
    if total == 0:
        return vals
    return vals / total


# ===================================================================
# Leakage metrics dataclass
# ===================================================================

@dataclass
class DirectLeakage:
    """Per-call leakage from individual tool invocations.

    Attributes
    ----------
    content_entropy : float
        Entropy of the parameter-value distribution (bits).  Higher
        entropy means each call is more informative to the server.
    avg_input_size_bytes : float
        Mean serialized input size — observable even under encryption
        if the ciphertext is not padded to a fixed length.
    timing_entropy : float
        Entropy of inter-call timing intervals (bits).  Non-trivial
        timing patterns can leak information about the computation
        graph on the client side.
    unique_parameters : int
        Count of distinct parameter keys observed across all calls.
    calls_analyzed : int
        Number of calls that contributed to this measurement.
    """

    content_entropy: float = 0.0
    avg_input_size_bytes: float = 0.0
    timing_entropy: float = 0.0
    unique_parameters: int = 0
    calls_analyzed: int = 0


@dataclass
class CorrelationLeakage:
    """Cross-call leakage from patterns within a session.

    Attributes
    ----------
    tool_cooccurrence_entropy : float
        Entropy of the tool co-occurrence matrix (bits).  Low entropy
        means the server can predict the next tool call given the
        previous ones.
    parameter_overlap_rate : float
        Fraction of consecutive call-pairs that share at least one
        parameter value (e.g., the same drug name appearing in a
        lookup followed by a scoring call).  Range [0, 1].
    sequence_predictability : float
        Fraction of bigram tool transitions that match the single most
        common transition.  1.0 = perfectly predictable; 1/|T|^2 =
        uniform.  Range [0, 1].
    unique_bigrams : int
        Number of distinct consecutive tool-name pairs observed.
    """

    tool_cooccurrence_entropy: float = 0.0
    parameter_overlap_rate: float = 0.0
    sequence_predictability: float = 0.0
    unique_bigrams: int = 0


@dataclass
class FrequencyLeakage:
    """Leakage from repeated query patterns across sessions.

    Attributes
    ----------
    query_distribution_entropy : float
        Entropy of the distribution over distinct query types (bits).
        Lower entropy means the server can narrow down which queries
        are most popular.
    repeat_query_rate : float
        Fraction of queries that are exact repeats of a previous query
        in the corpus.  High repeat rate aids frequency analysis.
    top_query_concentration : float
        Fraction of total queries accounted for by the single most
        common query.  Range [0, 1].
    unique_query_types : int
        Number of distinct (tool_name, method, param_hash) triples.
    """

    query_distribution_entropy: float = 0.0
    repeat_query_rate: float = 0.0
    top_query_concentration: float = 0.0
    unique_query_types: int = 0


@dataclass
class LeakageMetrics:
    """Aggregated leakage measurements across all channels.

    Attributes
    ----------
    mutual_information : float
        I(Query; UserAttribute) in bits.
    reidentification_prob : float
        Pr[server correctly links a session to a user].
    entropy_before : float
        H(Intent) — prior uncertainty about user intent.
    entropy_after : float
        H(Intent | Observations) — residual uncertainty after the
        server observes all tool calls.
    direct : DirectLeakage
        Per-call leakage breakdown.
    correlation : CorrelationLeakage
        Cross-call leakage breakdown.
    frequency : FrequencyLeakage
        Repeat-pattern leakage breakdown.
    """

    mutual_information: float = 0.0
    reidentification_prob: float = 0.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    direct: DirectLeakage = field(default_factory=DirectLeakage)
    correlation: CorrelationLeakage = field(default_factory=CorrelationLeakage)
    frequency: FrequencyLeakage = field(default_factory=FrequencyLeakage)

    @property
    def leakage_bits(self) -> float:
        """Total information leakage in bits = H(before) - H(after)."""
        return self.entropy_before - self.entropy_after

    @property
    def total_leakage_score(self) -> float:
        """Normalized composite leakage score in [0, 1].

        Averages the three channel-level indicators so that higher
        means more leakage.
        """
        direct_score = min(self.direct.content_entropy / 8.0, 1.0) if self.direct.content_entropy > 0 else 0.0
        corr_score = self.correlation.parameter_overlap_rate
        freq_score = self.frequency.top_query_concentration
        return (direct_score + corr_score + freq_score) / 3.0


# ===================================================================
# Helper: parameter fingerprinting
# ===================================================================

def _param_hash(params: dict[str, Any]) -> str:
    """Deterministic hash of a parameter dict for dedup and counting."""
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _param_values_set(params: dict[str, Any]) -> set[str]:
    """Flatten parameter values into a set of strings for overlap detection."""
    vals: set[str] = set()
    for v in params.values():
        if isinstance(v, (list, tuple)):
            vals.update(str(x) for x in v)
        else:
            vals.add(str(v))
    return vals


# ===================================================================
# LeakageAnalyzer
# ===================================================================

class LeakageAnalyzer:
    """Analyze information leakage for a given threat model.

    The analyzer ingests a collection of ``ToolCallSequence`` objects
    (sessions) and quantifies how much information the honest-but-
    curious server gains about user-private attributes from observing
    the calls.

    Example
    -------
    >>> analyzer = LeakageAnalyzer(threat_model=ThreatModel())
    >>> analyzer.add_sessions(sessions)
    >>> metrics = analyzer.compute_leakage()
    >>> print(f"Server learns {metrics.leakage_bits:.2f} bits per session")
    """

    def __init__(self, threat_model: ThreatModel) -> None:
        self.threat_model = threat_model
        self.sessions: list[ToolCallSequence] = []
        self._observation_matrix: NDArray[np.float64] | None = None

    def add_sessions(self, sessions: list[ToolCallSequence]) -> None:
        """Register observed tool-call sessions for analysis."""
        self.sessions.extend(sessions)
        self._observation_matrix = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _collect_all_feature_names(self) -> list[str]:
        """Derive the global feature vocabulary from all sessions."""
        all_methods: set[str] = set()
        for s in self.sessions:
            for c in s.calls:
                all_methods.add(f"{c.tool_name}.{c.method}")
        method_features = sorted(all_methods)
        # Extra features: call_count, duration, mean_interval, input_bytes
        return method_features + [
            "__call_count__",
            "__duration__",
            "__mean_interval__",
            "__total_input_bytes__",
        ]

    def _build_observation_matrix(self) -> NDArray[np.float64]:
        """Build a (n_sessions x n_features) observation matrix.

        Feature columns:
        - Per-method call counts (one column per unique tool.method)
        - Total call count
        - Session duration
        - Mean inter-call interval
        - Total input bytes
        """
        if self._observation_matrix is not None:
            return self._observation_matrix

        feature_names = self._collect_all_feature_names()
        method_features = [f for f in feature_names if not f.startswith("__")]
        n = len(self.sessions)
        d = len(feature_names)
        mat = np.zeros((n, d), dtype=np.float64)

        for i, session in enumerate(self.sessions):
            counts = session.tool_call_counts()
            for j, mf in enumerate(method_features):
                mat[i, j] = counts.get(mf, 0)

            offset = len(method_features)
            mat[i, offset] = session.call_count
            mat[i, offset + 1] = session.duration
            intervals = session.inter_call_intervals()
            mat[i, offset + 2] = float(np.mean(intervals)) if intervals else 0.0
            mat[i, offset + 3] = session.total_input_bytes

        self._observation_matrix = mat
        return mat

    # ------------------------------------------------------------------
    # Direct leakage
    # ------------------------------------------------------------------

    def _compute_direct_leakage(self) -> DirectLeakage:
        """Quantify per-call information leakage."""
        all_calls: list[ToolCall] = []
        for s in self.sessions:
            all_calls.extend(s.calls)
        if not all_calls:
            return DirectLeakage()

        # Content entropy: distribution over distinct parameter hashes
        param_hashes = [_param_hash(c.parameters) for c in all_calls]
        hash_counts = Counter(param_hashes)
        content_dist = _normalize_counts(hash_counts)
        content_ent = compute_entropy(content_dist)

        # Average input size
        avg_size = float(np.mean([c.input_size_bytes for c in all_calls]))

        # Timing entropy: bucket inter-call intervals into 20 bins
        all_intervals: list[float] = []
        for s in self.sessions:
            all_intervals.extend(s.inter_call_intervals())
        if all_intervals:
            hist, _ = np.histogram(all_intervals, bins=20, density=True)
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            timing_ent = compute_entropy(hist)
        else:
            timing_ent = 0.0

        # Unique parameter keys
        all_keys: set[str] = set()
        for c in all_calls:
            all_keys.update(c.parameters.keys())

        return DirectLeakage(
            content_entropy=content_ent,
            avg_input_size_bytes=avg_size,
            timing_entropy=timing_ent,
            unique_parameters=len(all_keys),
            calls_analyzed=len(all_calls),
        )

    # ------------------------------------------------------------------
    # Correlation leakage
    # ------------------------------------------------------------------

    def _compute_correlation_leakage(self) -> CorrelationLeakage:
        """Quantify cross-call correlations within sessions."""
        bigram_counter: Counter[tuple[str, str]] = Counter()
        overlap_pairs = 0
        total_pairs = 0

        for session in self.sessions:
            methods = session.to_method_sequence()
            for k in range(len(methods) - 1):
                bigram_counter[(methods[k], methods[k + 1])] += 1

            for k in range(len(session.calls) - 1):
                vals_a = _param_values_set(session.calls[k].parameters)
                vals_b = _param_values_set(session.calls[k + 1].parameters)
                total_pairs += 1
                if vals_a & vals_b:
                    overlap_pairs += 1

        # Co-occurrence entropy
        bigram_dist = _normalize_counts(
            {str(k): v for k, v in bigram_counter.items()}
        )
        cooccurrence_ent = compute_entropy(bigram_dist)

        # Parameter overlap rate
        overlap_rate = overlap_pairs / total_pairs if total_pairs > 0 else 0.0

        # Sequence predictability: max-bigram-fraction
        total_bigrams = sum(bigram_counter.values())
        if total_bigrams > 0:
            max_bigram = bigram_counter.most_common(1)[0][1]
            predictability = max_bigram / total_bigrams
        else:
            predictability = 0.0

        return CorrelationLeakage(
            tool_cooccurrence_entropy=cooccurrence_ent,
            parameter_overlap_rate=overlap_rate,
            sequence_predictability=predictability,
            unique_bigrams=len(bigram_counter),
        )

    # ------------------------------------------------------------------
    # Frequency leakage
    # ------------------------------------------------------------------

    def _compute_frequency_leakage(self) -> FrequencyLeakage:
        """Quantify leakage from repeated query patterns."""
        # A "query type" is (tool_name, method, param_hash)
        query_type_counter: Counter[str] = Counter()
        for session in self.sessions:
            for call in session.calls:
                qt = f"{call.tool_name}.{call.method}:{_param_hash(call.parameters)}"
                query_type_counter[qt] += 1

        total_queries = sum(query_type_counter.values())
        if total_queries == 0:
            return FrequencyLeakage()

        # Distribution entropy
        dist = _normalize_counts(query_type_counter)
        q_entropy = compute_entropy(dist)

        # Repeat rate: queries that appeared more than once
        repeated = sum(c for c in query_type_counter.values() if c > 1)
        repeat_rate = repeated / total_queries

        # Top-query concentration
        top_count = query_type_counter.most_common(1)[0][1]
        top_concentration = top_count / total_queries

        return FrequencyLeakage(
            query_distribution_entropy=q_entropy,
            repeat_query_rate=repeat_rate,
            top_query_concentration=top_concentration,
            unique_query_types=len(query_type_counter),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_leakage(
        self,
        user_attributes: NDArray[np.float64] | None = None,
    ) -> LeakageMetrics:
        """Compute all leakage metrics across the registered sessions.

        Parameters
        ----------
        user_attributes : ndarray, optional
            Ground-truth attribute matrix (n_sessions x n_attributes)
            for computing mutual information and re-identification
            probability.  If ``None``, only structure-based metrics
            are computed.

        Returns
        -------
        LeakageMetrics
        """
        direct = self._compute_direct_leakage()
        correlation = self._compute_correlation_leakage()
        frequency = self._compute_frequency_leakage()

        # Intent entropy (prior): computed from session intent labels
        intent_counts = Counter(s.intent for s in self.sessions if s.intent)
        if intent_counts:
            intent_dist = _normalize_counts(intent_counts)
            entropy_before = compute_entropy(intent_dist)
        else:
            entropy_before = 0.0

        # Observation-based mutual information and re-identification
        mi = 0.0
        reid_prob = 0.0
        entropy_after = entropy_before  # default: no reduction

        if user_attributes is not None and len(self.sessions) > 0:
            obs = self._build_observation_matrix()
            # Discretize observations into 10 bins per feature for MI
            obs_discrete = np.zeros_like(obs, dtype=np.int64)
            for col in range(obs.shape[1]):
                col_data = obs[:, col]
                if col_data.max() > col_data.min():
                    obs_discrete[:, col] = np.digitize(
                        col_data,
                        bins=np.linspace(col_data.min(), col_data.max(), 11)[1:],
                    )

            # Build joint distribution of (intent, observation-cluster)
            from sklearn.cluster import KMeans

            n_clusters = min(len(set(s.intent for s in self.sessions)), len(self.sessions), 10)
            if n_clusters >= 2:
                km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
                cluster_labels = km.fit_predict(obs)

                # Joint distribution
                intents = [s.intent for s in self.sessions]
                intent_set = sorted(set(intents))
                intent_idx = {name: i for i, name in enumerate(intent_set)}

                joint = np.zeros((len(intent_set), n_clusters), dtype=np.float64)
                for si, sess in enumerate(self.sessions):
                    ii = intent_idx.get(sess.intent, 0)
                    ci = cluster_labels[si]
                    joint[ii, ci] += 1
                joint /= joint.sum()

                marginal_intent = joint.sum(axis=1)
                marginal_cluster = joint.sum(axis=0)
                mi = mutual_information(joint, marginal_intent, marginal_cluster)

                # H(Intent | Cluster) = H(Intent, Cluster) - H(Cluster)
                h_joint = compute_entropy(joint.ravel())
                h_cluster = compute_entropy(marginal_cluster)
                entropy_after = max(0.0, h_joint - h_cluster)

                # Re-identification: fraction of sessions where the cluster
                # uniquely (or near-uniquely) identifies the intent
                correct = 0
                for ci in range(n_clusters):
                    col = joint[:, ci]
                    if col.sum() > 0:
                        best_intent = int(np.argmax(col))
                        # Count sessions in this cluster assigned to best intent
                        for si, sess in enumerate(self.sessions):
                            if cluster_labels[si] == ci and intent_idx.get(sess.intent, -1) == best_intent:
                                correct += 1
                reid_prob = correct / len(self.sessions) if self.sessions else 0.0

        return LeakageMetrics(
            mutual_information=mi,
            reidentification_prob=reid_prob,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            direct=direct,
            correlation=correlation,
            frequency=frequency,
        )

    def compare_protected_vs_unprotected(
        self,
        protected_sessions: list[ToolCallSequence],
        user_attributes: NDArray[np.float64] | None = None,
    ) -> dict[str, LeakageMetrics]:
        """Compare leakage with and without cryptographic protection.

        Parameters
        ----------
        protected_sessions : list[ToolCallSequence]
            Sessions where tool calls were encrypted (only metadata
            such as timing and size is visible to the server).
        user_attributes : ndarray, optional
            Ground-truth user attributes for MI computation.

        Returns
        -------
        dict[str, LeakageMetrics]
            Keys ``"unprotected"`` and ``"protected"``.
        """
        # Unprotected baseline
        unprotected_metrics = self.compute_leakage(user_attributes)

        # Protected: analyze the encrypted sessions
        protected_analyzer = LeakageAnalyzer(threat_model=self.threat_model)
        protected_analyzer.add_sessions(protected_sessions)
        protected_metrics = protected_analyzer.compute_leakage(user_attributes)

        return {
            "unprotected": unprotected_metrics,
            "protected": protected_metrics,
        }

    def generate_report(self) -> dict[str, Any]:
        """Generate a structured leakage report suitable for serialization.

        Returns
        -------
        dict
            Nested dict mirroring the ``LeakageMetrics`` structure,
            ready for ``json.dump`` or DataFrame conversion.
        """
        metrics = self.compute_leakage()
        return {
            "summary": {
                "total_sessions": len(self.sessions),
                "total_calls": sum(s.call_count for s in self.sessions),
                "leakage_bits": metrics.leakage_bits,
                "mutual_information": metrics.mutual_information,
                "reidentification_probability": metrics.reidentification_prob,
                "total_leakage_score": metrics.total_leakage_score,
            },
            "direct": {
                "content_entropy_bits": metrics.direct.content_entropy,
                "avg_input_size_bytes": metrics.direct.avg_input_size_bytes,
                "timing_entropy_bits": metrics.direct.timing_entropy,
                "unique_parameters": metrics.direct.unique_parameters,
                "calls_analyzed": metrics.direct.calls_analyzed,
            },
            "correlation": {
                "cooccurrence_entropy_bits": metrics.correlation.tool_cooccurrence_entropy,
                "parameter_overlap_rate": metrics.correlation.parameter_overlap_rate,
                "sequence_predictability": metrics.correlation.sequence_predictability,
                "unique_bigrams": metrics.correlation.unique_bigrams,
            },
            "frequency": {
                "query_distribution_entropy_bits": metrics.frequency.query_distribution_entropy,
                "repeat_query_rate": metrics.frequency.repeat_query_rate,
                "top_query_concentration": metrics.frequency.top_query_concentration,
                "unique_query_types": metrics.frequency.unique_query_types,
            },
        }
