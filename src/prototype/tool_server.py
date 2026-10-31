"""Simulated tool server (honest-but-curious).

The tool server hosts databases and computation services that the AI agent
queries. In the threat model, the server is honest-but-curious: it
correctly executes queries but logs everything it observes.

This module provides:
- DrugInteractionServer: answers drug-interaction queries
- RiskScoringServer: computes patient risk scores
- ServerObservationLog: records what the server sees (for leakage analysis)

In encrypted mode, the server operates on ciphertexts and learns nothing
about the query contents or results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ServerObservation:
    """A single observation recorded by the honest-but-curious server.

    Attributes:
        timestamp: When the query was received.
        tool_name: Which tool was invoked.
        is_encrypted: Whether the query was encrypted.
        observed_data: What the server could see (plaintext params or
            ciphertext metadata like size, timing).
    """

    timestamp: float
    tool_name: str
    is_encrypted: bool
    observed_data: dict[str, Any]


class ServerObservationLog:
    """Log of everything the server observes across all queries.

    Used by the leakage analysis module to quantify information leakage.
    """

    def __init__(self) -> None:
        self.observations: list[ServerObservation] = []

    def record(self, observation: ServerObservation) -> None:
        """Add an observation to the log."""
        self.observations.append(observation)

    def get_observations_for_tool(self, tool_name: str) -> list[ServerObservation]:
        """Filter observations by tool name."""
        return [o for o in self.observations if o.tool_name == tool_name]

    def plaintext_observations(self) -> list[ServerObservation]:
        """Return only unencrypted observations."""
        return [o for o in self.observations if not o.is_encrypted]

    def summary(self) -> dict[str, Any]:
        """Summary statistics of the observation log."""
        return {
            "total_queries": len(self.observations),
            "encrypted_queries": sum(1 for o in self.observations if o.is_encrypted),
            "plaintext_queries": sum(1 for o in self.observations if not o.is_encrypted),
            "tools_queried": list({o.tool_name for o in self.observations}),
        }


class DrugInteractionServer:
    """Simulated drug-interaction database server.

    Holds a database of drug pairs and their interaction severity.
    Supports both plaintext and encrypted (PIR-based) lookups.
    """

    def __init__(self) -> None:
        self._database: dict[tuple[str, str], dict[str, Any]] = {}
        self._log = ServerObservationLog()

    def load_database(self, records: list[dict[str, Any]]) -> None:
        """Load drug interaction records.

        Args:
            records: List of dicts with keys: drug_a, drug_b, severity,
                description.
        """
        # TODO: populate self._database from records
        # Key: (drug_a, drug_b) sorted tuple
        # Value: {severity, description, ...}
        raise NotImplementedError

    def query_plaintext(
        self, drug_a: str, drug_b: str,
    ) -> dict[str, Any] | None:
        """Look up interaction between two drugs (plaintext).

        The server sees exactly which drugs are being checked.

        Args:
            drug_a: First drug name.
            drug_b: Second drug name.

        Returns:
            Interaction record or None if no interaction.
        """
        # TODO: look up in database, log the observation
        raise NotImplementedError

    def query_encrypted(self, encrypted_query: Any) -> Any:
        """Process an encrypted (PIR) lookup.

        The server performs the PIR computation without learning
        which drug pair was queried.

        Args:
            encrypted_query: Encrypted selection vector from the agent.

        Returns:
            Encrypted result (the agent decrypts it).
        """
        # TODO: perform PIR computation on ciphertext
        # Log only metadata (ciphertext size, timing) — not content
        raise NotImplementedError

    @property
    def observation_log(self) -> ServerObservationLog:
        return self._log


class RiskScoringServer:
    """Simulated risk-scoring computation server.

    Evaluates a patient risk model. In plaintext mode, sees the patient
    features. In encrypted mode, computes on CKKS ciphertexts.
    """

    def __init__(self) -> None:
        self._model_weights: NDArray[np.float64] | None = None
        self._model_bias: float = 0.0
        self._log = ServerObservationLog()

    def load_model(
        self, weights: NDArray[np.float64], bias: float,
    ) -> None:
        """Load the risk-scoring model parameters.

        Args:
            weights: Linear model weight vector.
            bias: Model bias term.
        """
        self._model_weights = weights
        self._model_bias = bias

    def score_plaintext(
        self, features: NDArray[np.float64],
    ) -> float:
        """Compute risk score on plaintext features.

        The server sees the raw patient features.

        Args:
            features: Patient feature vector.

        Returns:
            Risk score (higher = more risk).
        """
        # TODO: compute w . x + b, log the observation
        raise NotImplementedError

    def score_encrypted(self, encrypted_features: Any) -> Any:
        """Compute risk score on encrypted features.

        Uses FHE to evaluate the linear model on ciphertext.
        The server learns nothing about the patient features or score.

        Args:
            encrypted_features: CKKS-encrypted feature vector.

        Returns:
            CKKS-encrypted risk score.
        """
        # TODO: homomorphic dot product + bias addition
        # Log only metadata (ciphertext size, timing)
        raise NotImplementedError

    @property
    def observation_log(self) -> ServerObservationLog:
        return self._log
