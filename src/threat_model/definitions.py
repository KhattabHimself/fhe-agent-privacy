"""Formal definitions for the tool-call privacy threat model.

This module establishes the formal framework for analyzing privacy leakage
in AI agent tool-call pipelines.  The definitions below correspond to
Section 3 of the accompanying paper and are intended to be self-contained:
every symbol used in the leakage and reconstruction analyses is grounded in
a concrete Python type defined here.

Core abstractions
-----------------
* **ToolCallType** enumerates the four functional classes of tool
  invocations an agent may issue.
* **AdversaryModel** captures the capability assumptions placed on the
  tool-server adversary.
* **PrivacyProperty** names the orthogonal dimensions of information
  that cryptographic countermeasures may protect.
* **ToolCall** is the atomic unit of observation: a single request from
  an AI agent to an external tool server, carrying typed inputs, a
  response, and timing metadata.
* **AgentSession** (aliased as *ToolCallSequence* for backward
  compatibility) is an ordered, timestamped sequence of ``ToolCall``
  objects that together constitute one user interaction.
* **ThreatModel** bundles the adversary model with the specific
  properties the adversary can observe and those we aim to protect.
* **PrivacyGoal** pairs a target privacy property with the cryptographic
  mechanism chosen to protect it and the maximum tolerable leakage.

All dataclasses use ``frozen=True`` where mutation is never required,
enforcing value semantics and making instances safe to use as dict keys.

These definitions follow the formalism in Section 3 of the paper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# 3.1  Tool-call taxonomy
# ---------------------------------------------------------------------------

class ToolCallType(Enum):
    """Functional classification of tool invocations.

    An AI agent's interactions with external services fall into four
    categories, each with distinct privacy implications:

    * ``DATA_RETRIEVAL`` — read-only lookups (e.g., querying a drug-
      interaction database).  Leaks the *query content* to the server.
    * ``COMPUTATION`` — stateless computation on agent-supplied inputs
      (e.g., risk-score evaluation).  Leaks the *input features*.
    * ``PATTERN_ACCESS`` — access to structured data where the *access
      pattern itself* (which indices are touched, in what order) is
      sensitive, even if individual values are encrypted.
    * ``ARBITRARY_EXECUTION`` — general-purpose code execution or
      multi-step workflows whose leakage surface is difficult to bound
      statically.
    """

    DATA_RETRIEVAL = auto()
    COMPUTATION = auto()
    PATTERN_ACCESS = auto()
    ARBITRARY_EXECUTION = auto()


# ---------------------------------------------------------------------------
# 3.2  Adversary model
# ---------------------------------------------------------------------------

class AdversaryModel(Enum):
    """Capability assumptions placed on the tool-server adversary.

    * ``HONEST_BUT_CURIOUS`` (semi-honest) — the server faithfully
      executes the protocol but records every observable quantity
      (query content, timing, response size, access pattern) and may
      later attempt offline inference.  This is the primary model
      studied in the paper.
    * ``ACTIVE`` — the server may deviate from the protocol: it can
      return incorrect results, replay old responses, or inject chosen
      inputs in an attempt to extract additional information.  Defending
      against active adversaries requires verifiable computation or
      authenticated responses, which is out of scope for this work but
      noted as future work.
    """

    HONEST_BUT_CURIOUS = auto()
    ACTIVE = auto()


# Keep backward compatibility — older code may reference these names.
MALICIOUS = AdversaryModel.ACTIVE
COVERT = AdversaryModel.ACTIVE  # treated as ACTIVE for simplicity


class PrivacyProperty(Enum):
    """Orthogonal dimensions of information that may leak.

    Each property can be targeted independently by a different
    cryptographic mechanism:

    * ``QUERY_CONTENT`` — the literal parameter values sent in a tool
      call.  Protected by FHE or PIR.
    * ``ACCESS_PATTERN`` — which logical records (database rows, array
      indices) are touched.  Protected by ORAM.
    * ``QUERY_FREQUENCY`` — how many queries of each type are issued
      and at what rate.  Protected by padding / dummy queries.
    * ``RESULT_CONTENT`` — the server's response.  Protected by
      returning encrypted results (FHE) or the full database (PIR).
    * ``USER_INTENT`` — the high-level goal that can be inferred from
      the *combination* of all other leaked properties.  This is the
      ultimate target; protecting all lower-level properties is a
      sufficient (but not always necessary) condition.
    """

    QUERY_CONTENT = auto()
    ACCESS_PATTERN = auto()
    QUERY_FREQUENCY = auto()
    RESULT_CONTENT = auto()
    USER_INTENT = auto()


# ---------------------------------------------------------------------------
# 3.3  Tool call — atomic unit of observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation from an AI agent to a tool server.

    This is the *atomic observation* in the threat model.  An honest-but-
    curious server records one ``ToolCall`` object for every request it
    processes.

    Attributes
    ----------
    tool_name : str
        Identifier of the tool being called (e.g.,
        ``"drug_interaction_db"``).
    method : str
        Specific endpoint within the tool (e.g., ``"lookup"``,
        ``"score"``).
    parameters : dict[str, Any]
        The query payload — this is the *sensitive content* that
        cryptographic protection aims to hide.
    timestamp : float
        Wall-clock time of the call relative to session start, in
        seconds.  Used for timing-based correlation analysis.
    session_id : str
        Groups calls belonging to the same user interaction.
    call_type : ToolCallType
        Functional classification of this invocation.
    input_size_bytes : int
        Size of the serialized input payload.  Observable even when
        the payload is encrypted.
    output : dict[str, Any] | None
        The server's response (``None`` if not yet executed).
    output_size_bytes : int
        Size of the serialized response.
    """

    tool_name: str
    method: str
    parameters: dict[str, Any]
    timestamp: float
    session_id: str
    call_type: ToolCallType = ToolCallType.DATA_RETRIEVAL
    input_size_bytes: int = 0
    output: dict[str, Any] | None = None
    output_size_bytes: int = 0

    def __post_init__(self) -> None:
        # Auto-compute input_size_bytes from parameters if not provided.
        if self.input_size_bytes == 0 and self.parameters:
            object.__setattr__(
                self,
                "input_size_bytes",
                len(json.dumps(self.parameters, default=str).encode()),
            )


# ---------------------------------------------------------------------------
# 3.4  Agent session (tool-call sequence)
# ---------------------------------------------------------------------------

@dataclass
class ToolCallSequence:
    """An ordered sequence of tool calls within a single user session.

    The honest-but-curious server observes the full sequence in real time
    and may apply temporal, correlational, and frequency-based analyses
    to reconstruct the user's high-level intent.

    ``ToolCallSequence`` is aliased as ``AgentSession`` for readability
    in higher-level modules; the two names are interchangeable.

    Attributes
    ----------
    session_id : str
        Unique session identifier.
    calls : list[ToolCall]
        Chronologically ordered list of tool calls.
    intent : str
        Ground-truth intent label (used only for evaluation, never
        visible to the adversary).
    encrypted : bool
        Whether the calls in this session were protected by
        cryptographic mechanisms.  Affects which leakage channels are
        active.
    """

    session_id: str
    calls: list[ToolCall] = field(default_factory=list)
    intent: str = ""
    encrypted: bool = False

    # -- mutators -----------------------------------------------------------

    def add_call(self, call: ToolCall) -> None:
        """Append a tool call to the sequence."""
        self.calls.append(call)

    # -- derived properties -------------------------------------------------

    @property
    def call_count(self) -> int:
        """Total number of tool calls in the session."""
        return len(self.calls)

    @property
    def unique_tools(self) -> set[str]:
        """Distinct tool names invoked in this session."""
        return {c.tool_name for c in self.calls}

    @property
    def unique_methods(self) -> set[str]:
        """Distinct (tool_name, method) pairs as ``"tool.method"`` strings."""
        return {f"{c.tool_name}.{c.method}" for c in self.calls}

    @property
    def duration(self) -> float:
        """Wall-clock span of the session in seconds."""
        if len(self.calls) < 2:
            return 0.0
        return self.calls[-1].timestamp - self.calls[0].timestamp

    @property
    def total_input_bytes(self) -> int:
        """Sum of input payload sizes across all calls."""
        return sum(c.input_size_bytes for c in self.calls)

    @property
    def total_output_bytes(self) -> int:
        """Sum of output payload sizes across all calls."""
        return sum(c.output_size_bytes for c in self.calls)

    def tool_call_counts(self) -> dict[str, int]:
        """Return a mapping from ``"tool.method"`` to invocation count."""
        counts: dict[str, int] = {}
        for c in self.calls:
            key = f"{c.tool_name}.{c.method}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def inter_call_intervals(self) -> list[float]:
        """Time deltas between consecutive calls (seconds)."""
        if len(self.calls) < 2:
            return []
        return [
            self.calls[i + 1].timestamp - self.calls[i].timestamp
            for i in range(len(self.calls) - 1)
        ]

    def to_tool_name_sequence(self) -> list[str]:
        """Return the ordered list of tool names (for n-gram analysis)."""
        return [c.tool_name for c in self.calls]

    def to_method_sequence(self) -> list[str]:
        """Return the ordered list of ``"tool.method"`` strings."""
        return [f"{c.tool_name}.{c.method}" for c in self.calls]


# Alias for readability in higher-level code.
AgentSession = ToolCallSequence


# ---------------------------------------------------------------------------
# 3.5  Threat model
# ---------------------------------------------------------------------------

@dataclass
class ThreatModel:
    """Complete specification of the adversary and the privacy target.

    A ``ThreatModel`` instance pins down:

    1. **Who** the adversary is (``adversary``).
    2. **What** the adversary can observe in the unprotected baseline
       (``observed_properties``).
    3. **What** we aim to hide after applying cryptographic protection
       (``target_properties``).
    4. **What** auxiliary (side-channel) knowledge the adversary
       possesses (``auxiliary_knowledge``).

    Attributes
    ----------
    adversary : AdversaryModel
        Adversary capability class.
    observed_properties : list[PrivacyProperty]
        Properties the server can observe in plaintext (no protection).
    target_properties : list[PrivacyProperty]
        Properties we aim to protect via cryptographic mechanisms.
    auxiliary_knowledge : dict[str, Any]
        Side information the adversary may exploit (e.g., population-
        level drug-prescription statistics, public medical ontologies).
    """

    adversary: AdversaryModel = AdversaryModel.HONEST_BUT_CURIOUS
    observed_properties: list[PrivacyProperty] = field(default_factory=lambda: [
        PrivacyProperty.QUERY_CONTENT,
        PrivacyProperty.ACCESS_PATTERN,
        PrivacyProperty.QUERY_FREQUENCY,
    ])
    target_properties: list[PrivacyProperty] = field(default_factory=lambda: [
        PrivacyProperty.QUERY_CONTENT,
        PrivacyProperty.USER_INTENT,
    ])
    auxiliary_knowledge: dict[str, Any] = field(default_factory=dict)

    def is_property_observed(self, prop: PrivacyProperty) -> bool:
        """Return True if the adversary can observe *prop* in plaintext."""
        return prop in self.observed_properties

    def is_property_targeted(self, prop: PrivacyProperty) -> bool:
        """Return True if *prop* is a protection target."""
        return prop in self.target_properties


# ---------------------------------------------------------------------------
# 3.6  Privacy goal
# ---------------------------------------------------------------------------

@dataclass
class PrivacyGoal:
    """A concrete, measurable privacy objective.

    Pairs a privacy property with the cryptographic mechanism chosen to
    protect it and states the maximum tolerable information leakage
    (in bits).  The ``is_satisfied`` predicate allows automated pass/fail
    evaluation during experiments.

    Attributes
    ----------
    property_to_protect : PrivacyProperty
        Which privacy dimension this goal addresses.
    mechanism : str
        Cryptographic mechanism (``"FHE"``, ``"PIR"``, ``"ORAM"``).
    security_parameter : int
        Target computational security level in bits (default 128).
    acceptable_leakage : float
        Upper bound on tolerable leakage in bits (default 0 = perfect
        privacy).
    """

    property_to_protect: PrivacyProperty
    mechanism: str
    security_parameter: int = 128
    acceptable_leakage: float = 0.0

    def is_satisfied(self, measured_leakage: float) -> bool:
        """Return True when measured leakage is within tolerance."""
        return measured_leakage <= self.acceptable_leakage
