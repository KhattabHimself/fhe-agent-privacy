"""Simulated AI agent that makes tool calls on behalf of a user.

The agent receives a user request (e.g., "check if my medications interact"),
plans a sequence of tool calls, and executes them. In the encrypted pipeline,
the agent holds the encryption keys and encrypts queries before sending them
to the tool server.

Two modes of operation:
1. Plaintext mode — calls are sent in the clear (baseline, leaks information).
2. Encrypted mode — calls are encrypted using PIR/FHE before sending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import tenseal as ts

from src.threat_model.definitions import ToolCall, ToolCallSequence


class AgentMode(Enum):
    """Operating mode for the agent."""

    PLAINTEXT = auto()   # No encryption — baseline
    ENCRYPTED = auto()   # PIR/FHE encrypted tool calls


@dataclass
class UserRequest:
    """A user's request to the agent.

    Attributes:
        request_id: Unique identifier.
        description: Natural-language description of the request.
        patient_data: Structured patient data (medications, vitals, etc.).
        intent: Ground-truth intent label (for evaluation).
    """

    request_id: str
    description: str
    patient_data: dict[str, Any]
    intent: str = ""


@dataclass
class AgentConfig:
    """Configuration for the simulated agent.

    Attributes:
        mode: PLAINTEXT or ENCRYPTED.
        fhe_params: FHE parameters (poly_modulus_degree, etc.).
        tools_available: List of tool names the agent can call.
    """

    mode: AgentMode = AgentMode.PLAINTEXT
    fhe_params: dict[str, Any] = field(default_factory=dict)
    tools_available: list[str] = field(default_factory=lambda: [
        "drug_interaction_db",
        "risk_scoring_service",
    ])


class Agent:
    """Simulated AI agent with optional encrypted tool calls.

    Example usage::

        agent = Agent(config=AgentConfig(mode=AgentMode.ENCRYPTED))
        agent.setup_encryption()
        result = agent.handle_request(user_request)
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._context: ts.Context | None = None
        self._call_history: list[ToolCallSequence] = []

    def setup_encryption(self) -> None:
        """Initialize FHE context and keys for encrypted mode.

        Creates a TenSEAL CKKS context. The agent keeps the secret key;
        the public context is sent to tool servers.
        """
        # TODO: initialize CKKS context if mode is ENCRYPTED
        # 1. Create context with agent's FHE params
        # 2. Generate galois keys
        # 3. Store full context (with secret key)
        raise NotImplementedError

    def get_public_context(self) -> bytes:
        """Serialize the public context (no secret key) for tool servers.

        Returns:
            Serialized public context bytes.
        """
        # TODO: serialize context without secret key
        raise NotImplementedError

    def handle_request(self, request: UserRequest) -> dict[str, Any]:
        """Process a user request by planning and executing tool calls.

        Args:
            request: The user's request with patient data.

        Returns:
            Dict with results from all tool calls.
        """
        # TODO: implement request handling
        # 1. Plan which tools to call based on request
        # 2. For each tool call:
        #    a. If ENCRYPTED: encrypt the query parameters
        #    b. Send to tool server
        #    c. If ENCRYPTED: decrypt the response
        # 3. Aggregate results
        # 4. Record call sequence in history
        raise NotImplementedError

    def _plan_tool_calls(self, request: UserRequest) -> list[dict[str, Any]]:
        """Decide which tools to call and with what parameters.

        Args:
            request: The user's request.

        Returns:
            Ordered list of planned tool calls with parameters.
        """
        # TODO: simple rule-based planner
        # - If request mentions medications -> call drug_interaction_db
        # - If request mentions risk/score -> call risk_scoring_service
        # - May call multiple tools in sequence
        raise NotImplementedError

    @property
    def call_history(self) -> list[ToolCallSequence]:
        """All tool-call sequences executed by this agent."""
        return self._call_history
