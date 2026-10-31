"""PIR-based encrypted drug-interaction lookup.

End-to-end pipeline for privately querying a drug-interaction database:
1. Agent maps (drug_a, drug_b) to a database index.
2. Agent creates an encrypted PIR query for that index.
3. Server processes the query homomorphically.
4. Agent decrypts the response to get the interaction record.

The server never learns which drug pair was queried.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.feasibility.pir_lookup import ComputationalPIR, PIRParams


# ===================================================================
# Drug record data class
# ===================================================================

@dataclass
class DrugRecord:
    """A single drug-interaction record.

    Attributes
    ----------
    drug_a : str
        First drug name (normalized to lowercase).
    drug_b : str
        Second drug name (normalized to lowercase).
    severity : int
        Interaction severity (0=none, 1=mild, 2=moderate, 3=severe).
    description : str
        Human-readable description of the interaction.
    recommendation : str
        Clinical recommendation.
    """

    drug_a: str
    drug_b: str
    severity: int
    description: str
    recommendation: str

    def to_bytes(self) -> bytes:
        """Serialize the record to JSON bytes for PIR storage."""
        obj = {
            "a": self.drug_a,
            "b": self.drug_b,
            "s": self.severity,
            "d": self.description,
            "r": self.recommendation,
        }
        return json.dumps(obj, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> DrugRecord:
        """Deserialize a record from JSON bytes.

        Strips trailing null padding automatically.
        """
        cleaned = data.rstrip(b"\x00")
        if not cleaned:
            return cls(drug_a="", drug_b="", severity=0,
                       description="", recommendation="")
        obj = json.loads(cleaned.decode("utf-8"))
        return cls(
            drug_a=obj["a"],
            drug_b=obj["b"],
            severity=obj["s"],
            description=obj["d"],
            recommendation=obj["r"],
        )


def _normalize_pair(drug_a: str, drug_b: str) -> tuple[str, str]:
    """Normalize a drug pair to (lowercase, alphabetical order)."""
    a, b = drug_a.strip().lower(), drug_b.strip().lower()
    return (a, b) if a <= b else (b, a)


# ===================================================================
# Encrypted lookup tool
# ===================================================================

@dataclass
class LookupEvent:
    """A logged lookup event (metadata visible to the server)."""
    timestamp: float
    query_size_bytes: int
    response_size_bytes: int
    processing_time_ms: float
    drug_pair_visible: bool = False  # always False for encrypted lookups


class EncryptedDrugLookup:
    """Private drug-interaction lookup using computational PIR.

    The agent can look up drug interactions without the server learning
    which drugs are being checked.  This is critical for patient privacy
    in a healthcare AI agent scenario.

    Example usage::

        lookup = EncryptedDrugLookup()
        lookup.load_drug_database(records)
        lookup.setup_pir()

        # Agent side: create encrypted query
        query = lookup.create_query("aspirin", "warfarin")

        # Server side: process query on ciphertext
        encrypted_result = lookup.server_process(query)

        # Agent side: decrypt to get the interaction record
        record = lookup.decrypt_result(encrypted_result)
        print(record.severity)  # 3 (severe interaction)
    """

    def __init__(self, pir_params: PIRParams | None = None) -> None:
        self._records: list[DrugRecord] = []
        self._drug_pair_to_index: dict[tuple[str, str], int] = {}
        self._pir: ComputationalPIR | None = None
        self._pir_params = pir_params
        self._record_size_bytes: int = 512
        # Observability
        self.lookup_log: list[LookupEvent] = []

    # ------------------------------------------------------------------
    # Database loading
    # ------------------------------------------------------------------

    def load_drug_database(self, records: list[DrugRecord]) -> None:
        """Load drug interaction records and build the lookup index.

        Drug pairs are normalized to lowercase alphabetical order so
        that ``("Aspirin", "Warfarin")`` and ``("warfarin", "aspirin")``
        map to the same index.
        """
        self._records = list(records)
        self._drug_pair_to_index.clear()
        for i, rec in enumerate(self._records):
            pair = _normalize_pair(rec.drug_a, rec.drug_b)
            self._drug_pair_to_index[pair] = i

        # Determine record size: max serialized length rounded up to 64
        max_len = max(len(r.to_bytes()) for r in self._records) if self._records else 64
        self._record_size_bytes = ((max_len // 64) + 1) * 64

    def setup_pir(self) -> None:
        """Initialize the PIR scheme with the loaded database."""
        if not self._records:
            raise RuntimeError("Load a database first")

        params = self._pir_params or PIRParams(
            db_size=len(self._records),
            record_size_bytes=self._record_size_bytes,
        )
        self._pir = ComputationalPIR(params)
        self._pir.setup()

        serialized = [r.to_bytes() for r in self._records]
        self._pir.load_database(serialized)

    # ------------------------------------------------------------------
    # Query pipeline
    # ------------------------------------------------------------------

    def create_query(self, drug_a: str, drug_b: str) -> bytes:
        """Agent creates an encrypted PIR query for a drug pair.

        Parameters
        ----------
        drug_a : str
            First drug name.
        drug_b : str
            Second drug name.

        Returns
        -------
        bytes
            Encrypted PIR query (serialized BFV ciphertext).

        Raises
        ------
        KeyError
            If the drug pair is not in the database.
        """
        if self._pir is None:
            raise RuntimeError("Call setup_pir() first")

        pair = _normalize_pair(drug_a, drug_b)
        if pair not in self._drug_pair_to_index:
            raise KeyError(f"Drug pair {pair} not in database")
        idx = self._drug_pair_to_index[pair]
        return self._pir.generate_query(idx)

    def server_process(self, encrypted_query: bytes) -> list[bytes]:
        """Server processes the encrypted query.

        This runs on the server side — no access to secret key.

        Parameters
        ----------
        encrypted_query : bytes
            Encrypted PIR query from the agent.

        Returns
        -------
        list[bytes]
            Encrypted result parts (ciphertexts).
        """
        if self._pir is None:
            raise RuntimeError("Call setup_pir() first")

        t0 = time.perf_counter()
        result = self._pir.process_query(encrypted_query)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.lookup_log.append(LookupEvent(
            timestamp=time.time(),
            query_size_bytes=len(encrypted_query),
            response_size_bytes=sum(len(p) for p in result),
            processing_time_ms=elapsed_ms,
            drug_pair_visible=False,
        ))
        return result

    def decrypt_result(self, encrypted_result: list[bytes]) -> DrugRecord:
        """Agent decrypts the PIR response to get the record.

        Parameters
        ----------
        encrypted_result : list[bytes]
            Encrypted result parts from the server.

        Returns
        -------
        DrugRecord
            The decrypted drug interaction record.
        """
        if self._pir is None:
            raise RuntimeError("Call setup_pir() first")
        raw = self._pir.decrypt_result(encrypted_result)
        return DrugRecord.from_bytes(raw)

    def query(self, drug_a: str, drug_b: str) -> DrugRecord:
        """End-to-end encrypted query (convenience wrapper).

        Runs create_query -> server_process -> decrypt_result in one call.
        """
        q = self.create_query(drug_a, drug_b)
        r = self.server_process(q)
        return self.decrypt_result(r)

    # ------------------------------------------------------------------
    # Plaintext baseline
    # ------------------------------------------------------------------

    def query_plaintext(self, drug_a: str, drug_b: str) -> DrugRecord | None:
        """Direct plaintext lookup (baseline, not private).

        Parameters
        ----------
        drug_a : str
            First drug name.
        drug_b : str
            Second drug name.

        Returns
        -------
        DrugRecord or None
            DrugRecord if interaction exists, else None.
        """
        pair = _normalize_pair(drug_a, drug_b)
        idx = self._drug_pair_to_index.get(pair)
        if idx is None:
            return None
        return self._records[idx]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def database_size(self) -> int:
        """Number of records in the database."""
        return len(self._records)

    @property
    def drug_pairs(self) -> list[tuple[str, str]]:
        """List of all drug pairs in the database."""
        return list(self._drug_pair_to_index.keys())


# ===================================================================
# Data loading helper
# ===================================================================

_SEVERITY_STR_TO_INT = {
    "none": 0, "mild": 1, "moderate": 2, "severe": 3, "contraindicated": 4,
}


def _parse_severity(raw: int | str) -> int:
    """Accept severity as int (legacy) or string (new schema)."""
    if isinstance(raw, int):
        return raw
    return _SEVERITY_STR_TO_INT.get(str(raw).lower(), 0)


def load_drug_records_from_json(path: str | Path) -> list[DrugRecord]:
    """Load drug interaction records from a JSON file.

    Supports both the legacy schema (``severity`` as int,
    ``description`` / ``recommendation`` fields) and the new schema
    (``severity`` as string, ``mechanism`` / ``clinical_recommendation``
    fields).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records: list[DrugRecord] = []
    for entry in data:
        severity = _parse_severity(entry.get("severity", 0))
        # Support both old and new field names
        description = entry.get("description",
                                entry.get("mechanism", ""))
        recommendation = entry.get("recommendation",
                                   entry.get("clinical_recommendation", ""))
        records.append(DrugRecord(
            drug_a=entry["drug_a"],
            drug_b=entry["drug_b"],
            severity=severity,
            description=description,
            recommendation=recommendation,
        ))
    return records
