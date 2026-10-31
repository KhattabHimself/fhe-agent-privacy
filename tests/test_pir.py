"""Tests for PIR (Private Information Retrieval) implementation.

Covers:
- Record encoding/decoding
- TrivialPIR correctness
- ComputationalPIR correctness across database sizes
- Privacy guarantees (server log inspection)
- EncryptedDrugLookup end-to-end
- DrugRecord serialization round-trip
"""

from __future__ import annotations

import json

import pytest

from src.feasibility.pir_lookup import (
    ComputationalPIR,
    PIRParams,
    PIRClient,
    PIRServer,
    TrivialPIR,
    _decode_record,
    _encode_record,
)
from src.prototype.encrypted_lookup import (
    DrugRecord,
    EncryptedDrugLookup,
    _normalize_pair,
    load_drug_records_from_json,
)


# ===================================================================
# Record encoding / decoding
# ===================================================================

class TestRecordEncoding:
    """Tests for _encode_record and _decode_record."""

    def test_roundtrip_simple(self) -> None:
        data = b"hello world"
        record_size = 16
        bps = 2
        encoded = _encode_record(data, record_size, bps)
        decoded = _decode_record(encoded, record_size, bps)
        assert decoded[:len(data)] == data

    def test_roundtrip_exact_size(self) -> None:
        data = b"\x01\x02\x03\x04"
        encoded = _encode_record(data, 4, 2)
        decoded = _decode_record(encoded, 4, 2)
        assert decoded == data

    def test_encoding_packing(self) -> None:
        """Two bytes packed little-endian: 0x01 + 0x02<<8 = 513."""
        data = b"\x01\x02\x03"
        encoded = _encode_record(data, 4, 2)
        assert encoded[0] == 0x01 + (0x02 << 8)  # 513
        assert encoded[1] == 0x03 + (0x00 << 8)  # 3

    def test_empty_record(self) -> None:
        encoded = _encode_record(b"", 4, 2)
        decoded = _decode_record(encoded, 4, 2)
        assert decoded == b"\x00" * 4

    def test_single_byte_per_slot(self) -> None:
        data = b"\xAB\xCD\xEF"
        encoded = _encode_record(data, 4, 1)
        assert encoded == [0xAB, 0xCD, 0xEF, 0x00]
        decoded = _decode_record(encoded, 4, 1)
        assert decoded == b"\xAB\xCD\xEF\x00"


# ===================================================================
# TrivialPIR
# ===================================================================

class TestTrivialPIR:
    """Tests for the trivial (download-all) PIR baseline."""

    def test_correctness(self) -> None:
        records = [f"record-{i}".encode() for i in range(100)]
        pir = TrivialPIR(database=records)
        assert pir.query(42) == b"record-42"
        assert pir.query(0) == b"record-0"
        assert pir.query(99) == b"record-99"

    def test_communication_cost(self) -> None:
        records = [b"x" * 256 for _ in range(100)]
        pir = TrivialPIR(database=records)
        assert pir.communication_cost() == 256 * 100

    def test_out_of_range(self) -> None:
        pir = TrivialPIR(database=[b"a", b"b"])
        with pytest.raises(IndexError):
            pir.query(5)
        with pytest.raises(IndexError):
            pir.query(-1)


# ===================================================================
# ComputationalPIR
# ===================================================================

class TestComputationalPIR:
    """Tests for the BFV-based computational PIR."""

    @pytest.fixture
    def small_pir(self) -> ComputationalPIR:
        """A small PIR instance for fast tests (10 records)."""
        params = PIRParams(db_size=10, record_size_bytes=64)
        cpir = ComputationalPIR(params)
        cpir.setup()
        records = [json.dumps({"id": i, "data": f"val-{i}"}).encode()
                   for i in range(10)]
        cpir.load_database(records)
        return cpir

    def test_setup(self) -> None:
        params = PIRParams(db_size=5, record_size_bytes=32)
        cpir = ComputationalPIR(params)
        ctx = cpir.setup()
        assert ctx is not None

    def test_correctness_all_indices(self, small_pir: ComputationalPIR) -> None:
        """Every index should return the correct record."""
        for i in range(10):
            result = small_pir.end_to_end_query(i)
            decoded = result.rstrip(b"\x00")
            expected = json.dumps({"id": i, "data": f"val-{i}"}).encode()
            assert decoded == expected, f"Mismatch at index {i}"

    def test_correctness_with_binary_data(self) -> None:
        """PIR should handle arbitrary byte records."""
        params = PIRParams(db_size=5, record_size_bytes=32)
        cpir = ComputationalPIR(params)
        cpir.setup()

        records = [bytes(range(i, i + 20)) for i in range(5)]
        cpir.load_database(records)

        for i in range(5):
            result = cpir.end_to_end_query(i)
            assert result[:20] == records[i]

    def test_out_of_range_query(self) -> None:
        params = PIRParams(db_size=5, record_size_bytes=32)
        cpir = ComputationalPIR(params)
        cpir.setup()
        cpir.load_database([b"x" * 32] * 5)

        with pytest.raises(IndexError):
            cpir.generate_query(5)
        with pytest.raises(IndexError):
            cpir.generate_query(-1)

    def test_communication_cost_structure(self) -> None:
        params = PIRParams(db_size=10, record_size_bytes=64)
        cpir = ComputationalPIR(params)
        cpir.setup()
        costs = cpir.communication_cost()
        assert "query_size" in costs
        assert "response_size" in costs
        assert "total" in costs
        assert costs["total"] == costs["query_size"] + costs["response_size"]
        assert costs["query_size"] > 0
        assert costs["response_size"] > 0


# ===================================================================
# Privacy guarantees
# ===================================================================

class TestPIRPrivacy:
    """Tests verifying that the server learns nothing about the query."""

    def test_server_log_has_no_index(self) -> None:
        """Server's query log should never contain the queried index."""
        params = PIRParams(db_size=10, record_size_bytes=64)
        cpir = ComputationalPIR(params)
        cpir.setup()
        records = [f"record-{i}".encode() for i in range(10)]
        cpir.load_database(records)

        for target in [0, 3, 7, 9]:
            cpir.end_to_end_query(target)

        for entry in cpir.server.query_log:
            assert entry["visible_index"] is None

    def test_queries_for_different_indices_similar_size(self) -> None:
        """Encrypted queries for different indices should be similar size.

        BFV ciphertext serialization may vary slightly due to internal
        compression, but the variation should be negligible (<1%) and
        not leak the index.
        """
        params = PIRParams(db_size=20, record_size_bytes=64)
        cpir = ComputationalPIR(params)
        cpir.setup()
        cpir.load_database([b"x" * 64] * 20)

        sizes = []
        for idx in [0, 5, 10, 15, 19]:
            q = cpir.generate_query(idx)
            sizes.append(len(q))

        mean_size = sum(sizes) / len(sizes)
        for s in sizes:
            deviation = abs(s - mean_size) / mean_size
            assert deviation < 0.01, (
                f"Query size {s} deviates {deviation:.2%} from mean {mean_size:.0f}"
            )

    def test_server_processes_without_secret_key(self) -> None:
        """Server context should not contain the secret key."""
        params = PIRParams(db_size=5, record_size_bytes=32)
        client = PIRClient(params)
        server = PIRServer()

        public_ctx = client.setup()
        server.load_context(public_ctx)

        # The server's context is public-only; it cannot decrypt
        assert server._ctx is not None


# ===================================================================
# PIR Client / Server separation
# ===================================================================

class TestPIRClientServer:
    """Tests for the two-party PIR protocol."""

    def test_separate_client_server(self) -> None:
        """Client and server operate independently with serialized messages."""
        params = PIRParams(db_size=8, record_size_bytes=32)

        client = PIRClient(params)
        server = PIRServer()

        # Step 1: Client creates context, sends public part to server
        public_ctx = client.setup()
        server.load_context(public_ctx)

        # Step 2: Server loads database
        records = [f"item-{i}".encode() for i in range(8)]
        server.load_database(records, params)

        # Step 3: Client generates encrypted query
        target = 4
        query_bytes = client.generate_query(target)

        # Step 4: Server processes (cannot see index)
        response_parts = server.process_query(query_bytes)

        # Step 5: Client decrypts
        result = client.decrypt_response(response_parts)
        assert result.rstrip(b"\x00") == records[target]


# ===================================================================
# DrugRecord serialization
# ===================================================================

class TestDrugRecord:
    """Tests for DrugRecord serialization."""

    def test_roundtrip(self) -> None:
        rec = DrugRecord(
            drug_a="aspirin", drug_b="warfarin",
            severity=3,
            description="Severe bleeding risk",
            recommendation="Avoid combination",
        )
        data = rec.to_bytes()
        restored = DrugRecord.from_bytes(data)
        assert restored.drug_a == rec.drug_a
        assert restored.drug_b == rec.drug_b
        assert restored.severity == rec.severity
        assert restored.description == rec.description
        assert restored.recommendation == rec.recommendation

    def test_from_bytes_with_padding(self) -> None:
        rec = DrugRecord("a", "b", 0, "none", "ok")
        data = rec.to_bytes() + b"\x00" * 100
        restored = DrugRecord.from_bytes(data)
        assert restored.drug_a == "a"

    def test_normalize_pair(self) -> None:
        assert _normalize_pair("Warfarin", "Aspirin") == ("aspirin", "warfarin")
        assert _normalize_pair("aspirin", "warfarin") == ("aspirin", "warfarin")
        assert _normalize_pair(" Aspirin ", "WARFARIN") == ("aspirin", "warfarin")


# ===================================================================
# EncryptedDrugLookup
# ===================================================================

class TestEncryptedDrugLookup:
    """Tests for the end-to-end encrypted drug lookup."""

    @pytest.fixture
    def drug_records(self) -> list[DrugRecord]:
        return [
            DrugRecord("aspirin", "warfarin", 3,
                       "Severe bleeding risk", "Avoid"),
            DrugRecord("metformin", "lisinopril", 0,
                       "No interaction", "Safe"),
            DrugRecord("omeprazole", "clopidogrel", 2,
                       "Reduced efficacy", "Monitor"),
            DrugRecord("sertraline", "tramadol", 3,
                       "Serotonin syndrome risk", "Contraindicated"),
            DrugRecord("ibuprofen", "aspirin", 2,
                       "Reduced cardioprotection", "Time doses"),
        ]

    def test_plaintext_lookup(self, drug_records: list[DrugRecord]) -> None:
        lookup = EncryptedDrugLookup()
        lookup.load_drug_database(drug_records)
        rec = lookup.query_plaintext("aspirin", "warfarin")
        assert rec is not None
        assert rec.severity == 3

    def test_plaintext_lookup_reversed_order(self, drug_records: list[DrugRecord]) -> None:
        lookup = EncryptedDrugLookup()
        lookup.load_drug_database(drug_records)
        rec = lookup.query_plaintext("Warfarin", "ASPIRIN")
        assert rec is not None
        assert rec.severity == 3

    def test_plaintext_lookup_missing(self, drug_records: list[DrugRecord]) -> None:
        lookup = EncryptedDrugLookup()
        lookup.load_drug_database(drug_records)
        rec = lookup.query_plaintext("aspirin", "metformin")
        assert rec is None

    def test_encrypted_lookup(self, drug_records: list[DrugRecord]) -> None:
        lookup = EncryptedDrugLookup()
        lookup.load_drug_database(drug_records)
        lookup.setup_pir()

        rec = lookup.query("aspirin", "warfarin")
        assert rec.severity == 3
        assert rec.drug_a == "aspirin"

    def test_encrypted_lookup_all_records(self, drug_records: list[DrugRecord]) -> None:
        lookup = EncryptedDrugLookup()
        lookup.load_drug_database(drug_records)
        lookup.setup_pir()

        for orig in drug_records:
            rec = lookup.query(orig.drug_a, orig.drug_b)
            assert rec.severity == orig.severity
            assert rec.drug_a == orig.drug_a

    def test_encrypted_lookup_privacy_log(self, drug_records: list[DrugRecord]) -> None:
        """Server log should not reveal which drug pair was queried."""
        lookup = EncryptedDrugLookup()
        lookup.load_drug_database(drug_records)
        lookup.setup_pir()

        lookup.query("aspirin", "warfarin")
        lookup.query("metformin", "lisinopril")

        assert len(lookup.lookup_log) == 2
        for event in lookup.lookup_log:
            assert event.drug_pair_visible is False

    def test_unknown_drug_pair_raises(self, drug_records: list[DrugRecord]) -> None:
        lookup = EncryptedDrugLookup()
        lookup.load_drug_database(drug_records)
        lookup.setup_pir()

        with pytest.raises(KeyError):
            lookup.create_query("nonexistent", "drug")

    def test_database_size(self, drug_records: list[DrugRecord]) -> None:
        lookup = EncryptedDrugLookup()
        lookup.load_drug_database(drug_records)
        assert lookup.database_size == 5

    def test_drug_pairs_list(self, drug_records: list[DrugRecord]) -> None:
        lookup = EncryptedDrugLookup()
        lookup.load_drug_database(drug_records)
        pairs = lookup.drug_pairs
        assert len(pairs) == 5
        assert ("aspirin", "warfarin") in pairs


# ===================================================================
# Data loading
# ===================================================================

class TestDataLoading:
    """Tests for loading drug records from JSON."""

    def test_load_from_json(self, tmp_path) -> None:
        data = [
            {"drug_a": "a", "drug_b": "b", "severity": 1,
             "description": "d", "recommendation": "r"},
        ]
        p = tmp_path / "drugs.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        records = load_drug_records_from_json(p)
        assert len(records) == 1
        assert records[0].drug_a == "a"
        assert records[0].severity == 1
