"""Private Information Retrieval (PIR) for database lookups.

PIR allows a client to retrieve a record from a server-held database
without revealing *which* record was requested.  This module implements
two PIR schemes:

1. **TrivialPIR** — the server sends the entire database.  Perfectly
   private (the server learns nothing beyond the fact that a query was
   made) but bandwidth is O(n).  Used as a correctness and privacy
   baseline.

2. **ComputationalPIR** — a BFV-based scheme using TenSEAL.  The client
   encrypts a one-hot selection vector; the server multiplies it
   element-wise with each "column" of the database and returns the
   encrypted inner products.  Only the client can decrypt.

   *Simplifications for this research prototype*:

   - The database is stored as a dense matrix of packed integers
     (2 bytes per BFV plaintext slot, using modulus 1 032 193 > 2^16).
   - There is no recursive/matrix-based query expansion (O(sqrt(n))
     communication); the selection vector has length n.  This is
     acceptable for n <= 10 000 in a research setting.
   - Galois keys are generated but SIMD rotation is not used for the
     inner-product accumulation; instead we sum on the plaintext side
     after decryption (equivalent to summing encrypted slots, but
     simpler to implement with TenSEAL's API).

   These choices trade communication efficiency for implementation
   clarity.  A production deployment would use SealPIR or FastPIR.

The module also exposes ``PIRServer`` and ``PIRClient`` as the canonical
two-party API used by higher-level code (``encrypted_lookup.py``).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

import tenseal as ts


# ===================================================================
# Parameters
# ===================================================================

@dataclass
class PIRParams:
    """Parameters for the PIR scheme.

    Attributes
    ----------
    db_size : int
        Number of records in the database.
    record_size_bytes : int
        Maximum size of each serialized record (records are padded to
        this length).
    poly_modulus_degree : int
        BFV polynomial ring dimension.  Larger values support more
        slots but increase ciphertext size.
    plain_modulus : int
        BFV plaintext modulus.  Must exceed 2^16 = 65 536 to support
        2-byte packing.
    bytes_per_slot : int
        Number of record bytes packed into each BFV plaintext slot.
    """

    db_size: int
    record_size_bytes: int = 512
    poly_modulus_degree: int = 8192
    plain_modulus: int = 1_032_193
    bytes_per_slot: int = 2
    # Derived — do not set manually
    security_level: int = 128

    @property
    def n_chunks(self) -> int:
        """Number of BFV column-multiplications needed per query."""
        return math.ceil(self.record_size_bytes / self.bytes_per_slot)


# ===================================================================
# Record encoding / decoding
# ===================================================================

def _encode_record(data: bytes, record_size: int, bytes_per_slot: int) -> list[int]:
    """Encode a byte-string record into a list of packed integers.

    Each integer packs ``bytes_per_slot`` consecutive bytes using
    little-endian base-256 encoding.  The record is right-padded with
    zeros to ``record_size`` before encoding.

    >>> _encode_record(b'\\x01\\x02\\x03', record_size=4, bytes_per_slot=2)
    [513, 3]
    """
    padded = data.ljust(record_size, b"\x00")
    chunks: list[int] = []
    for i in range(0, record_size, bytes_per_slot):
        val = 0
        for j in range(bytes_per_slot):
            if i + j < len(padded):
                val += padded[i + j] << (8 * j)
        chunks.append(val)
    return chunks


def _decode_record(chunks: list[int], record_size: int, bytes_per_slot: int) -> bytes:
    """Inverse of ``_encode_record``.

    >>> _decode_record([513, 3], record_size=4, bytes_per_slot=2)
    b'\\x01\\x02\\x03\\x00'
    """
    raw = bytearray()
    for val in chunks:
        for j in range(bytes_per_slot):
            raw.append((val >> (8 * j)) & 0xFF)
    return bytes(raw[:record_size])


# ===================================================================
# Trivial PIR (baseline)
# ===================================================================

class TrivialPIR:
    """Trivial PIR: download the entire database.

    Perfectly private (server learns nothing) but bandwidth = O(n).
    Used as a correctness and privacy baseline.
    """

    def __init__(self, database: list[bytes]) -> None:
        self.database = list(database)

    def query(self, index: int) -> bytes:
        """Retrieve a record by downloading everything.

        In a real deployment the full database is transferred over the
        network; here we simply simulate the selection locally.
        """
        if not 0 <= index < len(self.database):
            raise IndexError(f"Index {index} out of range [0, {len(self.database)})")
        # The "transfer" is the full database; client selects locally.
        local_copy = list(self.database)
        return local_copy[index]

    def communication_cost(self) -> int:
        """Total bytes transferred (entire database)."""
        return sum(len(r) for r in self.database)


# ===================================================================
# PIR Server (holds database, processes encrypted queries)
# ===================================================================

class PIRServer:
    """Server-side component of the computational PIR scheme.

    The server holds the database encoded as a matrix of packed
    integers (one column per chunk position, one row per record).
    When it receives an encrypted selection vector from the client,
    it multiplies the vector element-wise with each column and returns
    the encrypted column results.

    The server never possesses the secret key and therefore cannot
    decrypt the selection vector or the result — it learns *nothing*
    about which record was requested.
    """

    def __init__(self) -> None:
        self._ctx: ts.Context | None = None
        self._db_columns: list[list[int]] = []
        self._params: PIRParams | None = None
        # Observability log — records what the server can see
        self.query_log: list[dict[str, Any]] = []

    def load_context(self, serialized_ctx: bytes) -> None:
        """Load the public encryption context (no secret key)."""
        self._ctx = ts.context_from(serialized_ctx)

    def load_database(self, records: list[bytes], params: PIRParams) -> None:
        """Encode and store the database.

        Parameters
        ----------
        records : list[bytes]
            Serialized records (each <= ``params.record_size_bytes``).
        params : PIRParams
            Scheme parameters.
        """
        self._params = params
        n = params.db_size
        n_chunks = params.n_chunks

        # Build column-major matrix: columns[chunk_idx][record_idx]
        self._db_columns = [[] for _ in range(n_chunks)]
        for rec in records:
            encoded = _encode_record(rec, params.record_size_bytes, params.bytes_per_slot)
            for c, val in enumerate(encoded):
                self._db_columns[c].append(val)

        # Pad columns to db_size if fewer records were provided
        for c in range(n_chunks):
            while len(self._db_columns[c]) < n:
                self._db_columns[c].append(0)

    def process_query(self, encrypted_query_bytes: bytes) -> list[bytes]:
        """Process an encrypted PIR query.

        Parameters
        ----------
        encrypted_query_bytes : bytes
            Serialized BFV ciphertext of the selection vector.

        Returns
        -------
        list[bytes]
            One serialized ciphertext per chunk column (the client
            decrypts and reassembles the record).
        """
        if self._ctx is None:
            raise RuntimeError("Server context not loaded")
        if not self._db_columns:
            raise RuntimeError("Database not loaded")

        t_start = time.perf_counter()

        query_vec = ts.bfv_vector_from(self._ctx, encrypted_query_bytes)
        result_parts: list[bytes] = []
        for col in self._db_columns:
            product = query_vec * col          # element-wise encrypted * plain
            result_parts.append(product.serialize())

        t_end = time.perf_counter()

        # Log what the server observed (metadata only — never the index)
        self.query_log.append({
            "timestamp": time.time(),
            "query_ciphertext_size": len(encrypted_query_bytes),
            "response_parts": len(result_parts),
            "response_total_bytes": sum(len(p) for p in result_parts),
            "processing_time_ms": (t_end - t_start) * 1000,
            "visible_index": None,  # server cannot determine this
        })

        return result_parts


# ===================================================================
# PIR Client (generates queries, decrypts responses)
# ===================================================================

class PIRClient:
    """Client-side component of the computational PIR scheme.

    The client creates the BFV encryption context (and therefore holds
    the secret key), generates encrypted selection vectors, and decrypts
    the server's responses.
    """

    def __init__(self, params: PIRParams) -> None:
        self.params = params
        self._ctx: ts.Context | None = None
        self._full_ctx_bytes: bytes = b""

    def setup(self) -> bytes:
        """Initialize the BFV context and return the public context.

        Returns
        -------
        bytes
            Serialized public context (without secret key) for the
            server.
        """
        self._ctx = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=self.params.poly_modulus_degree,
            plain_modulus=self.params.plain_modulus,
        )
        # Save full context (with secret key) for decryption
        self._full_ctx_bytes = self._ctx.serialize(save_secret_key=True)
        # Make public copy for the server
        self._ctx.make_context_public()
        public_bytes = self._ctx.serialize()
        # Restore full context for client use
        self._ctx = ts.context_from(self._full_ctx_bytes)
        return public_bytes

    def generate_query(self, index: int) -> bytes:
        """Create an encrypted one-hot selection vector.

        Parameters
        ----------
        index : int
            The database index to retrieve (secret — not revealed).

        Returns
        -------
        bytes
            Serialized BFV ciphertext.
        """
        if self._ctx is None:
            raise RuntimeError("Call setup() first")
        if not 0 <= index < self.params.db_size:
            raise IndexError(f"Index {index} out of range [0, {self.params.db_size})")

        selector = [0] * self.params.db_size
        selector[index] = 1
        enc = ts.bfv_vector(self._ctx, selector)
        return enc.serialize()

    def decrypt_response(self, response_parts: list[bytes]) -> bytes:
        """Decrypt the server's response and reconstruct the record.

        Parameters
        ----------
        response_parts : list[bytes]
            One serialized ciphertext per chunk column from the server.

        Returns
        -------
        bytes
            The plaintext record bytes.
        """
        if self._ctx is None:
            raise RuntimeError("Call setup() first")

        chunks: list[int] = []
        for part_bytes in response_parts:
            vec = ts.bfv_vector_from(self._ctx, part_bytes)
            slots = vec.decrypt()
            # The inner product is the sum of the element-wise products.
            # Because the selector was one-hot, only slot[index] is non-zero,
            # so sum(slots) == db_column[index].
            total = sum(slots)
            # Handle potential negative wraparound from BFV modular arithmetic
            mod = self.params.plain_modulus
            pack_mod = 1 << (8 * self.params.bytes_per_slot)
            total = total % mod
            if total >= pack_mod:
                total = total % pack_mod
            chunks.append(total)

        return _decode_record(chunks, self.params.record_size_bytes, self.params.bytes_per_slot)

    def communication_cost(self) -> dict[str, int]:
        """Estimate communication cost in bytes."""
        if self._ctx is None:
            raise RuntimeError("Call setup() first")
        # Query: one BFV ciphertext (size depends on poly_modulus_degree)
        selector = [0] * self.params.db_size
        enc = ts.bfv_vector(self._ctx, selector)
        query_size = len(enc.serialize())
        # Response: n_chunks ciphertexts (same size as query)
        response_size = query_size * self.params.n_chunks
        return {
            "query_size": query_size,
            "response_size": response_size,
            "total": query_size + response_size,
        }


# ===================================================================
# Convenience wrapper (backward-compatible with old ComputationalPIR)
# ===================================================================

class ComputationalPIR:
    """Unified PIR wrapper combining PIRClient and PIRServer.

    This class provides the combined interface expected by existing code
    (``encrypted_lookup.py``, ``run_benchmarks.py``, and tests).
    Internally it delegates to ``PIRClient`` and ``PIRServer``.
    """

    def __init__(self, params: PIRParams) -> None:
        self.params = params
        self.client = PIRClient(params)
        self.server = PIRServer()
        self._is_setup = False

    def setup(self) -> ts.Context:
        """Initialize encryption and return a TenSEAL context object.

        Returns
        -------
        ts.Context
            The public context (without secret key).
        """
        public_ctx_bytes = self.client.setup()
        self.server.load_context(public_ctx_bytes)
        self._is_setup = True
        return ts.context_from(public_ctx_bytes)

    def load_database(self, records: list[bytes]) -> None:
        """Load the database on the server side."""
        if not self._is_setup:
            raise RuntimeError("Call setup() first")
        self.server.load_database(records, self.params)

    def generate_query(self, index: int) -> bytes:
        """Client generates an encrypted query for a given index."""
        return self.client.generate_query(index)

    def process_query(self, encrypted_query: bytes) -> list[bytes]:
        """Server processes the encrypted query."""
        return self.server.process_query(encrypted_query)

    def decrypt_result(self, encrypted_result: list[bytes]) -> bytes:
        """Client decrypts the server's response."""
        return self.client.decrypt_response(encrypted_result)

    def end_to_end_query(self, index: int) -> bytes:
        """Run a full query round-trip (client -> server -> client)."""
        q = self.generate_query(index)
        r = self.process_query(q)
        return self.decrypt_result(r)

    def communication_cost(self) -> dict[str, int]:
        """Estimate communication cost in bytes."""
        return self.client.communication_cost()
