"""Cryptographic utility functions.

Shared helpers for key management, serialization, random number
generation, and hash-based operations used across the project.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

import numpy as np
from numpy.typing import NDArray

import tenseal as ts
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


def generate_aes_key(key_size: int = 256) -> bytes:
    """Generate a random AES key.

    Args:
        key_size: Key size in bits (128, 192, or 256).

    Returns:
        Random key bytes.

    Raises:
        ValueError: If key_size is not 128, 192, or 256.
    """
    if key_size not in (128, 192, 256):
        raise ValueError(f"key_size must be 128, 192, or 256, got {key_size}")
    return secrets.token_bytes(key_size // 8)


def aes_encrypt(plaintext: bytes, key: bytes) -> tuple[bytes, bytes, bytes]:
    """Encrypt data with AES-GCM.

    Args:
        plaintext: Data to encrypt.
        key: AES key.

    Returns:
        Tuple of (nonce, ciphertext, tag).
    """
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return cipher.nonce, ciphertext, tag


def aes_decrypt(nonce: bytes, ciphertext: bytes, tag: bytes, key: bytes) -> bytes:
    """Decrypt AES-GCM ciphertext.

    Args:
        nonce: GCM nonce.
        ciphertext: Encrypted data.
        tag: Authentication tag.
        key: AES key.

    Returns:
        Decrypted plaintext.

    Raises:
        ValueError: If authentication fails.
    """
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)


def serialize_context(context: ts.Context, include_secret: bool = False) -> bytes:
    """Serialize a TenSEAL context.

    Args:
        context: TenSEAL context to serialize.
        include_secret: Whether to include the secret key.

    Returns:
        Serialized context bytes.
    """
    if not include_secret:
        context.make_context_public()
    return context.serialize(save_secret_key=include_secret)


def deserialize_context(data: bytes) -> ts.Context:
    """Deserialize a TenSEAL context from bytes.

    Args:
        data: Serialized context.

    Returns:
        Reconstructed TenSEAL context.
    """
    return ts.context_from(data)


def deterministic_hash(data: str, length: int = 32) -> bytes:
    """Compute a deterministic hash for use as a PRF.

    Args:
        data: Input string.
        length: Output length in bytes.

    Returns:
        Hash digest.
    """
    return hashlib.sha256(data.encode()).digest()[:length]


def encode_as_integers(
    data: bytes, modulus: int,
) -> NDArray[np.int64]:
    """Encode byte data as a vector of integers mod ``modulus``.

    Used for BFV scheme encoding where plaintexts must be integers.
    Each byte becomes one element in the output vector, already
    in ``[0, modulus)`` since byte values are in ``[0, 255]``.

    Args:
        data: Raw byte data.
        modulus: Plaintext modulus for the BFV scheme.

    Returns:
        Integer vector with values in ``[0, modulus)``.
    """
    return np.array([b % modulus for b in data], dtype=np.int64)


def decode_from_integers(
    integers: NDArray[np.int64], modulus: int, length: int,
) -> bytes:
    """Decode an integer vector back to bytes.

    Inverse of ``encode_as_integers``.

    Args:
        integers: Integer vector.
        modulus: Plaintext modulus used during encoding.
        length: Expected output byte length.

    Returns:
        Decoded byte data.
    """
    raw = bytes(int(v) % 256 for v in integers[:length])
    return raw
