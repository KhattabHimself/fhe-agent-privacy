"""FHE-based encrypted computation for tool-call privacy.

This module wraps TenSEAL to provide encrypted arithmetic operations
that a tool server can perform on ciphertext without learning the
plaintext values.  The primary use case is risk scoring: the agent
encrypts patient features, and the server evaluates a scoring model
on the ciphertext, returning an encrypted score only the agent can
decrypt.

Supported operations:
- Encrypted vector addition / subtraction
- Encrypted vector–scalar multiplication
- Encrypted dot product (for linear models)
- Encrypted polynomial evaluation (for simple nonlinear models)

CKKS parameter choices
----------------------
CKKS is an *approximate* HE scheme — small rounding errors accumulate
with each operation.  Two knobs control the accuracy/capability trade-off:

* **poly_modulus_degree** — determines (a) max number of "slots" and
  (b) the total bit-budget for the coefficient modulus chain.

  - 8192  →  218-bit budget, ~4096 slots.  Enough for depth-2 operations
    (dot product OR quadratic polynomial, but usually not both).
  - 16384 →  438-bit budget, ~8192 slots.  Enough for depth-4, which
    covers dot-product + cubic polynomial activation.

* **coeff_mod_bit_sizes** — the modulus chain.  First and last entries
  are "special" primes; middle entries each provide one multiplicative
  level.  ``global_scale = 2^(middle-prime bits)`` keeps precision
  balanced across levels.

We provide two presets:

+------------------+-------+----------------------------+-------+-------+
| Preset           | N     | coeff_mod_bit_sizes        | Depth | Use   |
+------------------+-------+----------------------------+-------+-------+
| LIGHT (default)  | 8192  | [60, 40, 40, 60]           |   2   | dot   |
| DEEP             | 16384 | [60, 40, 40, 40, 40, 60]   |   4   | dot + |
|                  |       |                            |       | poly  |
+------------------+-------+----------------------------+-------+-------+

Both provide 128-bit security (per the SEAL/HE standard tables).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

import tenseal as ts


# ===================================================================
# Parameter presets
# ===================================================================

def _light_params() -> "FHEParams":
    """Preset for depth-2 operations (dot product, quadratic)."""
    return FHEParams(
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
        global_scale=2**40,
    )


def _deep_params() -> "FHEParams":
    """Preset for depth-4 operations (dot product + cubic polynomial)."""
    return FHEParams(
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60],
        global_scale=2**40,
    )


# ===================================================================
# FHEParams
# ===================================================================

@dataclass
class FHEParams:
    """Parameters for the CKKS FHE scheme.

    Attributes
    ----------
    poly_modulus_degree : int
        Ring dimension.  8192 for light workloads, 16384 for deeper
        computation chains.
    coeff_mod_bit_sizes : list[int]
        Coefficient modulus chain.  The number of *middle* entries
        determines the multiplicative depth.
    global_scale : float
        CKKS encoding scale.  Should equal ``2**<middle-prime bits>``
        (e.g. 2**40 when the middle primes are 40 bits each).
    security_level : int
        Target security level in bits (128 for both presets).
    """

    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: list[int] | None = None
    global_scale: float = 2**40
    security_level: int = 128

    def __post_init__(self) -> None:
        if self.coeff_mod_bit_sizes is None:
            self.coeff_mod_bit_sizes = [60, 40, 40, 60]

    @property
    def multiplicative_depth(self) -> int:
        """Available multiplicative depth (middle primes count)."""
        assert self.coeff_mod_bit_sizes is not None
        return len(self.coeff_mod_bit_sizes) - 2

    @classmethod
    def light(cls) -> FHEParams:
        """Depth-2 preset (8192, suitable for dot products)."""
        return _light_params()

    @classmethod
    def deep(cls) -> FHEParams:
        """Depth-4 preset (16384, suitable for dot + polynomial)."""
        return _deep_params()


# ===================================================================
# FHEContext
# ===================================================================

class FHEContext:
    """Manage TenSEAL CKKS context for encrypted computation.

    Provides a high-level interface for creating encryption contexts,
    encrypting/decrypting vectors, and performing homomorphic operations.
    The class maintains both a full context (with secret key, for the
    client/agent) and can produce a public-only context for the server.
    """

    def __init__(self, params: FHEParams | None = None) -> None:
        self.params = params or FHEParams()
        self._context: ts.Context | None = None
        self._full_context_bytes: bytes = b""

    def setup(self) -> None:
        """Initialize the CKKS context with the configured parameters."""
        self._context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.params.poly_modulus_degree,
            coeff_mod_bit_sizes=self.params.coeff_mod_bit_sizes,
        )
        self._context.global_scale = self.params.global_scale
        self._context.generate_galois_keys()
        # Save the full context (with secret key) for decryption
        self._full_context_bytes = self._context.serialize(save_secret_key=True)

    def _require_ctx(self) -> ts.Context:
        if self._context is None:
            raise RuntimeError("Call setup() first")
        return self._context

    # ------------------------------------------------------------------
    # Context serialization
    # ------------------------------------------------------------------

    @property
    def public_context(self) -> ts.Context:
        """Return a context without the secret key (for the server)."""
        ctx = self._require_ctx()
        pub_bytes = self._public_context_bytes()
        return ts.context_from(pub_bytes)

    def _public_context_bytes(self) -> bytes:
        """Serialize the public context (no secret key)."""
        ctx = self._require_ctx()
        # Make a copy, strip secret key, serialize, then restore full ctx
        ctx.make_context_public()
        pub_bytes = ctx.serialize()
        # Restore full context from saved bytes
        self._context = ts.context_from(self._full_context_bytes)
        return pub_bytes

    def public_context_bytes(self) -> bytes:
        """Serialize the public context as bytes for network transfer."""
        return self._public_context_bytes()

    # ------------------------------------------------------------------
    # Encrypt / decrypt
    # ------------------------------------------------------------------

    def encrypt_vector(self, plaintext: list[float] | NDArray[np.float64]) -> ts.CKKSVector:
        """Encrypt a plaintext vector using CKKS.

        Parameters
        ----------
        plaintext : list[float] or ndarray
            Vector of real numbers to encrypt.

        Returns
        -------
        ts.CKKSVector
            Encrypted CKKS vector.
        """
        ctx = self._require_ctx()
        if isinstance(plaintext, np.ndarray):
            plaintext = plaintext.tolist()
        return ts.ckks_vector(ctx, plaintext)

    def decrypt_vector(self, ciphertext: ts.CKKSVector) -> list[float]:
        """Decrypt a CKKS ciphertext back to plaintext.

        If the current context has been replaced with a public-only copy
        (e.g. after calling ``public_context``), this restores the full
        context from the saved bytes before decrypting.
        """
        # Ensure we have the secret key
        if self._full_context_bytes:
            self._context = ts.context_from(self._full_context_bytes)
        ctx = self._require_ctx()
        # Re-link the ciphertext to our full context for decryption
        ct_bytes = ciphertext.serialize()
        local_ct = ts.ckks_vector_from(ctx, ct_bytes)
        return local_ct.decrypt()

    def decrypt_result(self, ciphertext: ts.CKKSVector) -> float:
        """Decrypt a scalar result (first slot of the ciphertext)."""
        values = self.decrypt_vector(ciphertext)
        return values[0]

    # ------------------------------------------------------------------
    # Homomorphic operations
    # ------------------------------------------------------------------

    def encrypted_dot_product(
        self,
        enc_vec: ts.CKKSVector,
        plain_weights: list[float] | NDArray[np.float64],
    ) -> ts.CKKSVector:
        """Compute the dot product of an encrypted vector with plaintext weights.

        Parameters
        ----------
        enc_vec : ts.CKKSVector
            Encrypted input vector.
        plain_weights : list[float] or ndarray
            Plaintext weight vector (same length as enc_vec).

        Returns
        -------
        ts.CKKSVector
            Encrypted scalar (dot product result in slot 0).
        """
        if isinstance(plain_weights, np.ndarray):
            plain_weights = plain_weights.tolist()
        return enc_vec.dot(plain_weights)

    def encrypted_polynomial_eval(
        self,
        enc_input: ts.CKKSVector,
        coefficients: list[float],
    ) -> ts.CKKSVector:
        """Evaluate a polynomial on an encrypted input using TenSEAL's polyval.

        The polynomial is ``c[0] + c[1]*x + c[2]*x^2 + ...``.

        Parameters
        ----------
        enc_input : ts.CKKSVector
            Encrypted scalar (or vector — polynomial is applied element-wise).
        coefficients : list[float]
            Polynomial coefficients ``[c0, c1, c2, ...]``.

        Returns
        -------
        ts.CKKSVector
            Encrypted polynomial result.

        Notes
        -----
        Each degree of the polynomial consumes one multiplicative level.
        Ensure the context was created with enough depth.  A degree-d
        polynomial needs ``d`` middle primes in ``coeff_mod_bit_sizes``.
        """
        return enc_input.polyval(coefficients)


# ===================================================================
# EncryptedLinearModel
# ===================================================================

class EncryptedLinearModel:
    """Evaluate a linear model on encrypted inputs.

    Given plaintext weights *w* and encrypted input *x*, computes::

        score = w · x + b

    entirely in the encrypted domain.  The server learns nothing about
    *x* or the resulting score.
    """

    def __init__(self, weights: list[float] | NDArray[np.float64], bias: float) -> None:
        if isinstance(weights, np.ndarray):
            self.weights = weights.tolist()
        else:
            self.weights = list(weights)
        self.bias = bias

    def predict_encrypted(self, encrypted_input: ts.CKKSVector) -> ts.CKKSVector:
        """Compute encrypted prediction: w · x + b.

        Parameters
        ----------
        encrypted_input : ts.CKKSVector
            Encrypted feature vector (same length as weights).

        Returns
        -------
        ts.CKKSVector
            Encrypted scalar score.
        """
        dot = encrypted_input.dot(self.weights)
        return dot + self.bias

    def predict_plaintext(self, features: list[float] | NDArray[np.float64]) -> float:
        """Plaintext prediction (baseline)."""
        return float(np.dot(features, self.weights) + self.bias)


# ===================================================================
# EncryptedPolynomialModel
# ===================================================================

class EncryptedPolynomialModel:
    """Evaluate a polynomial model on encrypted inputs.

    Computes ``c[0] + c[1]*x + c[2]*x^2 + ...`` using TenSEAL's
    ``polyval``, which internally applies an optimized evaluation
    strategy (baby-step giant-step or Horner's method).

    Common use cases:
    - Sigmoid approximation: ``[0.5, 0.197, 0.0, -0.004]``
    - Quadratic risk scoring: ``[c0, c1, c2]``
    """

    def __init__(self, coefficients: list[float]) -> None:
        """Initialize with polynomial coefficients ``[c0, c1, c2, ...]``."""
        self.coefficients = list(coefficients)

    def evaluate_encrypted(self, encrypted_input: ts.CKKSVector) -> ts.CKKSVector:
        """Evaluate the polynomial on an encrypted input.

        Parameters
        ----------
        encrypted_input : ts.CKKSVector
            Encrypted scalar or vector.

        Returns
        -------
        ts.CKKSVector
            Encrypted polynomial evaluation result.
        """
        return encrypted_input.polyval(self.coefficients)

    def evaluate_plaintext(self, x: float) -> float:
        """Plaintext evaluation (baseline)."""
        return float(sum(c * x**i for i, c in enumerate(self.coefficients)))

    @property
    def degree(self) -> int:
        """Polynomial degree (highest non-zero power)."""
        for i in range(len(self.coefficients) - 1, -1, -1):
            if self.coefficients[i] != 0.0:
                return i
        return 0

    @property
    def multiplicative_depth(self) -> int:
        """Multiplicative depth needed to evaluate this polynomial."""
        return max(0, len(self.coefficients) - 1)
