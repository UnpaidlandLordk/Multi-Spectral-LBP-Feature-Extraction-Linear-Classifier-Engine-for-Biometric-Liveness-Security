"""Q2.14 fixed-point primitives.

Everything the hardware does to numbers is reproduced here with integers only,
so simulation and RTL can be compared bit-for-bit.

Conventions
-----------
Q2.14  : 16-bit signed, 2 integer bits, 14 fractional bits
         ONE = 16384, range [-2.0, +1.99994], resolution 2**-14
round  : half away from zero (symmetric, no sign-dependent bias)
acc    : Q4.28 (product of two Q2.14 values), stored in a 48-bit register
"""

from __future__ import annotations

import numpy as np

Q = 14
ONE = 1 << Q
QMAX = (1 << 15) - 1      # +32767
QMIN = -(1 << 15)         # -32768
ACC_BITS = 48
ACC_MAX = (1 << (ACC_BITS - 1)) - 1


def sat16(values):
    """Saturate to the signed 16-bit range used by every Q2.14 register."""
    return np.clip(np.asarray(values, dtype=np.int64), QMIN, QMAX).astype(np.int64)


def div_round_half_away(a, b: int):
    """Integer division rounded half away from zero (b > 0)."""
    if b <= 0:
        raise ValueError("denominator must be positive")
    a = np.asarray(a, dtype=np.int64)
    q = (2 * np.abs(a) + b) // (2 * b)
    return np.where(a < 0, -q, q).astype(np.int64)


def from_float(values) -> np.ndarray:
    """Real -> Q2.14, round-half-away, saturating."""
    x = np.asarray(values, dtype=np.float64) * ONE
    rounded = np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))
    return sat16(rounded.astype(np.int64))


def to_float(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / ONE


def hist_to_q14(counts, n_pixels: int) -> np.ndarray:
    """Normalise raw histogram counts to Q2.14.

    n_pixels = 16384 = 2**14 makes this a no-op: p * 2**14 == count exactly.
    With any other n_pixels a rounding divide is used (hardware would need a
    reciprocal multiply), which is why the frame is sized to 130 x 130.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if n_pixels <= 0:
        raise ValueError("n_pixels must be positive")
    return sat16(div_round_half_away(counts * ONE, n_pixels))


def dot_q14(features, weights) -> int:
    """Q2.14 x Q2.14 partial products, summed in the wide accumulator."""
    f = np.asarray(features, dtype=np.int64)
    w = np.asarray(weights, dtype=np.int64)
    if f.shape != w.shape:
        raise ValueError(f"shape mismatch: features {f.shape} vs weights {w.shape}")
    acc = int(np.dot(f, w))
    if not (-ACC_MAX - 1 <= acc <= ACC_MAX):
        raise OverflowError(f"accumulator overflow: {acc} exceeds {ACC_BITS} bits")
    return acc


def acc_to_q14(acc: int) -> int:
    """Q4.28 accumulator -> Q2.14 score, one rounding and one saturation."""
    acc = int(acc)
    if not (-ACC_MAX - 1 <= acc <= ACC_MAX):
        raise OverflowError(f"accumulator overflow: {acc} exceeds {ACC_BITS} bits")
    return int(sat16(div_round_half_away(acc, 1 << Q)))


def quantize_weights(w_float, max_abs: float = 1.9):
    """Scale a float weight vector into Q2.14 without clipping.

    Returns (w_q14, scale). The same scale must be applied to the bias and the
    threshold; a global scale leaves the decision boundary unchanged.
    """
    w = np.asarray(w_float, dtype=np.float64)
    peak = float(np.max(np.abs(w))) if w.size else 0.0
    scale = 1.0 if peak <= max_abs else max_abs / peak
    return from_float(w * scale), scale


def bit_exact_features(counts0, counts1, n_pixels: int = 1 << 14) -> np.ndarray:
    """Concatenate per-channel histograms into the 512-value Q2.14 feature vector."""
    h0 = hist_to_q14(counts0, n_pixels)
    h1 = hist_to_q14(counts1, n_pixels)
    return np.concatenate([h0, h1]).astype(np.int64)
