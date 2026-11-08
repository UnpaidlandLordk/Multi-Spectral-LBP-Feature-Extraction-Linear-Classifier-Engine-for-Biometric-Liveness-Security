"""Local Binary Pattern reference implementation.

Two implementations of the same definition:

* ``lbp_codes``        vectorised NumPy - used for training and bulk work
* ``iter_codes_slow``  literal per-pixel loops - the golden model, used to
                       prove the fast path and (later) the RTL

Definition (frozen):
    bit 0 = top-left, then clockwise: top, top-right, right, bottom-right,
    bottom, bottom-left, left
    bit_i = 1 if neighbour_i >= centre else 0
    border pixels are skipped, so a 130 x 130 frame yields 128 x 128 codes
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

# (dy, dx), clockwise starting at top-left, bit 0 first
OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),  # 0 top-left
    (-1, 0),   # 1 top
    (-1, 1),   # 2 top-right
    (0, 1),    # 3 right
    (1, 1),    # 4 bottom-right
    (1, 0),    # 5 bottom
    (1, -1),   # 6 bottom-left
    (0, -1),   # 7 left
)

N_BINS = 256


def _as_image(img) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D greyscale image, got shape {a.shape}")
    if a.shape[0] < 3 or a.shape[1] < 3:
        raise ValueError("image must be at least 3x3")
    return a


def lbp_code_at(img, y: int, x: int) -> int:
    """LBP code of one interior pixel. y is the row, x the column."""
    a = _as_image(img)
    c = int(a[y, x])
    code = 0
    for i, (dy, dx) in enumerate(OFFSETS):
        if int(a[y + dy, x + dx]) >= c:
            code |= 1 << i
    return code


def iter_codes_slow(img) -> Iterator[Tuple[int, int, int]]:
    """Golden model: raster-scan, one python comparison per neighbour."""
    a = _as_image(img)
    h, w = a.shape
    for y in range(1, h - 1):
        row = a[y]
        for x in range(1, w - 1):
            c = int(row[x])
            code = 0
            if int(a[y - 1, x - 1]) >= c: code |= 1
            if int(a[y - 1, x])     >= c: code |= 2
            if int(a[y - 1, x + 1]) >= c: code |= 4
            if int(a[y,     x + 1]) >= c: code |= 8
            if int(a[y + 1, x + 1]) >= c: code |= 16
            if int(a[y + 1, x])     >= c: code |= 32
            if int(a[y + 1, x - 1]) >= c: code |= 64
            if int(a[y,     x - 1]) >= c: code |= 128
            yield y, x, code


def lbp_codes(img) -> np.ndarray:
    """Vectorised LBP over the interior; returns uint8 array (H-2, W-2)."""
    a = _as_image(img)
    h, w = a.shape
    centre = a[1:h - 1, 1:w - 1]
    code = np.zeros(centre.shape, dtype=np.uint16)
    for i, (dy, dx) in enumerate(OFFSETS):
        neighbour = a[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        code |= (neighbour >= centre).astype(np.uint16) << i
    return code.astype(np.uint8)


def histogram(codes) -> np.ndarray:
    """256-bin histogram of an array of LBP codes."""
    flat = np.asarray(codes, dtype=np.uint8).ravel()
    return np.bincount(flat, minlength=N_BINS).astype(np.int64)


def lbp_histogram(img) -> np.ndarray:
    return histogram(lbp_codes(img))


def slow_histogram(img) -> np.ndarray:
    """Histogram built from the golden-model iterator (used only in tests)."""
    counts = np.zeros(N_BINS, dtype=np.int64)
    for _y, _x, code in iter_codes_slow(img):
        counts[code] += 1
    return counts


def interior_pixels(height: int, width: int) -> int:
    return (height - 2) * (width - 2)
