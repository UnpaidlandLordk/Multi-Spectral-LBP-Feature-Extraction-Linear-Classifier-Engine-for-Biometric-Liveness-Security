"""Frame -> histogram -> score -> decision, exactly as the hardware does it.

This is the object the RTL is compared against. One method uses the vectorised
LBP (fast, for bulk work); the other walks pixel by pixel (golden, for proving
the fast path and for the RTL test vectors).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fixed_point as fp
from .classifier import Q14Model
from .lbp import interior_pixels, lbp_codes, lbp_histogram, slow_histogram

FRAME_SIZE = 130            # interior 128 x 128
N_PIXELS = interior_pixels(FRAME_SIZE, FRAME_SIZE)   # 16384 = 2**14

# cycle cost of the streaming engine: one pixel per cycle per channel
# plus a fixed 16-cycle FSM overhead (fill + normalise + classify + decide)
CYCLE_OVERHEAD = 16


@dataclass
class FrameResult:
    hist0: np.ndarray
    hist1: np.ndarray
    features: np.ndarray
    acc: int
    score: int
    decision: int          # 1 = live, 0 = spoof
    method: str = "fast"

    def as_dict(self) -> dict:
        return {
            "score": int(self.score),
            "decision": int(self.decision),
            "acc": int(self.acc),
            "hist0": [int(v) for v in self.hist0],
            "hist1": [int(v) for v in self.hist1],
        }


def validate_frame(img, frame_size: int = FRAME_SIZE) -> np.ndarray:
    a = np.asarray(img)
    if a.shape != (frame_size, frame_size):
        raise ValueError(f"expected a {frame_size}x{frame_size} frame, got {a.shape}")
    if a.dtype != np.uint8:
        raise ValueError(f"expected uint8 pixels, got {a.dtype}")
    return a


class LivenessEngine:
    """Reference implementation of liveness_top.sv."""

    def __init__(self, model: Q14Model, frame_size: int = FRAME_SIZE):
        self.model = model
        self.frame_size = frame_size
        self.n_pixels = interior_pixels(frame_size, frame_size)
        if self.n_pixels != N_PIXELS:
            raise ValueError(
                "this design assumes a 130x130 frame so that N = 2**14 and "
                "histogram normalisation is a wiring shift"
            )

    # -- features ---------------------------------------------------------
    def histograms(self, ch0, ch1, method: str = "fast"):
        a = validate_frame(ch0, self.frame_size)
        b = validate_frame(ch1, self.frame_size)
        if method == "fast":
            return lbp_histogram(a), lbp_histogram(b)
        if method == "golden":
            return slow_histogram(a), slow_histogram(b)
        raise ValueError(f"unknown method {method!r} (use 'fast' or 'golden')")

    def features(self, ch0, ch1, method: str = "fast") -> np.ndarray:
        h0, h1 = self.histograms(ch0, ch1, method)
        return fp.bit_exact_features(h0, h1, self.n_pixels)

    # -- full pipeline ----------------------------------------------------
    def process(self, ch0, ch1, method: str = "fast") -> FrameResult:
        h0, h1 = self.histograms(ch0, ch1, method)
        feats = fp.bit_exact_features(h0, h1, self.n_pixels)
        acc = fp.dot_q14(feats, self.model.weights) + (int(self.model.bias) << fp.Q)
        score = fp.acc_to_q14(acc)
        return FrameResult(
            hist0=h0,
            hist1=h1,
            features=feats,
            acc=acc,
            score=score,
            decision=1 if score >= self.model.threshold else 0,
            method=method,
        )

    # -- timing model used for the latency plot ---------------------------
    def cycles(self, channels: int = 2) -> int:
        return channels * self.frame_size * self.frame_size + CYCLE_OVERHEAD

    def latency_us(self, clock_mhz: float = 150.0, channels: int = 2) -> float:
        return self.cycles(channels) / (clock_mhz * 1e6) * 1e6
