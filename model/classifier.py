"""Fixed-point linear classifier: training, quantisation, inference.

The reference model trains with a float linear SVM, then quantises weights and
bias to Q2.14 and *re-evaluates* with the integer datapath, so the reported
metrics describe the hardware, not the float model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import fixed_point as fp
from .lbp import N_BINS

FEATURE_LEN = 2 * N_BINS
LIVE, SPOOF = 1, 0


@dataclass
class Q14Model:
    """Deployable classifier. All arithmetic is integer at inference time."""

    weights: np.ndarray                 # int16[512], Q2.14
    bias: int = 0                       # Q2.14
    threshold: int = 0                  # Q2.14, score >= threshold -> live
    scale: float = 1.0                  # global scale applied to float weights
    meta: Dict = field(default_factory=dict)

    def score(self, features) -> int:
        if features.shape[0] != FEATURE_LEN:
            raise ValueError(f"expected {FEATURE_LEN} features, got {features.shape[0]}")
        acc = fp.dot_q14(features, self.weights)
        acc += int(self.bias) << fp.Q      # bias is Q2.14, accumulator is Q4.28
        return fp.acc_to_q14(acc)

    def decide(self, features) -> int:
        return LIVE if self.score(features) >= self.threshold else SPOOF

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": "q14-linear-v1",
                "q": fp.Q,
                "feature_len": int(self.weights.size),
                "feature_layout": "ch0 bins 0..255, then ch1 bins 0..255",
                "weights": [int(v) for v in self.weights],
                "bias": int(self.bias),
                "threshold": int(self.threshold),
                "scale": float(self.scale),
                "meta": self.meta,
            },
            indent=2,
        )

    def save(self, path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Q14Model":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != "q14-linear-v1":
            raise ValueError(f"unsupported model format: {d.get('format')}")
        return cls(
            weights=np.asarray(d["weights"], dtype=np.int64),
            bias=int(d["bias"]),
            threshold=int(d["threshold"]),
            scale=float(d.get("scale", 1.0)),
            meta=d.get("meta", {}),
        )


def train_svm(features: np.ndarray, labels: np.ndarray, *, seed: int = 0, C: float = 0.05):
    """Train a linear SVM on Q2.14-normalised features.

    features are integers (histogram counts); dividing by ONE gives values in
    [0, 1], which keeps the SVM's regularisation meaningful across frame sizes.
    """
    from sklearn.svm import LinearSVC

    x = np.asarray(features, dtype=np.float64) / fp.ONE
    y = np.asarray(labels, dtype=np.int64)
    if len(np.unique(y)) != 2:
        raise ValueError("training data must contain both live and spoof samples")

    clf = LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=50_000,
        dual="auto",
        random_state=seed,
    )
    clf.fit(x, y)
    if clf.coef_.shape[0] != 1:
        raise RuntimeError("expected a binary classifier")
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])
    return np.where(y == LIVE, 1, -1), w, b


def quantise(w: np.ndarray, b: float, threshold_float: float = 0.0) -> Q14Model:
    """Scale (w, b, threshold) so |w| < 2, then round to Q2.14."""
    w_q, scale = fp.quantize_weights(w, max_abs=1.9)
    b_q = int(fp.from_float(b * scale))
    thr_q = int(fp.from_float(threshold_float * scale))
    return Q14Model(weights=w_q, bias=b_q, threshold=thr_q, scale=scale)


def pick_threshold(scores, labels) -> int:
    """Youden's J (maximise TPR - FPR) on the integer score scale."""
    s = np.asarray(scores, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    candidates = np.unique(s)
    if candidates.size == 0:
        return 0
    best, best_j = int(candidates[0]), -np.inf
    for t in candidates:
        pred = s >= t
        tpr = np.mean(pred[y == LIVE]) if np.any(y == LIVE) else 0.0
        fpr = np.mean(pred[y == SPOOF]) if np.any(y == SPOOF) else 0.0
        j = tpr - fpr
        if j > best_j:
            best, best_j = int(t), j
    return best


def evaluate(model: Q14Model, features, labels, scores: Optional[List[int]] = None):
    """Integer-path metrics; also returns per-sample scores for plotting."""
    y = np.asarray(labels, dtype=np.int64)
    if scores is None:
        scores = [model.score(f) for f in features]
    s = np.asarray(scores, dtype=np.int64)
    pred = (s >= model.threshold).astype(np.int64)

    tp = int(np.sum((pred == LIVE) & (y == LIVE)))
    tn = int(np.sum((pred == SPOOF) & (y == SPOOF)))
    fp_ = int(np.sum((pred == LIVE) & (y == SPOOF)))
    fn = int(np.sum((pred == SPOOF) & (y == LIVE)))
    far = fp_ / max(fp_ + tn, 1)      # APCER when spoof is the attack class
    frr = fn / max(fn + tp, 1)        # BPCER
    acc = (tp + tn) / max(len(y), 1)
    return {
        "n": int(len(y)),
        "accuracy": float(acc),
        "apcer": float(far),
        "bpcer": float(frr),
        "acer": float((far + frr) / 2),
        "tp": tp, "tn": tn, "fp": fp_, "fn": fn,
    }
