"""Train the Q2.14 linear classifier and save the deployable model.

    python scripts/train.py --data data/synthetic.npz

Outputs
    artifacts/model.json    weights, bias, threshold (integers only)
    artifacts/scores.csv    per-frame score / decision for every split
    artifacts/metrics.json  float vs quantised metrics, threshold choice
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import fixed_point as fp  # noqa: E402
from model.classifier import (  # noqa: E402
    LIVE, Q14Model, evaluate, pick_threshold, quantise, train_svm,
)
from model.pipeline import LivenessEngine  # noqa: E402


def featurise(x: np.ndarray, engine: LivenessEngine) -> np.ndarray:
    out = np.zeros((x.shape[0], 512), dtype=np.int64)
    for i in range(x.shape[0]):
        out[i] = engine.features(x[i, 0], x[i, 1], method="fast")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data" / "synthetic.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--C", type=float, default=0.2, help="SVM regularisation (lower = smoother)")
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"missing {data_path}\nrun: python scripts/make_synthetic.py", file=sys.stderr)
        return 2
    with np.load(data_path, allow_pickle=False) as z:
        ds = {k: z[k] for k in z.files}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # a zero-weight placeholder lets us featurise before the model exists
    engine = LivenessEngine(Q14Model(weights=np.zeros(512, dtype=np.int64)))
    feats = featurise(ds["X"], engine)
    y = ds["y"].astype(np.int64)
    split = ds["split"]

    tr, dv, te = [(split == s) for s in ("train", "dev", "test")]
    if not tr.any():
        print("no training frames in this dataset", file=sys.stderr)
        return 2

    _y_signed, w_float, b_float = train_svm(feats[tr], y[tr], seed=args.seed, C=args.C)

    model = quantise(w_float, b_float)

    # threshold on dev (never test); fall back to train if dev is empty
    tune = dv if dv.sum() >= 4 else tr
    tune_scores = [model.score(f) for f in feats[tune]]
    model.threshold = pick_threshold(tune_scores, y[tune])

    model.meta = {
        "trained_on": str(data_path),
        "n_train": int(tr.sum()),
        "seed": args.seed,
        "weight_scale": round(float(model.scale), 6),
        "max_abs_weight_q14": int(np.max(np.abs(model.weights))),
        "features": "Q2.14 histogram counts, ch0 then ch1",
    }
    model.save(out_dir / "model.json")

    # ---- metrics -------------------------------------------------------
    float_scores = feats @ w_float + b_float
    report = {}
    for name, mask in (("train", tr), ("dev", dv), ("test", te)):
        if not mask.any():
            continue
        int_scores = [model.score(f) for f in feats[mask]]
        m = evaluate(model, feats[mask], y[mask], scores=int_scores)
        try:
            m["auc_float"] = float(roc_auc_score(y[mask], float_scores[mask]))
            m["auc_q14"] = float(roc_auc_score(y[mask], np.asarray(int_scores)))
        except ValueError:
            m["auc_float"] = m["auc_q14"] = float("nan")
        report[name] = m

    # ---- per-frame scores ----------------------------------------------
    with (out_dir / "scores.csv").open("w", encoding="utf-8") as fh:
        fh.write("row,split,subject,label,attack,score_q14,score_float,decision\n")
        for i in range(len(y)):
            s = model.score(feats[i])
            fh.write(f"{i},{split[i]},{int(ds['subject'][i])},{int(y[i])},"
                     f"{str(ds['attack'][i])},{s},{float_scores[i]:.6f},"
                     f"{int(s >= model.threshold)}\n")

    metrics = {
        "model": {
            "weights_dtype": "int16 Q2.14",
            "acc_bits": fp.ACC_BITS,
            "threshold_q14": int(model.threshold),
            "threshold_float_equiv": round(model.threshold / fp.ONE, 6),
            "bias_q14": int(model.bias),
            "clipped_weights": int(np.sum(np.abs(w_float * model.scale) * fp.ONE > fp.QMAX)),
        },
        "splits": report,
        "latency_model": {
            "cycles_per_frame": engine.cycles(),
            "us_at_150MHz": round(engine.latency_us(150.0), 3),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"model    : {out_dir / 'model.json'}")
    print(f"threshold: {model.threshold} ({model.threshold / fp.ONE:+.4f} in real units)")
    print(f"weights  : |w|max = {int(np.max(np.abs(model.weights)))} Q2.14 "
          f"(scale {model.scale:.4f})")
    print()
    print(f"{'split':<6}{'n':>5}{'AUC float':>11}{'AUC Q2.14':>11}"
          f"{'APCER':>8}{'BPCER':>8}{'ACER':>8}")
    for name, m in report.items():
        print(f"{name:<6}{m['n']:>5}{m['auc_float']:>11.4f}{m['auc_q14']:>11.4f}"
              f"{m['apcer']:>8.3f}{m['bpcer']:>8.3f}{m['acer']:>8.3f}")
    print("\nAUC Q2.14 is the number the hardware produces; report that one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
