"""Select a 02 accept threshold with the same probability formula used at eval.

Official calibrate fits Platt with sklearn.predict_proba, then eval reapplies
Platt through a 1e-6 clip. For some seeds the selected 0.99 threshold sits
above that clip ceiling, so test-time accept count is identically zero.

This script keeps the official 00 head recipe, official geometry gates and the
0.99 / max-precision selection rule, but scores 02 with the eval clip formula.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from calibrate_kitti_three_way_decision import read_calibration_source, read_csv, weighted_quantile
from train_kitti_decision_head import LogisticDecisionHead, Standardizer


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze a seed head with clip-consistent 02 threshold selection.")
    parser.add_argument("--head_checkpoint", required=True)
    parser.add_argument("--calibration_dir", required=True)
    parser.add_argument("--full_pairs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target_precision", type=float, default=0.99)
    parser.add_argument("--reject_probability", type=float, default=0.05)
    return parser.parse_args()


def clip_calibrated_probability(raw_logits, coef, intercept):
    raw_scores = 1.0 / (1.0 + np.exp(-raw_logits))
    clipped = np.log(np.clip(raw_scores, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - raw_scores, 1e-6, 1.0))
    return 1.0 / (1.0 + np.exp(-(coef * clipped + intercept)))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    checkpoint = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
    features = checkpoint["feature_names"]
    standardizer = Standardizer(**checkpoint["standardizer"])
    model = LogisticDecisionHead(len(features))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    rows = read_calibration_source(None, args.calibration_dir)
    full_rows = read_csv(args.full_pairs)
    indices = {int(row["pair_row_index"]) for row in rows}
    if len(rows) != len(full_rows) or indices != set(range(len(full_rows))):
        raise RuntimeError(f"Incomplete calibration features: found {len(indices)} / {len(full_rows)} pairs.")
    population = {}
    for row in full_rows:
        key = (int(row["candidate_rank"]), row["label"])
        population[key] = population.get(key, 0) + 1
    sample = {}
    for row in rows:
        key = (int(row["candidate_rank"]), row["label"])
        sample[key] = sample.get(key, 0) + 1
    weights = np.asarray(
        [population[(int(row["candidate_rank"]), row["label"])] / sample[(int(row["candidate_rank"]), row["label"])] for row in rows],
        dtype=np.float64,
    )
    values = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float32)
    labels = np.asarray([row["label"] == "positive" for row in rows], dtype=bool)
    trainable = np.isin([row["label"] for row in rows], ["positive", "negative"])
    with torch.no_grad():
        raw_logits = model(torch.from_numpy(standardizer.transform(values))).numpy()
    raw_scores = 1.0 / (1.0 + np.exp(-raw_logits))
    clipped_logits = np.log(np.clip(raw_scores, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - raw_scores, 1e-6, 1.0))
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(clipped_logits[trainable, None], labels[trainable], sample_weight=weights[trainable])
    coef = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    probabilities = clip_calibrated_probability(raw_logits, coef, intercept)

    positive = labels
    track_gate = weighted_quantile(values[positive, features.index("mean_track_confidence")], weights[positive], 0.01)
    visibility_gate = weighted_quantile(values[positive, features.index("mean_visibility")], weights[positive], 0.01)
    residual_gate = weighted_quantile(values[positive, features.index("median_3d_residual")], weights[positive], 0.99)
    strong = (
        (values[:, features.index("mean_track_confidence")] >= track_gate)
        & (values[:, features.index("mean_visibility")] >= visibility_gate)
        & (values[:, features.index("median_3d_residual")] <= residual_gate)
    )
    _, _, thresholds = precision_recall_curve(
        labels[trainable], probabilities[trainable], sample_weight=weights[trainable]
    )
    candidates = []
    all_points = []
    for threshold in thresholds:
        accepted = trainable & strong & (probabilities >= threshold)
        weight_accepted = float(np.sum(weights[accepted]))
        weighted_precision = float(np.sum(weights[accepted & labels]) / max(weight_accepted, 1e-12))
        weighted_recall = float(np.sum(weights[accepted & labels]) / np.sum(weights[labels]))
        point = {
            "threshold": float(threshold),
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "accepted": int(accepted.sum()),
            "accepted_positive": int((accepted & labels).sum()),
            "accepted_negative": int((accepted & trainable & ~labels).sum()),
        }
        all_points.append(point)
        if weighted_precision >= args.target_precision:
            candidates.append((weighted_recall, -float(threshold), float(threshold), point))
    if candidates:
        selection = max(candidates)
        accept_threshold = selection[2]
        selection_rule = "max_recall_at_weighted_precision_0.99"
        selected_point = selection[3]
    else:
        selected_point = max(
            all_points,
            key=lambda row: (row["weighted_precision"], row["weighted_recall"], -row["threshold"]),
        )
        accept_threshold = selected_point["threshold"]
        selection_rule = "fallback_max_weighted_precision_because_0.99_empty"
    rules = {
        "accept_probability": float(accept_threshold),
        "reject_probability": args.reject_probability,
        "track_confidence_accept_min": track_gate,
        "visibility_accept_min": visibility_gate,
        "median_3d_residual_accept_max": residual_gate,
    }
    report = {
        "training_seed": args.seed,
        "head_checkpoint": str(Path(args.head_checkpoint).resolve()),
        "calibration_source": str(Path(args.calibration_dir).resolve()),
        "features": features,
        "calibration": {"type": "weighted_platt_clip_consistent", "coef": coef, "intercept": intercept},
        "rules": rules,
        "selection_rule": selection_rule,
        "selected_02_point": selected_point,
        "n_feasible_0.99_thresholds": len(candidates),
        "clip_probability_max": float(probabilities.max()),
        "weighted_roc_auc": float(roc_auc_score(labels[trainable], probabilities[trainable], sample_weight=weights[trainable])),
        "weighted_average_precision": float(
            average_precision_score(labels[trainable], probabilities[trainable], sample_weight=weights[trainable])
        ),
        "unique_variable": "00 logistic training seed",
        "held_fixed": [
            "official 7-d features and 00 training recipe except seed",
            "02-only Platt and threshold selection",
            "eval-consistent 1e-6 clip probability",
            "official geometry-gate quantiles on 02 positives",
        ],
        "note": (
            "Threshold is selected only on sequence 02. Official seed-17 freeze and "
            "accept-only reports are not modified."
        ),
    }
    output_dir.mkdir(parents=True)
    torch.save(
        {
            "head_checkpoint": checkpoint,
            "platt_coef": calibrator.coef_,
            "platt_intercept": calibrator.intercept_,
            "rules": rules,
        },
        output_dir / "frozen_three_way_decision.pt",
    )
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
