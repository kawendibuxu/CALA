"""Read-only 02 calibration diagnostic for a 7-d decision head.

Does not write official calibration or accept-only directories.
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
    parser = argparse.ArgumentParser(description="Diagnose whether a head admits the official 0.99 02 threshold.")
    parser.add_argument("--head_checkpoint", required=True)
    parser.add_argument("--calibration_dir", required=True)
    parser.add_argument("--full_pairs", required=True)
    parser.add_argument("--target_precision", type=float, default=0.99)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
    features = checkpoint["feature_names"]
    standardizer = Standardizer(**checkpoint["standardizer"])
    model = LogisticDecisionHead(len(features))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    rows = read_calibration_source(None, args.calibration_dir)
    full_rows = read_csv(args.full_pairs)
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
    raw_scores = 1 / (1 + np.exp(-raw_logits))
    clipped_logits = np.log(np.clip(raw_scores, 1e-6, 1 - 1e-6) / np.clip(1 - raw_scores, 1e-6, 1))
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(clipped_logits[trainable, None], labels[trainable], sample_weight=weights[trainable])
    probabilities = calibrator.predict_proba(clipped_logits[:, None])[:, 1]

    positive = labels
    track_gate = weighted_quantile(values[positive, features.index("mean_track_confidence")], weights[positive], 0.01)
    visibility_gate = weighted_quantile(values[positive, features.index("mean_visibility")], weights[positive], 0.01)
    residual_gate = weighted_quantile(values[positive, features.index("median_3d_residual")], weights[positive], 0.99)
    strong = (
        (values[:, features.index("mean_track_confidence")] >= track_gate)
        & (values[:, features.index("mean_visibility")] >= visibility_gate)
        & (values[:, features.index("median_3d_residual")] <= residual_gate)
    )
    precision, recall, thresholds = precision_recall_curve(
        labels[trainable], probabilities[trainable], sample_weight=weights[trainable]
    )
    rows_out = []
    for threshold in thresholds:
        accepted = trainable & strong & (probabilities >= threshold)
        weight_accepted = float(np.sum(weights[accepted]))
        weighted_precision = float(np.sum(weights[accepted & labels]) / max(weight_accepted, 1e-12))
        weighted_recall = float(np.sum(weights[accepted & labels]) / np.sum(weights[labels]))
        rows_out.append(
            {
                "threshold": float(threshold),
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "accepted": int(accepted.sum()),
                "accepted_positive": int((accepted & labels).sum()),
                "accepted_negative": int((accepted & trainable & ~labels).sum()),
            }
        )
    feasible = [row for row in rows_out if row["weighted_precision"] >= args.target_precision]
    best = max(rows_out, key=lambda row: (row["weighted_precision"], row["weighted_recall"]))
    report = {
        "head_checkpoint": str(Path(args.head_checkpoint).resolve()),
        "target_precision": args.target_precision,
        "weighted_roc_auc": float(roc_auc_score(labels[trainable], probabilities[trainable], sample_weight=weights[trainable])),
        "weighted_average_precision": float(
            average_precision_score(labels[trainable], probabilities[trainable], sample_weight=weights[trainable])
        ),
        "platt_coef": float(calibrator.coef_[0, 0]),
        "platt_intercept": float(calibrator.intercept_[0]),
        "geometry_gates": {
            "track_confidence_accept_min": track_gate,
            "visibility_accept_min": visibility_gate,
            "median_3d_residual_accept_max": residual_gate,
        },
        "strong_trainable": int((trainable & strong).sum()),
        "strong_trainable_positive": int((trainable & strong & labels).sum()),
        "strong_trainable_negative": int((trainable & strong & ~labels).sum()),
        "max_weighted_precision_under_strong_gate": best,
        "n_feasible_thresholds": len(feasible),
        "selected_if_feasible": max(feasible, key=lambda row: (row["weighted_recall"], -row["threshold"])) if feasible else None,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
