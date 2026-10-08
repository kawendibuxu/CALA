"""Calibrate the descriptor-only logistic on sequence 02 only.

Platt scaling and the accept probability are selected on 02 with the same
target_precision=0.99 protocol as the official head. Geometry hard gates are
copied from the official frozen 02 rules and are not retuned. 05/06/07 are
not read.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

from calibrate_kitti_three_way_decision import (
    metrics,
    read_calibration_source,
    read_csv,
)
from train_kitti_decision_head import LogisticDecisionHead, Standardizer

GEOMETRY_RULE_KEYS = (
    "track_confidence_accept_min",
    "visibility_accept_min",
    "median_3d_residual_accept_max",
    "track_confidence_reject_max",
    "visibility_reject_max",
    "reject_probability",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate descriptor-only logistic on KITTI 02; copy official geometry gates."
    )
    parser.add_argument("--head_checkpoint", required=True)
    parser.add_argument("--official_frozen_decision", required=True)
    parser.add_argument("--calibration_dir", required=True)
    parser.add_argument("--full_pairs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_precision", type=float, default=0.99)
    return parser.parse_args()


def geometry_arrays(rows):
    return {
        "mean_track_confidence": np.asarray(
            [float(row["mean_track_confidence"]) for row in rows], dtype=np.float64
        ),
        "mean_visibility": np.asarray([float(row["mean_visibility"]) for row in rows], dtype=np.float64),
        "median_3d_residual": np.asarray(
            [float(row["median_3d_residual"]) for row in rows], dtype=np.float64
        ),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    checkpoint = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
    features = checkpoint["feature_names"]
    if features != ["descriptor_similarity"]:
        raise RuntimeError(
            f"This calibration is for descriptor_similarity only, got {features}."
        )
    standardizer = Standardizer(**checkpoint["standardizer"])
    model = LogisticDecisionHead(len(features))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    official = torch.load(args.official_frozen_decision, map_location="cpu", weights_only=False)
    official_rules = official["rules"]
    geometry_rules = {key: float(official_rules[key]) for key in GEOMETRY_RULE_KEYS}

    rows = read_calibration_source(None, args.calibration_dir)
    full_rows = read_csv(args.full_pairs)
    indices = {int(row["pair_row_index"]) for row in rows}
    if len(rows) != len(full_rows) or indices != set(range(len(full_rows))):
        raise RuntimeError(
            f"Incomplete calibration features: found {len(indices)} / {len(full_rows)} pairs."
        )
    population = {}
    for row in full_rows:
        key = (int(row["candidate_rank"]), row["label"])
        population[key] = population.get(key, 0) + 1
    sample = {}
    for row in rows:
        key = (int(row["candidate_rank"]), row["label"])
        sample[key] = sample.get(key, 0) + 1
    weights = np.asarray(
        [
            population[(int(row["candidate_rank"]), row["label"])]
            / sample[(int(row["candidate_rank"]), row["label"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    values = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float32)
    labels = np.asarray([row["label"] == "positive" for row in rows], dtype=bool)
    trainable = np.isin([row["label"] for row in rows], ["positive", "negative"])
    standard_values = standardizer.transform(values)
    with torch.no_grad():
        raw_logits = model(torch.from_numpy(standard_values)).numpy()
    raw_scores = 1 / (1 + np.exp(-raw_logits))
    clipped_logits = np.log(
        np.clip(raw_scores, 1e-6, 1 - 1e-6) / np.clip(1 - raw_scores, 1e-6, 1)
    )
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(
        clipped_logits[trainable, None],
        labels[trainable],
        sample_weight=weights[trainable],
    )
    probabilities = calibrator.predict_proba(clipped_logits[:, None])[:, 1]

    geom = geometry_arrays(rows)
    strong = (
        (geom["mean_track_confidence"] >= geometry_rules["track_confidence_accept_min"])
        & (geom["mean_visibility"] >= geometry_rules["visibility_accept_min"])
        & (geom["median_3d_residual"] <= geometry_rules["median_3d_residual_accept_max"])
    )
    _, _, thresholds = precision_recall_curve(
        labels[trainable],
        probabilities[trainable],
        sample_weight=weights[trainable],
    )
    candidates = []
    for threshold in thresholds:
        accepted = trainable & strong & (probabilities >= threshold)
        weighted_precision = np.sum(weights[accepted & labels]) / max(np.sum(weights[accepted]), 1e-12)
        weighted_recall = np.sum(weights[accepted & labels]) / np.sum(weights[labels])
        if weighted_precision >= args.target_precision:
            candidates.append((weighted_recall, -float(threshold), float(threshold)))
    if not candidates:
        raise RuntimeError("No accept threshold meets the requested weighted precision.")
    accept_threshold = max(candidates)[2]
    weak = (
        (geom["mean_track_confidence"] <= geometry_rules["track_confidence_reject_max"])
        & (geom["mean_visibility"] <= geometry_rules["visibility_reject_max"])
    )
    state = np.full(len(rows), "uncertain", dtype=object)
    state[trainable & strong & (probabilities >= accept_threshold)] = "accept"
    state[trainable & weak & (probabilities <= geometry_rules["reject_probability"])] = "reject"
    accepted = state == "accept"
    rules = {
        "accept_probability": float(accept_threshold),
        **geometry_rules,
    }
    report = {
        "head_checkpoint": str(Path(args.head_checkpoint).resolve()),
        "official_frozen_decision": str(Path(args.official_frozen_decision).resolve()),
        "calibration_source": str(Path(args.calibration_dir).resolve()),
        "full_pairs": str(Path(args.full_pairs).resolve()),
        "features": features,
        "calibration": {
            "type": "weighted_platt",
            "coef": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
        },
        "rules": rules,
        "geometry_gates": "copied from official frozen 02 rules; not retuned",
        "unique_variable": (
            "logistic input is descriptor_similarity only; Platt and accept_probability "
            "are recalibrated on 02; geometry hard gates stay at the official frozen values"
        ),
        "weighted_metrics": {
            "roc_auc": float(
                roc_auc_score(
                    labels[trainable],
                    probabilities[trainable],
                    sample_weight=weights[trainable],
                )
            ),
            "average_precision": float(
                average_precision_score(
                    labels[trainable],
                    probabilities[trainable],
                    sample_weight=weights[trainable],
                )
            ),
            "brier": float(
                brier_score_loss(
                    labels[trainable],
                    probabilities[trainable],
                    sample_weight=weights[trainable],
                )
            ),
            "accept": metrics(labels[trainable], accepted[trainable]),
        },
        "state_counts_sample": {name: int((state == name).sum()) for name in ("accept", "uncertain", "reject")},
        "note": "Rules are selected only on sequence 02. Geometry gates are copied, not re-estimated.",
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
    with (output_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "calibration_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0])
            + ["raw_score", "calibrated_probability", "state", "population_weight"],
        )
        writer.writeheader()
        for row, raw, probability, name, weight in zip(
            rows, raw_scores, probabilities, state, weights
        ):
            writer.writerow(
                {
                    **row,
                    "raw_score": f"{raw:.8f}",
                    "calibrated_probability": f"{probability:.8f}",
                    "state": name,
                    "population_weight": f"{weight:.8f}",
                }
            )
    print(f"Saved frozen descriptor-only decision to {output_dir / 'frozen_three_way_decision.pt'}")
    print(json.dumps(report["weighted_metrics"], indent=2))
    print(json.dumps(report["rules"], indent=2))


if __name__ == "__main__":
    main()
