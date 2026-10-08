"""Fit 02 Platt for a new 7-d head, then attach the official frozen rules.

Unique variable: 00 training seed (head weights and the induced 02 Platt).
Held fixed: official accept probability 0.947... and official geometry gates.
Does not rerun threshold selection and does not write official directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from calibrate_kitti_three_way_decision import read_calibration_source, read_csv
from train_kitti_decision_head import LogisticDecisionHead, Standardizer


OFFICIAL_FROZEN = Path(
    "/data1/jiaming/UniPR-3D-main/outputs/kitti_cross_token_3d_full_three_way_calibration_02/frozen_three_way_decision.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze a new seed head under official 02 rules.")
    parser.add_argument("--head_checkpoint", required=True)
    parser.add_argument("--calibration_dir", required=True)
    parser.add_argument("--full_pairs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--official_frozen", default=str(OFFICIAL_FROZEN))
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    official = torch.load(args.official_frozen, map_location="cpu", weights_only=False)
    official_rules = {key: float(value) for key, value in official["rules"].items()}
    checkpoint = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
    features = checkpoint["feature_names"]
    expected = [
        "descriptor_similarity",
        "weighted_3d_inlier_ratio",
        "median_3d_residual",
        "p90_3d_residual",
        "mean_track_confidence",
        "mean_visibility",
        "temporal_consistency",
    ]
    if features != expected:
        raise RuntimeError(f"Expected official 7-d features, got {features}.")
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
    raw_scores = 1 / (1 + np.exp(-raw_logits))
    clipped_logits = np.log(np.clip(raw_scores, 1e-6, 1 - 1e-6) / np.clip(1 - raw_scores, 1e-6, 1))
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(clipped_logits[trainable, None], labels[trainable], sample_weight=weights[trainable])
    probabilities = calibrator.predict_proba(clipped_logits[:, None])[:, 1]

    accept = trainable & (probabilities >= official_rules["accept_probability"])
    accept = accept & (values[:, features.index("mean_track_confidence")] >= official_rules["track_confidence_accept_min"])
    accept = accept & (values[:, features.index("mean_visibility")] >= official_rules["visibility_accept_min"])
    accept = accept & (values[:, features.index("median_3d_residual")] <= official_rules["median_3d_residual_accept_max"])
    report = {
        "training_seed": args.seed,
        "head_checkpoint": str(Path(args.head_checkpoint).resolve()),
        "official_frozen": str(Path(args.official_frozen).resolve()),
        "calibration_source": str(Path(args.calibration_dir).resolve()),
        "full_pairs": str(Path(args.full_pairs).resolve()),
        "features": features,
        "calibration": {
            "type": "weighted_platt",
            "coef": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
        },
        "rules": official_rules,
        "unique_variable": "00 logistic training seed; 02 Platt is refit for the new head",
        "held_fixed": [
            "official accept_probability",
            "official reject_probability",
            "official geometry hard gates",
        ],
        "weighted_metrics": {
            "roc_auc": float(roc_auc_score(labels[trainable], probabilities[trainable], sample_weight=weights[trainable])),
            "average_precision": float(
                average_precision_score(labels[trainable], probabilities[trainable], sample_weight=weights[trainable])
            ),
            "brier": float(brier_score_loss(labels[trainable], probabilities[trainable], sample_weight=weights[trainable])),
            "official_rule_accept_count": int(accept.sum()),
            "official_rule_accept_positive": int((accept & labels).sum()),
            "official_rule_accept_negative": int((accept & ~labels).sum()),
        },
        "note": (
            "Thresholds and geometry gates are copied from the official seed-17 freeze. "
            "They are not reselected. Official calibration files are not modified."
        ),
    }
    output_dir.mkdir(parents=True)
    torch.save(
        {
            "head_checkpoint": checkpoint,
            "platt_coef": calibrator.coef_,
            "platt_intercept": calibrator.intercept_,
            "rules": official_rules,
        },
        output_dir / "frozen_three_way_decision.pt",
    )
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
