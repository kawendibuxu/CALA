"""Calibrate a frozen decision head and freeze accept/uncertain/reject rules."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

from train_kitti_decision_head import LogisticDecisionHead, MLPDecisionHead, Standardizer


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate frozen KITTI loop scores on a representative validation subset.")
    parser.add_argument("--head_checkpoint", required=True)
    calibration_source = parser.add_mutually_exclusive_group(required=True)
    calibration_source.add_argument("--calibration_csv", help="Sampled or complete feature CSV.")
    calibration_source.add_argument("--calibration_dir", help="Directory containing complete feature chunks.")
    parser.add_argument("--full_pairs", required=True, help="Full sequence-02 top-K CSV used only for population weights.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_precision", type=float, default=0.99)
    parser.add_argument("--reject_probability", type=float, default=0.05)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_calibration_source(csv_path, directory):
    if csv_path:
        return read_csv(csv_path)
    paths = sorted((Path(directory) / "chunks").glob("chunk_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No calibration chunks found in {directory}")
    rows = []
    for path in paths:
        rows.extend(read_csv(path))
    return rows


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(quantile, cumulative, values))


def metrics(labels, predicted):
    tp = float(np.sum(predicted & labels))
    fp = float(np.sum(predicted & ~labels))
    fn = float(np.sum(~predicted & labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": precision, "recall": recall}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    checkpoint = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
    features = checkpoint["feature_names"]
    standardizer = Standardizer(**checkpoint["standardizer"])
    model = LogisticDecisionHead(len(features)) if checkpoint["head"] == "logistic" else MLPDecisionHead(len(features), checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    rows = read_calibration_source(args.calibration_csv, args.calibration_dir)
    full_rows = read_csv(args.full_pairs)
    if args.calibration_dir:
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
    weights = np.asarray([population[(int(row["candidate_rank"]), row["label"])] / sample[(int(row["candidate_rank"]), row["label"])] for row in rows], dtype=np.float64)
    values = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float32)
    labels = np.asarray([row["label"] == "positive" for row in rows], dtype=bool)
    trainable = np.isin([row["label"] for row in rows], ["positive", "negative"])
    standard_values = standardizer.transform(values)
    with torch.no_grad():
        raw_logits = model(torch.from_numpy(standard_values)).numpy()
    raw_scores = 1 / (1 + np.exp(-raw_logits))
    clipped_logits = np.log(np.clip(raw_scores, 1e-6, 1 - 1e-6) / np.clip(1 - raw_scores, 1e-6, 1))
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(clipped_logits[trainable, None], labels[trainable], sample_weight=weights[trainable])
    probabilities = calibrator.predict_proba(clipped_logits[:, None])[:, 1]

    # Strong-evidence gates are intentionally set from weighted positive tails.
    # They convert high-score, low-confidence conflicts into uncertain candidates.
    positive = labels
    track_gate = weighted_quantile(values[positive, features.index("mean_track_confidence")], weights[positive], 0.01)
    visibility_gate = weighted_quantile(values[positive, features.index("mean_visibility")], weights[positive], 0.01)
    residual_gate = weighted_quantile(values[positive, features.index("median_3d_residual")], weights[positive], 0.99)
    negative = ~labels & trainable
    reject_track = weighted_quantile(values[negative, features.index("mean_track_confidence")], weights[negative], 0.99)
    reject_visibility = weighted_quantile(values[negative, features.index("mean_visibility")], weights[negative], 0.99)
    strong = (
        (values[:, features.index("mean_track_confidence")] >= track_gate)
        & (values[:, features.index("mean_visibility")] >= visibility_gate)
        & (values[:, features.index("median_3d_residual")] <= residual_gate)
    )
    precision, recall, thresholds = precision_recall_curve(labels[trainable], probabilities[trainable], sample_weight=weights[trainable])
    candidates = []
    for threshold in thresholds:
        accepted = trainable & strong & (probabilities >= threshold)
        weighted_precision = np.sum(weights[accepted & labels]) / max(np.sum(weights[accepted]), 1e-12)
        weighted_recall = np.sum(weights[accepted & labels]) / np.sum(weights[labels])
        if weighted_precision >= args.target_precision:
            candidates.append((weighted_recall, -threshold, threshold))
    if not candidates:
        raise RuntimeError("No accept threshold meets the requested weighted precision.")
    accept_threshold = max(candidates)[2]
    weak = (
        (values[:, features.index("mean_track_confidence")] <= reject_track)
        & (values[:, features.index("mean_visibility")] <= reject_visibility)
    )
    state = np.full(len(rows), "uncertain", dtype=object)
    state[trainable & strong & (probabilities >= accept_threshold)] = "accept"
    state[trainable & weak & (probabilities <= args.reject_probability)] = "reject"
    accepted = state == "accept"
    rejected = state == "reject"
    report = {
        "head_checkpoint": str(Path(args.head_checkpoint).resolve()),
        "calibration_source": str(Path(args.calibration_csv or args.calibration_dir).resolve()),
        "full_pairs": str(Path(args.full_pairs).resolve()),
        "features": features,
        "calibration": {"type": "weighted_platt", "coef": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0])},
        "rules": {
            "accept_probability": float(accept_threshold),
            "reject_probability": args.reject_probability,
            "track_confidence_accept_min": track_gate,
            "visibility_accept_min": visibility_gate,
            "median_3d_residual_accept_max": residual_gate,
            "track_confidence_reject_max": reject_track,
            "visibility_reject_max": reject_visibility,
        },
        "weighted_metrics": {
            "roc_auc": float(roc_auc_score(labels[trainable], probabilities[trainable], sample_weight=weights[trainable])),
            "average_precision": float(average_precision_score(labels[trainable], probabilities[trainable], sample_weight=weights[trainable])),
            "brier": float(brier_score_loss(labels[trainable], probabilities[trainable], sample_weight=weights[trainable])),
            "accept": metrics(labels[trainable], accepted[trainable]),
        },
        "state_counts_sample": {name: int((state == name).sum()) for name in ("accept", "uncertain", "reject")},
        "state_weighted_population": {name: float(weights[state == name].sum()) for name in ("accept", "uncertain", "reject")},
        "note": "Rules are selected only on sequence 02 and must remain frozen for sequence 05.",
    }
    output_dir.mkdir(parents=True)
    torch.save({"head_checkpoint": checkpoint, "platt_coef": calibrator.coef_, "platt_intercept": calibrator.intercept_, "rules": report["rules"]}, output_dir / "frozen_three_way_decision.pt")
    with (output_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "calibration_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) + ["raw_score", "calibrated_probability", "state", "population_weight"])
        writer.writeheader()
        for row, raw, probability, name, weight in zip(rows, raw_scores, probabilities, state, weights):
            writer.writerow({**row, "raw_score": f"{raw:.8f}", "calibrated_probability": f"{probability:.8f}", "state": name, "population_weight": f"{weight:.8f}"})
    print(f"Saved frozen three-way decision to {output_dir / 'frozen_three_way_decision.pt'}")
    print(json.dumps(report["weighted_metrics"], indent=2))
    print(json.dumps(report["rules"], indent=2))


if __name__ == "__main__":
    main()
