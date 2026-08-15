"""Train a standalone loop-closure decision head from retrieved KITTI pairs.

This script intentionally consumes only exported descriptor and VGGT geometry
features.  It does not import, fine-tune, or otherwise modify UniPR.
"""

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_FEATURES = [
    "inlier_ratio",
    "mean_sampson_error",
    "mean_track_confidence",
    "mean_visibility",
    "descriptor_similarity",
]


@dataclass
class Standardizer:
    feature_names: list[str]
    medians: list[float]
    means: list[float]
    stds: list[float]

    @classmethod
    def fit(cls, values, feature_names):
        medians = np.nanmedian(values, axis=0)
        filled = np.where(np.isfinite(values), values, medians)
        means = filled.mean(axis=0)
        stds = filled.std(axis=0)
        stds = np.maximum(stds, 1e-6)
        return cls(feature_names, medians.tolist(), means.tolist(), stds.tolist())

    def transform(self, values):
        medians = np.asarray(self.medians, dtype=np.float32)
        means = np.asarray(self.means, dtype=np.float32)
        stds = np.asarray(self.stds, dtype=np.float32)
        values = np.where(np.isfinite(values), values, medians)
        return ((values - means) / stds).astype(np.float32)


class LogisticDecisionHead(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, features):
        return self.linear(features).squeeze(-1)


class MLPDecisionHead(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a standalone geometry-aware KITTI loop decision head."
    )
    parser.add_argument("--train_csv", required=True, help="Sampled geometry CSV, e.g. sequence 00.")
    parser.add_argument("--val_csv", required=True, help="Disjoint sampled geometry CSV, e.g. sequence 02.")
    parser.add_argument("--output_dir", default="outputs/kitti_decision_head_00_to_02")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--head", choices=("logistic", "mlp"), default="logistic")
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--target_precision", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_labeled_rows(path, feature_names):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"label", "query_dataset_index", "candidate_dataset_index", *feature_names}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    kept = [row for row in rows if row["label"] in {"positive", "negative"}]
    if not kept:
        raise ValueError(f"{path} has no positive/negative rows.")
    values = np.asarray(
        [[float(row[name]) for name in feature_names] for row in kept], dtype=np.float32
    )
    labels = np.asarray([row["label"] == "positive" for row in kept], dtype=np.float32)
    return kept, values, labels


def precision_recall_at_threshold(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    tp = int(np.logical_and(predicted, labels == 1).sum())
    fp = int(np.logical_and(predicted, labels == 0).sum())
    fn = int(np.logical_and(~predicted, labels == 1).sum())
    tn = int(np.logical_and(~predicted, labels == 0).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def choose_high_precision_threshold(labels, probabilities, target_precision):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    candidates = []
    for index, threshold in enumerate(thresholds):
        if precision[index] >= target_precision:
            candidates.append((recall[index], -threshold, threshold))
    if candidates:
        # Maximize recall while retaining the requested precision; break ties by
        # preferring a lower score threshold.
        return max(candidates)[2], True

    # A target can be impossible on a held-out sequence.  Use the best observed
    # precision so the report remains explicit instead of silently choosing 0.5.
    best_index = int(np.argmax(precision[:-1]))
    return float(thresholds[best_index]), False


def query_metrics(rows, labels, probabilities, threshold):
    # These are intentionally labelled "sampled": a query may have unextracted
    # top-K candidates, so this is not the final end-to-end loop metric yet.
    by_query = {}
    for row, label, probability in zip(rows, labels, probabilities):
        by_query.setdefault(row["query_dataset_index"], []).append((int(label), float(probability)))

    positive_queries = 0
    detected_queries = 0
    top1_correct = 0
    false_positives = 0
    selected_pairs = 0
    for candidates in by_query.values():
        has_positive = any(label for label, _ in candidates)
        selected = [(label, score) for label, score in candidates if score >= threshold]
        if has_positive:
            positive_queries += 1
            detected_queries += int(any(label for label, _ in selected))
            top1_label = max(candidates, key=lambda item: item[1])[0]
            top1_correct += top1_label
        false_positives += sum(1 for label, _ in selected if not label)
        selected_pairs += len(selected)

    num_queries = len(by_query)
    return {
        "scope": "sampled_pairs_only_not_full_top_k",
        "num_queries": num_queries,
        "positive_queries": positive_queries,
        "selected_pairs": selected_pairs,
        "query_recall_at_threshold": detected_queries / positive_queries if positive_queries else 0.0,
        "observed_reranked_recall_at_1": top1_correct / positive_queries if positive_queries else 0.0,
        "false_positives_per_query": false_positives / num_queries if num_queries else 0.0,
    }


def probabilities(model, features, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(device)).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def main():
    args = parse_args()
    if not 0 < args.target_precision <= 1:
        raise ValueError("--target_precision must be in (0, 1].")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_rows, train_raw, train_labels = read_labeled_rows(args.train_csv, args.features)
    val_rows, val_raw, val_labels = read_labeled_rows(args.val_csv, args.features)
    standardizer = Standardizer.fit(train_raw, args.features)
    train_features = standardizer.transform(train_raw)
    val_features = standardizer.transform(val_raw)

    device = torch.device(args.device)
    if args.head == "logistic":
        model = LogisticDecisionHead(len(args.features))
    else:
        model = MLPDecisionHead(len(args.features), args.hidden_dim)
    model.to(device)

    num_positive = int(train_labels.sum())
    num_negative = len(train_labels) - num_positive
    pos_weight = torch.tensor([num_negative / max(num_positive, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device))
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()

    # Sequence 02 is reserved for the operating threshold, so it must not be
    # used for checkpoint selection or early stopping.
    train_probabilities = probabilities(model, train_features, device)
    val_probabilities = probabilities(model, val_features, device)
    threshold, reached_target = choose_high_precision_threshold(
        val_labels, val_probabilities, args.target_precision
    )

    train_metrics = {
        "roc_auc": float(roc_auc_score(train_labels, train_probabilities)),
        "average_precision": float(average_precision_score(train_labels, train_probabilities)),
        **precision_recall_at_threshold(train_labels, train_probabilities, threshold),
    }
    val_metrics = {
        "roc_auc": float(roc_auc_score(val_labels, val_probabilities)),
        "average_precision": float(average_precision_score(val_labels, val_probabilities)),
        **precision_recall_at_threshold(val_labels, val_probabilities, threshold),
        "sampled_query_metrics": query_metrics(val_rows, val_labels, val_probabilities, threshold),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "head": args.head,
        "hidden_dim": args.hidden_dim,
        "feature_names": args.features,
        "standardizer": asdict(standardizer),
        "state_dict": model.cpu().state_dict(),
        "threshold": float(threshold),
        "threshold_target_precision": args.target_precision,
        "probability_note": (
            "Scores are trained on a 1:5 sampled positive/negative dataset with weighted BCE. "
            "They rank and threshold candidates, but are not calibrated deployment probabilities "
            "until geometry features are extracted using a representative top-K sampling policy."
        ),
    }
    torch.save(checkpoint, output_dir / "decision_head.pt")

    with (output_dir / "val_predictions.csv").open("w", newline="") as handle:
        fieldnames = list(val_rows[0]) + ["target", "decision_score", "accepted_by_head"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, label, probability in zip(val_rows, val_labels, val_probabilities):
            writer.writerow(
                {
                    **row,
                    "target": int(label),
                    "decision_score": f"{probability:.8f}",
                    "accepted_by_head": int(probability >= threshold),
                }
            )

    report = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "val_csv": str(Path(args.val_csv).resolve()),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_class_counts": {"positive": num_positive, "negative": num_negative},
        "val_class_counts": {"positive": int(val_labels.sum()), "negative": int(len(val_labels) - val_labels.sum())},
        "features": args.features,
        "head": args.head,
        "epochs_completed": epoch,
        "weighted_bce_pos_weight": float(pos_weight.item()),
        "threshold_target_reached": reached_target,
        "train_pair_metrics": train_metrics,
        "val_pair_metrics": val_metrics,
        "limitations": [
            "Validation contains a sampled 100 positive / 500 negative subset, not every top-K candidate.",
            "Observed reranked Recall@1 and false positives per query are sampled-pair metrics, not full end-to-end metrics.",
            "Descriptor Recall@K must be read from build_kitti_topk_pairs.py output; a decision head cannot recover positives absent from top-K.",
            "The threshold is selected on this validation sequence. A third, unseen sequence is required for an unbiased thresholded test result.",
        ],
    }
    with (output_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved decision head to {output_dir / 'decision_head.pt'}")
    print(f"Saved validation predictions to {output_dir / 'val_predictions.csv'}")
    print(f"Saved report to {output_dir / 'report.json'}")
    print(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"Validation AP: {val_metrics['average_precision']:.4f}")
    print(
        "Threshold: "
        f"{threshold:.6f} (target precision {args.target_precision:.3f}, reached={reached_target})"
    )
    print(
        "Validation pair metrics: "
        f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} "
        f"f1={val_metrics['f1']:.4f} TP={val_metrics['tp']} FP={val_metrics['fp']}"
    )
    query = val_metrics["sampled_query_metrics"]
    print(
        "Validation sampled-query metrics: "
        f"recall={query['query_recall_at_threshold']:.4f} "
        f"observed_reranked_R@1={query['observed_reranked_recall_at_1']:.4f} "
        f"FP/query={query['false_positives_per_query']:.4f}"
    )


if __name__ == "__main__":
    main()
