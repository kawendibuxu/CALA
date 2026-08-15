"""Train the formal decision head from complete sequence-00 feature chunks.

All positives are retained.  Every epoch has a fixed positive:negative batch
ratio; negatives are sampled from rank-1--10 hard negatives plus a rank-
stratified pool from ranks 11--50.  Sequence 02 is never read here.
"""

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_kitti_decision_head import LogisticDecisionHead, MLPDecisionHead, Standardizer


DEFAULT_FEATURES = [
    "descriptor_similarity", "weighted_3d_inlier_ratio", "median_3d_residual",
    "p90_3d_residual", "mean_track_confidence", "mean_visibility", "temporal_consistency",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a full-data KITTI cross-token 3D decision head.")
    parser.add_argument("--feature_dir", required=True, help="Directory containing chunks/chunk_*.csv from sequence 00.")
    parser.add_argument("--pairs", required=True, help="Original complete sequence-00 top-K CSV used to verify chunk completeness.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--head", choices=("logistic", "mlp"), default="logistic")
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--positive_per_batch", type=int, default=16)
    parser.add_argument("--regular_negative_per_rank", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_rows(feature_dir, pairs_path):
    paths = sorted((Path(feature_dir) / "chunks").glob("chunk_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No chunks found in {feature_dir}")
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    with Path(pairs_path).open(newline="") as handle:
        expected = sum(1 for _ in csv.DictReader(handle))
    indices = {int(row["pair_row_index"]) for row in rows}
    if len(rows) != expected or indices != set(range(expected)):
        raise RuntimeError(f"Incomplete features: found {len(indices)} / {expected} top-K pairs.")
    return rows, paths


def build_negative_pool(rows, regular_per_rank, rng):
    hard = [row for row in rows if row["label"] == "negative" and int(row["candidate_rank"]) <= 10]
    regular = []
    for rank in range(11, 51):
        candidates = [row for row in rows if row["label"] == "negative" and int(row["candidate_rank"]) == rank]
        rng.shuffle(candidates)
        regular.extend(candidates[:regular_per_rank])
    return hard + regular


def to_values(rows, features):
    return np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float32)


def probabilities(model, values, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(values).to(device)).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def main():
    args = parse_args()
    if not 0 < args.positive_per_batch < args.batch_size:
        raise ValueError("--positive_per_batch must be in [1, batch_size - 1].")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows, paths = load_rows(args.feature_dir, args.pairs)
    positives = [row for row in rows if row["label"] == "positive"]
    negatives = build_negative_pool(rows, args.regular_negative_per_rank, rng)
    if not positives or not negatives:
        raise ValueError("Need both positive samples and a negative pool.")
    raw = to_values(positives + negatives, args.features)
    standardizer = Standardizer.fit(raw, args.features)
    positive_values = standardizer.transform(to_values(positives, args.features))
    negative_values = standardizer.transform(to_values(negatives, args.features))
    device = torch.device(args.device)
    model = LogisticDecisionHead(len(args.features)) if args.head == "logistic" else MLPDecisionHead(len(args.features), args.hidden_dim)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    negative_per_batch = args.batch_size - args.positive_per_batch
    epoch_losses = []
    for epoch in range(args.epochs):
        pos_order = np.random.permutation(len(positive_values))
        neg_order = np.random.choice(len(negative_values), size=len(pos_order) * negative_per_batch // args.positive_per_batch + negative_per_batch, replace=len(negative_values) < len(pos_order) * negative_per_batch // args.positive_per_batch + negative_per_batch)
        model.train()
        losses = []
        for start in range(0, len(pos_order), args.positive_per_batch):
            pos_indices = pos_order[start:start + args.positive_per_batch]
            if len(pos_indices) != args.positive_per_batch:
                continue
            neg_start = (start // args.positive_per_batch) * negative_per_batch
            neg_indices = neg_order[neg_start:neg_start + negative_per_batch]
            features = np.concatenate((positive_values[pos_indices], negative_values[neg_indices]))
            labels = np.concatenate((np.ones(len(pos_indices)), np.zeros(len(neg_indices)))).astype(np.float32)
            order = np.random.permutation(len(labels))
            tensor_features = torch.from_numpy(features[order]).to(device)
            tensor_labels = torch.from_numpy(labels[order]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(tensor_features), tensor_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_losses.append(float(np.mean(losses)))
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={epoch_losses[-1]:.6f}")
    all_values = standardizer.transform(to_values(positives + negatives, args.features))
    labels = np.asarray([1] * len(positives) + [0] * len(negatives))
    scores = probabilities(model, all_values, device)
    output_dir.mkdir(parents=True)
    checkpoint = {
        "head": args.head, "hidden_dim": args.hidden_dim, "feature_names": args.features,
        "standardizer": asdict(standardizer), "state_dict": model.cpu().state_dict(),
        "training_protocol": {
            "positive_source": "all sequence-00 top-50 positive pairs",
            "negative_source": "all rank<=10 negatives plus rank-stratified ranks 11-50 negatives",
            "batch_size": args.batch_size, "positive_per_batch": args.positive_per_batch,
            "negative_per_batch": negative_per_batch, "ignore_used_for_bce": False,
        },
    }
    torch.save(checkpoint, output_dir / "decision_head.pt")
    report = {
        "feature_dir": str(Path(args.feature_dir).resolve()), "pairs": str(Path(args.pairs).resolve()), "feature_chunk_count": len(paths),
        "feature_rows": len(rows), "class_counts_full": {name: sum(row["label"] == name for row in rows) for name in ("positive", "negative", "ignore")},
        "train_positive_rows": len(positives), "negative_pool_rows": len(negatives),
        "features": args.features, "head": args.head, "epochs": args.epochs,
        "batch_composition": {"positive": args.positive_per_batch, "negative": negative_per_batch},
        "training_pool_metrics": {"roc_auc": float(roc_auc_score(labels, scores)), "average_precision": float(average_precision_score(labels, scores))},
        "epoch_losses": epoch_losses,
        "note": "Sequence 02 is reserved for full-distribution calibration; no threshold is selected during training.",
    }
    with (output_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved full-data decision head to {output_dir / 'decision_head.pt'}")


if __name__ == "__main__":
    main()
