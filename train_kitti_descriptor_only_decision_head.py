"""Train a sequence-00 logistic head using descriptor similarity only.

The only variable versus the official 7-d head is the feature set:
descriptor_similarity, with all VGGT geometry features removed. Sampling,
batch composition, seed and optimizer match train_kitti_full_decision_head.py.
Sequence 02 is not read.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from train_kitti_decision_head import LogisticDecisionHead, Standardizer
from train_kitti_full_decision_head import (
    build_negative_pool,
    load_rows,
    probabilities,
    to_values,
)

FEATURES = ["descriptor_similarity"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the official 00 protocol with descriptor_similarity only."
    )
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--positive_per_batch", type=int, default=16)
    parser.add_argument("--regular_negative_per_rank", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


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
    raw = to_values(positives + negatives, FEATURES)
    standardizer = Standardizer.fit(raw, FEATURES)
    positive_values = standardizer.transform(to_values(positives, FEATURES))
    negative_values = standardizer.transform(to_values(negatives, FEATURES))
    device = torch.device(args.device)
    model = LogisticDecisionHead(len(FEATURES))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    negative_per_batch = args.batch_size - args.positive_per_batch
    epoch_losses = []
    for epoch in range(args.epochs):
        pos_order = np.random.permutation(len(positive_values))
        neg_needed = len(pos_order) * negative_per_batch // args.positive_per_batch + negative_per_batch
        neg_order = np.random.choice(
            len(negative_values),
            size=neg_needed,
            replace=len(negative_values) < neg_needed,
        )
        model.train()
        losses = []
        for start in range(0, len(pos_order), args.positive_per_batch):
            pos_indices = pos_order[start : start + args.positive_per_batch]
            if len(pos_indices) != args.positive_per_batch:
                continue
            neg_start = (start // args.positive_per_batch) * negative_per_batch
            neg_indices = neg_order[neg_start : neg_start + negative_per_batch]
            features = np.concatenate((positive_values[pos_indices], negative_values[neg_indices]))
            labels = np.concatenate(
                (np.ones(len(pos_indices)), np.zeros(len(neg_indices)))
            ).astype(np.float32)
            order = np.random.permutation(len(labels))
            tensor_features = torch.from_numpy(features[order]).to(device)
            tensor_labels = torch.from_numpy(labels[order]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(tensor_features), tensor_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_losses.append(float(np.mean(losses)))
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={epoch_losses[-1]:.6f}", flush=True)
    all_values = standardizer.transform(to_values(positives + negatives, FEATURES))
    labels = np.asarray([1] * len(positives) + [0] * len(negatives))
    scores = probabilities(model, all_values, device)
    output_dir.mkdir(parents=True)
    checkpoint = {
        "head": "logistic",
        "hidden_dim": 16,
        "feature_names": list(FEATURES),
        "standardizer": asdict(standardizer),
        "state_dict": model.cpu().state_dict(),
        "training_protocol": {
            "positive_source": "all sequence-00 top-50 positive pairs",
            "negative_source": "all rank<=10 negatives plus rank-stratified ranks 11-50 negatives",
            "batch_size": args.batch_size,
            "positive_per_batch": args.positive_per_batch,
            "negative_per_batch": negative_per_batch,
            "ignore_used_for_bce": False,
            "unique_variable": "logistic input is descriptor_similarity only; VGGT geometry features are excluded",
        },
    }
    torch.save(checkpoint, output_dir / "decision_head.pt")
    report = {
        "feature_dir": str(Path(args.feature_dir).resolve()),
        "pairs": str(Path(args.pairs).resolve()),
        "feature_chunk_count": len(paths),
        "feature_rows": len(rows),
        "class_counts_full": {
            name: sum(row["label"] == name for row in rows) for name in ("positive", "negative", "ignore")
        },
        "train_positive_rows": len(positives),
        "negative_pool_rows": len(negatives),
        "features": list(FEATURES),
        "head": "logistic",
        "epochs": args.epochs,
        "batch_composition": {"positive": args.positive_per_batch, "negative": negative_per_batch},
        "seed": args.seed,
        "training_pool_metrics": {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
        },
        "epoch_losses": epoch_losses,
        "unique_variable": (
            "remove VGGT geometry features from the logistic input; keep the official 00 training recipe"
        ),
        "note": "Sequence 02 is reserved for calibration; no threshold is selected during training.",
    }
    with (output_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved descriptor-only decision head to {output_dir / 'decision_head.pt'}", flush=True)


if __name__ == "__main__":
    main()
