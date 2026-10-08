"""Rescore existing VGGT chunks with a new frozen 7-d decision head.

Used for the decision-head seed study. Official three-way chunks are read only
for features and labels; their stored state is ignored. Official accept-only
reports are not modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_kitti_loop_association import (
    REQUIRED_ASSOCIATION_COLUMNS,
    frozen_raw_logit_threshold,
    historical_positive_query_ids,
    load_frozen_scoring_head,
)
from evaluate_kitti_loop_association_descriptor_only_head import (
    build_report,
    build_selections,
    rescore,
)
from summarize_kitti_frozen_three_way import load_evaluation, validate_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI associations by rescoring frozen VGGT chunks with a new head."
    )
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--evaluation_dir", required=True)
    parser.add_argument("--frozen_decision", required=True)
    parser.add_argument("--sequence_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cluster_gap", type=int, default=5)
    parser.add_argument("--positive_radius", type=float, default=5.0)
    parser.add_argument("--min_temporal_gap", type=int, default=100)
    parser.add_argument("--distance_axes", type=int, nargs=2, default=(0, 2))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs)
    result = load_evaluation(args.evaluation_dir)
    validate_evaluation(result, len(pairs))
    missing = REQUIRED_ASSOCIATION_COLUMNS - set(result.columns)
    if missing:
        raise RuntimeError(f"Evaluation is missing association columns: {sorted(missing)}.")

    model, features, standardizer, rules, frozen = load_frozen_scoring_head(args.frozen_decision)
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
    platt_coef = float(np.asarray(frozen["platt_coef"]).reshape(-1)[0])
    platt_intercept = float(np.asarray(frozen["platt_intercept"]).reshape(-1)[0])
    scored = rescore(result, model, features, standardizer, platt_coef, platt_intercept, rules)
    selections = build_selections(scored, args.cluster_gap)

    descriptor_data = torch.load(args.descriptors, map_location="cpu", weights_only=False)
    historical_positive_queries = len(
        historical_positive_query_ids(
            descriptor_data["center_indices"],
            descriptor_data["translations"],
            args.min_temporal_gap,
            args.positive_radius,
            args.distance_axes,
        )
    )
    expected_queries = int(pairs.query_dataset_index.nunique())
    if len(selections) != expected_queries:
        raise RuntimeError(
            f"Expected one row per retrieval query ({expected_queries}), got {len(selections)}."
        )

    protocol = {
        "positive_radius_m": args.positive_radius,
        "minimum_temporal_gap": args.min_temporal_gap,
        "distance_axes": list(args.distance_axes),
        "temporal_cluster_gap_frames": args.cluster_gap,
        "ground_truth": "center-frame X/Z distance only; yaw and visual overlap are not used",
        "candidate_selection": (
            "rescored 7-d logistic accept; temporal clusters; highest uncalibrated raw logit"
        ),
        "ignore_policy": (
            "strict precision treats ignore as incorrect; clear-label precision excludes ignore"
        ),
        "accept_probability": float(rules["accept_probability"]),
        "frozen_accept_raw_logit_threshold": frozen_raw_logit_threshold(frozen),
        "training_seed": args.seed,
        "unique_variable": "00 logistic training seed; VGGT features, pairs and association protocol are frozen",
        "held_fixed": [
            "window seq_len=5 stride=1 center-first",
            "historical top-50",
            "historical gap>=100",
            "GT X/Z 5m/25m, no yaw",
            "official 7-d feature set",
            "official 00 training recipe except seed",
            "02-only Platt and accept threshold",
            "accept-only single window",
        ],
    }
    inputs = {
        "descriptors": str(Path(args.descriptors).resolve()),
        "pairs": str(Path(args.pairs).resolve()),
        "evaluation_dir": str(Path(args.evaluation_dir).resolve()),
        "frozen_decision": str(Path(args.frozen_decision).resolve()),
        "mode": "rescored_7d_seed_head",
        "training_seed": args.seed,
    }
    report, curve = build_report(
        pairs=pairs,
        selections=selections,
        historical_positive_queries=historical_positive_queries,
        sequence_id=args.sequence_id,
        inputs=inputs,
        protocol=protocol,
    )
    report["method"] = "official_7d_logistic_seed"
    report["training_seed"] = args.seed
    report["note"] = (
        "Decision-head seed study. Existing VGGT chunks are rescored only. "
        "Official seed-17 accept-only reports are not overwritten."
    )
    selections.to_csv(output_dir / "selected_associations.csv", index=False)
    curve.to_csv(output_dir / "association_precision_recall_curve.csv", index=False)
    with (output_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
