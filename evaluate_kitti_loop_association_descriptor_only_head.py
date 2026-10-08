"""Association eval for the descriptor-only logistic head.

Existing VGGT three-way chunks are only reread for features and labels. The
unique variable versus frozen accept-only is the 00-trained descriptor-only
logistic plus its 02-only Platt / accept probability. Official geometry hard
gates are reused from that frozen file. Official three-way chunks and
accept-only reports are not modified.
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
    association_precision_recall_curve,
    frozen_raw_logit_threshold,
    historical_positive_query_ids,
    load_frozen_scoring_head,
    select_one_candidate,
)
from summarize_kitti_frozen_three_way import load_evaluation, safe_divide, validate_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI associations with the descriptor-only logistic head."
    )
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--evaluation_dir", required=True)
    parser.add_argument("--frozen_decision", required=True)
    parser.add_argument("--sequence_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cluster_gap", type=int, default=5)
    parser.add_argument("--positive_radius", type=float, default=5.0)
    parser.add_argument("--min_temporal_gap", type=int, default=100)
    parser.add_argument("--distance_axes", type=int, nargs=2, default=(0, 2))
    return parser.parse_args()


def calibrated_probability(raw_logit, platt_coef, platt_intercept):
    raw_score = 1.0 / (1.0 + np.exp(-raw_logit))
    clipped = np.log(np.clip(raw_score, 1e-6, 1 - 1e-6) / np.clip(1 - raw_score, 1e-6, 1))
    return float(1.0 / (1.0 + np.exp(-(platt_coef * clipped + platt_intercept))))


def geometry_accept(row, rules):
    return (
        float(row.mean_track_confidence) >= rules["track_confidence_accept_min"]
        and float(row.mean_visibility) >= rules["visibility_accept_min"]
        and float(row.median_3d_residual) <= rules["median_3d_residual_accept_max"]
    )


def rescore(result, model, features, standardizer, platt_coef, platt_intercept, rules):
    if set(features) - set(result.columns):
        raise RuntimeError(f"Evaluation is missing head features: {features}.")
    missing_geometry = {
        "mean_track_confidence",
        "mean_visibility",
        "median_3d_residual",
    } - set(result.columns)
    if missing_geometry:
        raise RuntimeError(f"Evaluation is missing geometry columns: {sorted(missing_geometry)}.")
    values = result[features].to_numpy(dtype=np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(standardizer.transform(values))).numpy().astype(np.float64)
    scored = result.copy()
    scored["raw_logit"] = logits
    scored["calibrated_probability"] = [
        calibrated_probability(float(logit), platt_coef, platt_intercept) for logit in logits
    ]
    accept_probability = float(rules["accept_probability"])
    states = []
    for row in scored.itertuples(index=False):
        if float(row.calibrated_probability) >= accept_probability and geometry_accept(row, rules):
            states.append("accept")
        else:
            states.append("non_accept")
    scored["state"] = states
    return scored


def build_selections(result, cluster_gap):
    rows = []
    for query_index, group in result.groupby("query_dataset_index", sort=True):
        accepted = group.loc[group.state.eq("accept")]
        selected = select_one_candidate(accepted, cluster_gap)
        base = {
            "query_dataset_index": int(query_index),
            "query_center_idx": int(group.query_center_idx.iloc[0]),
            "retrieved_positive": bool(group.label.eq("positive").any()),
            "eligible_accept_candidates": int(len(accepted)),
        }
        if selected is None:
            rows.append(
                {
                    **base,
                    "association_state": "abstain",
                    "selected_label": "",
                    "selected_pair_row_index": pd.NA,
                    "candidate_dataset_index": pd.NA,
                    "candidate_center_idx": pd.NA,
                    "candidate_rank": pd.NA,
                    "gt_distance_m": np.nan,
                    "raw_logit": np.nan,
                    "calibrated_probability": np.nan,
                    "candidate_count": 0,
                    "cluster_count": 0,
                    "selected_cluster_size": 0,
                }
            )
            continue
        rows.append(
            {
                **base,
                "association_state": "selected",
                "selected_label": selected.label,
                "selected_pair_row_index": int(selected.pair_row_index),
                "candidate_dataset_index": int(selected.candidate_dataset_index),
                "candidate_center_idx": int(selected.candidate_center_idx),
                "candidate_rank": int(selected.candidate_rank),
                "gt_distance_m": float(selected.gt_distance_m),
                "raw_logit": float(selected.raw_logit),
                "calibrated_probability": float(selected.calibrated_probability),
                "candidate_count": int(selected.candidate_count),
                "cluster_count": int(selected.cluster_count),
                "selected_cluster_size": int(selected.selected_cluster_size),
            }
        )
    return pd.DataFrame(rows)


def build_report(pairs, selections, historical_positive_queries, sequence_id, inputs, protocol):
    selected = selections.loc[selections.association_state.eq("selected")]
    abstained = int(selections.association_state.eq("abstain").sum())
    selected_positive = int(selected.selected_label.eq("positive").sum())
    selected_negative = int(selected.selected_label.eq("negative").sum())
    selected_ignore = int(selected.selected_label.eq("ignore").sum())
    retrieved_positive_queries = int(pairs.loc[pairs.label.eq("positive"), "query_dataset_index"].nunique())
    if selected.empty:
        curve, average_precision, recall_at_100_precision = association_precision_recall_curve(
            pd.DataFrame(columns=["query_dataset_index", "raw_logit", "label"]),
            historical_positive_queries,
        )
    else:
        curve, average_precision, recall_at_100_precision = association_precision_recall_curve(
            selected[["query_dataset_index", "raw_logit", "selected_label"]].rename(
                columns={"selected_label": "label"}
            ),
            historical_positive_queries,
        )
    return {
        "sequence_id": str(sequence_id),
        "method": "descriptor_only_logistic",
        "inputs": inputs,
        "protocol": protocol,
        "retrieval_queries": int(pairs.query_dataset_index.nunique()),
        "historical_positive_queries": int(historical_positive_queries),
        "retrieved_positive_queries": retrieved_positive_queries,
        "retrieval_missed_positive_queries": int(historical_positive_queries - retrieved_positive_queries),
        "queries_with_selected_association": int(len(selected)),
        "queries_abstained": abstained,
        "selected_label_counts": selected.selected_label.value_counts().to_dict(),
        "correct_associations": selected_positive,
        "explicitly_wrong_negative_associations": selected_negative,
        "gray_ignore_associations": selected_ignore,
        "association_precision_clear_labels": safe_divide(
            selected_positive,
            selected_positive + selected_negative,
        ),
        "association_precision_strict": safe_divide(selected_positive, len(selected)),
        "association_recall_conditional_retrieval": safe_divide(
            selected_positive,
            retrieved_positive_queries,
        ),
        "association_recall_end_to_end": safe_divide(
            selected_positive,
            historical_positive_queries,
        ),
        "abstention_rate_retrieval_queries": safe_divide(abstained, len(selections)),
        "association_average_precision": average_precision,
        "association_recall_at_100_precision": recall_at_100_precision,
        "pr_curve_points": len(curve),
        "note": (
            "Single-component ablation of classifier inputs: 00 logistic trained on "
            "descriptor_similarity only, Platt and accept probability calibrated on 02, "
            "official geometry hard gates copied not retuned. Official 7-d reports are "
            "not modified."
        ),
    }, curve


def main():
    args = parse_args()
    if args.cluster_gap < 0:
        raise ValueError("--cluster_gap must be non-negative.")
    if args.positive_radius <= 0:
        raise ValueError("--positive_radius must be positive.")
    if args.min_temporal_gap < 1:
        raise ValueError("--min_temporal_gap must be positive.")

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
    if features != ["descriptor_similarity"]:
        raise RuntimeError(f"Expected descriptor_similarity-only head, got {features}.")
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
            "descriptor-only logistic accept (02-calibrated probability and copied "
            "official geometry gates); temporal clusters; highest uncalibrated raw logit"
        ),
        "ignore_policy": (
            "strict precision treats ignore as incorrect; clear-label precision excludes ignore"
        ),
        "accept_probability": float(rules["accept_probability"]),
        "frozen_accept_raw_logit_threshold": frozen_raw_logit_threshold(frozen),
        "unique_variable": (
            "replace the 7-d VGGT-geometry logistic with a descriptor_similarity-only "
            "logistic trained on 00 and calibrated on 02"
        ),
        "held_fixed": [
            "window seq_len=5 stride=1 center-first",
            "historical top-50",
            "historical gap>=100",
            "GT X/Z 5m/25m, no yaw",
            "official 00 training recipe except feature set",
            "official frozen geometry hard gates",
            "accept-only single window",
        ],
    }
    inputs = {
        "descriptors": str(Path(args.descriptors).resolve()),
        "pairs": str(Path(args.pairs).resolve()),
        "evaluation_dir": str(Path(args.evaluation_dir).resolve()),
        "frozen_decision": str(Path(args.frozen_decision).resolve()),
        "mode": "descriptor_only_logistic",
    }
    report, curve = build_report(
        pairs=pairs,
        selections=selections,
        historical_positive_queries=historical_positive_queries,
        sequence_id=args.sequence_id,
        inputs=inputs,
        protocol=protocol,
    )
    selections.to_csv(output_dir / "selected_associations.csv", index=False)
    curve.to_csv(output_dir / "association_precision_recall_curve.csv", index=False)
    with (output_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
