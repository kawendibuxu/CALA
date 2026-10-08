"""Rank-1 retrieval + frozen full backend association ablation.

The only variable versus frozen accept-only is the candidate pool: each query
keeps its historical UniPR descriptor rank-1 pair, then the frozen 02 backend
is applied unchanged (VGGT features already in the three-way chunks, logistic,
Platt, p >= 0.947, geometry hard gates, accept-only, raw-logit / cluster
selection). Existing accept-only reports and previous ablations are not
modified.
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
    add_raw_logits,
    association_precision_recall_curve,
    frozen_raw_logit_threshold,
    historical_positive_query_ids,
    load_frozen_scoring_head,
    select_one_candidate,
)
from summarize_kitti_frozen_three_way import load_evaluation, safe_divide, validate_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate KITTI associations using descriptor rank-1 candidates "
            "and the frozen accept-only backend."
        )
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


def rank1_group(group):
    ranked = group.loc[group.candidate_rank.eq(1)]
    if len(ranked) != 1:
        query_index = int(group.query_dataset_index.iloc[0])
        raise RuntimeError(
            f"Query {query_index} has {len(ranked)} historical rank-1 evaluation rows; expected 1."
        )
    return ranked.iloc[0]


def build_rank1_backend_selections(result, cluster_gap):
    rows = []
    for query_index, group in result.groupby("query_dataset_index", sort=True):
        rank1 = rank1_group(group)
        accepted = group.loc[group.candidate_rank.eq(1) & group.state.eq("accept")]
        selected = select_one_candidate(accepted, cluster_gap)
        base = {
            "query_dataset_index": int(query_index),
            "query_center_idx": int(group.query_center_idx.iloc[0]),
            "retrieved_positive_top50": bool(group.label.eq("positive").any()),
            "rank1_positive": bool(rank1.label == "positive"),
            "rank1_label": rank1.label,
            "rank1_three_way_state": rank1.state,
            "rank1_pair_row_index": int(rank1.pair_row_index),
            "rank1_candidate_rank": int(rank1.candidate_rank),
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
                    "selected_three_way_state": "",
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
                "selected_three_way_state": selected.state if "state" in selected.index else "accept",
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
    rank1_positive_queries = int(selections.rank1_positive.sum())
    top50_retrieved_positive_queries = int(selections.retrieved_positive_top50.sum())
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
    rank1_state_counts = selections.rank1_three_way_state.value_counts().to_dict()
    return {
        "sequence_id": str(sequence_id),
        "method": "rank1_full_backend",
        "inputs": inputs,
        "protocol": protocol,
        "retrieval_queries": int(pairs.query_dataset_index.nunique()),
        "historical_positive_queries": int(historical_positive_queries),
        "retrieved_positive_queries": rank1_positive_queries,
        "top50_retrieved_positive_queries": top50_retrieved_positive_queries,
        "retrieval_missed_positive_queries": int(historical_positive_queries - rank1_positive_queries),
        "rank1_three_way_state_counts": rank1_state_counts,
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
            rank1_positive_queries,
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
            "Ablation of candidate retrieval only: historical UniPR rank-1, then the "
            "frozen accept-only backend. The 00 head, 02 Platt, 0.947 threshold and "
            "geometry gates are unchanged. Accept-only reports and previous ablation "
            "reports are not modified. retrieved_positive_queries counts rank-1 GT "
            "positives. Association AP/RP100 rank the unique selected window by raw "
            "logit and are not comparable to the frozen geometry-gated AP."
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
    if "state" not in result.columns:
        raise RuntimeError("Evaluation is missing the frozen three-way state column.")

    model, features, standardizer, rules, frozen = load_frozen_scoring_head(args.frozen_decision)
    accept_probability = float(rules["accept_probability"])
    result = add_raw_logits(result, model, features, standardizer)
    selections = build_rank1_backend_selections(result, args.cluster_gap)

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
    if not selections.candidate_rank.dropna().eq(1).all():
        raise RuntimeError("Selected associations must all be descriptor rank-1.")
    selected = selections.loc[selections.association_state.eq("selected")]
    if not selected.empty and not selected.selected_three_way_state.eq("accept").all():
        raise RuntimeError("Rank-1 backend selected a pair whose frozen three-way state is not accept.")

    protocol = {
        "positive_radius_m": args.positive_radius,
        "minimum_temporal_gap": args.min_temporal_gap,
        "distance_axes": list(args.distance_axes),
        "temporal_cluster_gap_frames": args.cluster_gap,
        "ground_truth": "center-frame X/Z distance only; yaw and visual overlap are not used",
        "candidate_selection": (
            "historical UniPR descriptor rank-1 only; then frozen accept-only backend "
            "(p >= 0.947 and geometry hard gates); temporal clusters; highest uncalibrated raw logit"
        ),
        "ignore_policy": (
            "strict precision treats ignore as incorrect; clear-label precision excludes ignore"
        ),
        "accept_probability": accept_probability,
        "frozen_accept_raw_logit_threshold": frozen_raw_logit_threshold(frozen),
        "unique_variable": (
            "restrict the frozen backend to the historical descriptor rank-1 pair; "
            "keep VGGT features, logistic, Platt, 0.947 and geometry gates"
        ),
        "held_fixed": [
            "window seq_len=5 stride=1 center-first",
            "historical gap>=100",
            "GT X/Z 5m/25m, no yaw",
            "frozen 00 logistic and 02 Platt",
            "accept_probability=0.947",
            "geometry hard gates",
            "accept-only single window",
        ],
    }
    inputs = {
        "descriptors": str(Path(args.descriptors).resolve()),
        "pairs": str(Path(args.pairs).resolve()),
        "evaluation_dir": str(Path(args.evaluation_dir).resolve()),
        "frozen_decision": str(Path(args.frozen_decision).resolve()),
        "mode": "rank1_full_backend",
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
