"""Geometry-gate association ablation: no learned logistic score.

The only variable versus frozen accept-only is that candidates must pass the
frozen 02 vis/track/residual gates, and the surviving window is chosen by
descriptor rank. Calibrated probability and raw logit are not used. Existing
accept-only reports and previous ablations are not modified.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_kitti_loop_association import (
    REQUIRED_ASSOCIATION_COLUMNS,
    association_precision_recall_curve,
    historical_positive_query_ids,
)
from summarize_kitti_frozen_three_way import load_evaluation, safe_divide, validate_evaluation


GEOMETRY_RULE_KEYS = (
    "track_confidence_accept_min",
    "visibility_accept_min",
    "median_3d_residual_accept_max",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI associations using only frozen geometry gates."
    )
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--evaluation_dir", required=True)
    parser.add_argument("--frozen_decision", required=True)
    parser.add_argument("--sequence_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--positive_radius", type=float, default=5.0)
    parser.add_argument("--min_temporal_gap", type=int, default=100)
    parser.add_argument("--distance_axes", type=int, nargs=2, default=(0, 2))
    return parser.parse_args()


def load_frozen_geometry_rules(path):
    frozen = torch.load(path, map_location="cpu", weights_only=False)
    rules = frozen["rules"]
    missing = set(GEOMETRY_RULE_KEYS) - set(rules)
    if missing:
        raise RuntimeError(f"Frozen decision is missing geometry rules: {sorted(missing)}.")
    return {key: float(rules[key]) for key in GEOMETRY_RULE_KEYS}


def geometry_mask(frame, rules):
    return (
        frame.mean_track_confidence.ge(rules["track_confidence_accept_min"])
        & frame.mean_visibility.ge(rules["visibility_accept_min"])
        & frame.median_3d_residual.le(rules["median_3d_residual_accept_max"])
    )


def select_by_descriptor_rank(candidates):
    if candidates.empty:
        return None
    ordered = candidates.sort_values(
        ["candidate_rank", "candidate_center_idx", "pair_row_index"],
        ascending=[True, True, True],
        kind="stable",
    )
    return ordered.iloc[0]


def build_geometry_only_selections(result, rules):
    rows = []
    for query_index, group in result.groupby("query_dataset_index", sort=True):
        eligible = group.loc[geometry_mask(group, rules)]
        selected = select_by_descriptor_rank(eligible)
        base = {
            "query_dataset_index": int(query_index),
            "query_center_idx": int(group.query_center_idx.iloc[0]),
            "retrieved_positive": bool(group.label.eq("positive").any()),
            "eligible_geometry_candidates": int(len(eligible)),
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
                    "score_name": "neg_candidate_rank",
                    "score": np.nan,
                    "raw_logit": np.nan,
                    "mean_track_confidence": np.nan,
                    "mean_visibility": np.nan,
                    "median_3d_residual": np.nan,
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
                "score_name": "neg_candidate_rank",
                "score": float(-selected.candidate_rank),
                "raw_logit": float(-selected.candidate_rank),
                "mean_track_confidence": float(selected.mean_track_confidence),
                "mean_visibility": float(selected.mean_visibility),
                "median_3d_residual": float(selected.median_3d_residual),
                "selected_three_way_state": selected.state if "state" in selected.index else "",
                "candidate_count": int(len(eligible)),
                "cluster_count": 1,
                "selected_cluster_size": 1,
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
        "method": "geometry_gates_only",
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
            "Ablation of association selection only. Frozen 02 geometry gates are reused; "
            "the 00 logistic head and 02 probability threshold are not used. Accept-only "
            "reports and previous ablation reports are not modified. Association AP/RP100 "
            "rank the unique selected window by negative candidate rank and are not "
            "comparable to the frozen accept-only AP."
        ),
    }, curve


def main():
    args = parse_args()
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
    rules = load_frozen_geometry_rules(args.frozen_decision)
    selections = build_geometry_only_selections(result, rules)

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
    if len(selections) != pairs.query_dataset_index.nunique():
        raise RuntimeError(
            f"Expected one row per retrieval query ({pairs.query_dataset_index.nunique()}), "
            f"got {len(selections)}."
        )

    protocol = {
        "positive_radius_m": args.positive_radius,
        "minimum_temporal_gap": args.min_temporal_gap,
        "distance_axes": list(args.distance_axes),
        "ground_truth": "center-frame X/Z distance only; yaw and visual overlap are not used",
        "candidate_selection": (
            "frozen 02 vis/track/residual gates only; among passers pick lowest "
            "candidate rank, then center index, then pair_row_index; no logistic score"
        ),
        "ignore_policy": (
            "strict precision treats ignore as incorrect; clear-label precision excludes ignore"
        ),
        "geometry_rules": rules,
        "unique_variable": (
            "remove the learned 00/02 score; keep only frozen geometry accept gates"
        ),
    }
    inputs = {
        "descriptors": str(Path(args.descriptors)),
        "pairs": str(Path(args.pairs)),
        "evaluation_dir": str(Path(args.evaluation_dir)),
        "frozen_decision": str(Path(args.frozen_decision)),
        "mode": "geometry_gates_only",
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
