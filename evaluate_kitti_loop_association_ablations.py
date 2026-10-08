"""Same-protocol association ablations that do not retune the frozen system.

Modes:
  descriptor_rank1  — select historical UniPR rank-1; no VGGT, no decision head.
  score_rerank      — select the highest frozen raw logit in top-50; no accept gate.

Existing accept-only reports and frozen three-way chunks are not modified.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from evaluate_kitti_loop_association import (
    REQUIRED_ASSOCIATION_COLUMNS,
    add_raw_logits,
    association_precision_recall_curve,
    historical_positive_query_ids,
    load_frozen_scoring_head,
    select_one_candidate,
)
from summarize_kitti_frozen_three_way import load_evaluation, safe_divide, validate_evaluation


MODES = ("descriptor_rank1", "score_rerank")
SELECTION_DESCRIPTION = {
    "descriptor_rank1": (
        "historical UniPR rank-1 only; no VGGT geometry, no decision head, no accept gate"
    ),
    "score_rerank": (
        "all top-50 candidates; highest uncalibrated raw logit; "
        "ties by lower candidate rank, center index, then pair_row_index; no accept gate"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate frozen association ablations on KITTI.")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--sequence_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--evaluation_dir", default=None)
    parser.add_argument("--frozen_decision", default=None)
    parser.add_argument("--cluster_gap", type=int, default=5)
    parser.add_argument("--positive_radius", type=float, default=5.0)
    parser.add_argument("--min_temporal_gap", type=int, default=100)
    parser.add_argument("--distance_axes", type=int, nargs=2, default=(0, 2))
    return parser.parse_args()


def pair_row_index_of(selected):
    if "pair_row_index" in selected.index and pd.notna(selected.pair_row_index):
        return int(selected.pair_row_index)
    return pd.NA


def selected_row(query_index, group, selected, score_name, score_value, extra=None):
    row = {
        "query_dataset_index": int(query_index),
        "query_center_idx": int(group.query_center_idx.iloc[0]),
        "retrieved_positive": bool(group.label.eq("positive").any()),
        "association_state": "selected",
        "selected_label": selected.label,
        "selected_pair_row_index": pair_row_index_of(selected),
        "candidate_dataset_index": int(selected.candidate_dataset_index),
        "candidate_center_idx": int(selected.candidate_center_idx),
        "candidate_rank": int(selected.candidate_rank),
        "gt_distance_m": float(selected.gt_distance_m),
        "score_name": score_name,
        "score": float(score_value),
        "raw_logit": float(score_value),
    }
    if extra:
        row.update(extra)
    return row


def ensure_pair_row_index(pairs):
    if "pair_row_index" in pairs.columns:
        return pairs
    frame = pairs.copy()
    frame["pair_row_index"] = range(len(frame))
    return frame


def build_rank1_selections(pairs):
    rows = []
    for query_index, group in pairs.groupby("query_dataset_index", sort=True):
        ranked = group.loc[group.candidate_rank.eq(1)]
        if ranked.empty:
            raise RuntimeError(f"Query {int(query_index)} has no historical rank-1 candidate.")
        if len(ranked) != 1:
            raise RuntimeError(f"Query {int(query_index)} has duplicate historical rank-1 rows.")
        selected = ranked.iloc[0]
        rows.append(
            selected_row(
                query_index,
                group,
                selected,
                "descriptor_similarity",
                selected.descriptor_similarity,
                extra={
                    "candidate_count": 1,
                    "cluster_count": 1,
                    "selected_cluster_size": 1,
                },
            )
        )
    return pd.DataFrame(rows)


def build_score_rerank_selections(result, cluster_gap):
    rows = []
    for query_index, group in result.groupby("query_dataset_index", sort=True):
        selected = select_one_candidate(group, cluster_gap)
        extra = {
            "candidate_count": int(selected.candidate_count),
            "cluster_count": int(selected.cluster_count),
            "selected_cluster_size": int(selected.selected_cluster_size),
        }
        if "calibrated_probability" in selected.index and pd.notna(selected.calibrated_probability):
            extra["calibrated_probability"] = float(selected.calibrated_probability)
        if "state" in selected.index:
            extra["selected_three_way_state"] = selected.state
        rows.append(selected_row(query_index, group, selected, "raw_logit", selected.raw_logit, extra=extra))
    return pd.DataFrame(rows)


def build_ablation_report(
    pairs,
    selections,
    historical_positive_queries,
    sequence_id,
    mode,
    inputs,
    protocol,
):
    selected = selections.loc[selections.association_state.eq("selected")]
    selected_positive = int(selected.selected_label.eq("positive").sum())
    selected_negative = int(selected.selected_label.eq("negative").sum())
    selected_ignore = int(selected.selected_label.eq("ignore").sum())
    retrieved_positive_queries = int(pairs.loc[pairs.label.eq("positive"), "query_dataset_index"].nunique())
    curve_predictions = selected[["query_dataset_index", "raw_logit", "selected_label"]].rename(
        columns={"selected_label": "label"}
    )
    curve, average_precision, recall_at_100_precision = association_precision_recall_curve(
        curve_predictions,
        historical_positive_queries,
    )
    return {
        "sequence_id": str(sequence_id),
        "method": mode,
        "inputs": inputs,
        "protocol": protocol,
        "retrieval_queries": int(pairs.query_dataset_index.nunique()),
        "historical_positive_queries": int(historical_positive_queries),
        "retrieved_positive_queries": retrieved_positive_queries,
        "retrieval_missed_positive_queries": int(historical_positive_queries - retrieved_positive_queries),
        "queries_with_selected_association": int(len(selected)),
        "queries_abstained": 0,
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
        "abstention_rate_retrieval_queries": 0.0,
        "association_average_precision": average_precision,
        "association_recall_at_100_precision": recall_at_100_precision,
        "pr_curve_points": len(curve),
        "note": (
            "Ablation of association selection only. Frozen 00/02 parameters, three-way chunks, "
            "and accept-only reports are not modified. Association AP/RP100 here rank the unique "
            "selected association per query by the ablation score; they are not comparable to "
            "the frozen accept-only geometry-gated AP."
        ),
    }, curve


def main():
    args = parse_args()
    if args.cluster_gap < 0:
        raise ValueError("--cluster_gap must be non-negative.")
    if args.mode == "score_rerank":
        if not args.evaluation_dir or not args.frozen_decision:
            raise ValueError("score_rerank requires --evaluation_dir and --frozen_decision.")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = ensure_pair_row_index(pd.read_csv(args.pairs))
    descriptor_data = torch.load(args.descriptors, map_location="cpu", weights_only=False)
    historical_positive_ids = historical_positive_query_ids(
        descriptor_data["center_indices"],
        descriptor_data["translations"],
        args.min_temporal_gap,
        args.positive_radius,
        args.distance_axes,
    )
    protocol = {
        "positive_radius_m": args.positive_radius,
        "minimum_temporal_gap": args.min_temporal_gap,
        "distance_axes": list(args.distance_axes),
        "temporal_cluster_gap_frames": args.cluster_gap,
        "ground_truth": "center-frame X/Z distance only; yaw and visual overlap are not used",
        "candidate_selection": SELECTION_DESCRIPTION[args.mode],
        "ignore_policy": (
            "strict precision treats ignore as incorrect; clear-label precision excludes ignore"
        ),
        "unique_variable": (
            "historical descriptor rank-1, no geometric verification"
            if args.mode == "descriptor_rank1"
            else "remove frozen accept gate; still use frozen raw logit over top-50"
        ),
    }
    inputs = {
        "descriptors": str(Path(args.descriptors)),
        "pairs": str(Path(args.pairs)),
        "evaluation_dir": None if args.evaluation_dir is None else str(Path(args.evaluation_dir)),
        "frozen_decision": None if args.frozen_decision is None else str(Path(args.frozen_decision)),
        "mode": args.mode,
    }

    if args.mode == "descriptor_rank1":
        selections = build_rank1_selections(pairs)
    else:
        result = load_evaluation(args.evaluation_dir)
        validate_evaluation(result, len(pairs))
        missing = REQUIRED_ASSOCIATION_COLUMNS - set(result.columns)
        if missing:
            raise RuntimeError(f"Evaluation is missing association columns: {sorted(missing)}.")
        model, features, standardizer, _rules, _frozen = load_frozen_scoring_head(args.frozen_decision)
        result = add_raw_logits(result, model, features, standardizer)
        selections = build_score_rerank_selections(result, args.cluster_gap)

    expected_queries = int(pairs.query_dataset_index.nunique())
    if len(selections) != expected_queries:
        raise RuntimeError(
            f"Expected one selection per retrieval query ({expected_queries}), got {len(selections)}."
        )

    report, curve = build_ablation_report(
        pairs=pairs,
        selections=selections,
        historical_positive_queries=len(historical_positive_ids),
        sequence_id=args.sequence_id,
        mode=args.mode,
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
