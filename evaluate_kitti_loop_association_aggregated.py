"""Apply temporal association aggregation on top of frozen accept-only selections.

This script does not retune UniPR, VGGT, Platt, three-way thresholds, or the
5m/25m ground-truth protocol. Frozen accept associations are copied unchanged.
Abstaining queries may inherit a neighboring accepted historical cluster.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from evaluate_kitti_loop_association import (
    REQUIRED_ASSOCIATION_COLUMNS,
    add_raw_logits,
    build_potential_predictions,
    build_report,
    build_selections,
    historical_positive_query_ids,
    load_frozen_scoring_head,
    select_one_candidate,
)
from summarize_kitti_frozen_three_way import load_evaluation, validate_evaluation


DEFAULT_WINDOW = 3
DEFAULT_CLUSTER_GAP = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate frozen KITTI loop associations across neighboring queries."
    )
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--evaluation_dir", required=True)
    parser.add_argument("--frozen_decision", required=True)
    parser.add_argument("--sequence_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--aggregation_window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--cluster_gap", type=int, default=DEFAULT_CLUSTER_GAP)
    parser.add_argument("--positive_radius", type=float, default=5.0)
    parser.add_argument("--min_temporal_gap", type=int, default=100)
    parser.add_argument("--distance_axes", type=int, nargs=2, default=(0, 2))
    return parser.parse_args()


def geometry_mask(frame, rules):
    return (
        frame.mean_track_confidence.ge(rules["track_confidence_accept_min"])
        & frame.mean_visibility.ge(rules["visibility_accept_min"])
        & frame.median_3d_residual.le(rules["median_3d_residual_accept_max"])
    )


def neighbor_seeds(frozen_selected, query_index, window):
    if frozen_selected.empty:
        return frozen_selected.iloc[0:0]
    delta = (frozen_selected.query_dataset_index - query_index).abs()
    return frozen_selected.loc[delta.between(1, window)]


def select_aggregated_candidate(group, seeds, cluster_gap, rules):
    if seeds.empty:
        return None, None
    eligible_parts = []
    for seed in seeds.itertuples(index=False):
        part = group.loc[
            (group.candidate_center_idx - seed.candidate_center_idx).abs().le(cluster_gap)
            & group.state.ne("reject")
            & geometry_mask(group, rules)
        ]
        if not part.empty:
            eligible_parts.append(part)
    if not eligible_parts:
        return None, None
    eligible = pd.concat(eligible_parts, ignore_index=True).drop_duplicates("pair_row_index")
    selected = select_one_candidate(eligible, cluster_gap)
    matching = seeds.loc[
        (seeds.candidate_center_idx - selected.candidate_center_idx).abs().le(cluster_gap)
    ].copy()
    matching["query_distance"] = (matching.query_dataset_index - int(group.query_dataset_index.iloc[0])).abs()
    matching = matching.sort_values(
        ["query_distance", "query_dataset_index"],
        kind="stable",
    )
    seed = matching.iloc[0]
    return selected, seed


def _value(row, *names):
    for name in names:
        if name in row.index:
            return row[name]
    raise KeyError(f"Missing one of {names}")


def selection_row(query_index, group, selected, association_state, source, seed=None):
    base = {
        "query_dataset_index": int(query_index),
        "query_center_idx": int(group.query_center_idx.iloc[0]),
        "retrieved_positive": bool(group.label.eq("positive").any()),
        "aggregation_source": source,
        "seed_query_dataset_index": pd.NA if seed is None else int(seed.query_dataset_index),
        "seed_candidate_center_idx": pd.NA if seed is None else int(seed.candidate_center_idx),
    }
    if selected is None:
        return {
            **base,
            "association_state": "abstain",
            "selected_label": "",
            "selected_pair_row_index": pd.NA,
            "candidate_dataset_index": pd.NA,
            "candidate_center_idx": pd.NA,
            "candidate_rank": pd.NA,
            "gt_distance_m": float("nan"),
            "raw_logit": float("nan"),
            "calibrated_probability": float("nan"),
            "candidate_count": 0,
            "cluster_count": 0,
            "selected_cluster_size": 0,
        }
    return {
        **base,
        "association_state": association_state,
        "selected_label": _value(selected, "selected_label", "label"),
        "selected_pair_row_index": int(_value(selected, "selected_pair_row_index", "pair_row_index")),
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


def aggregate_selections(result, frozen_selections, window, cluster_gap, rules):
    grouped = {int(query_index): group for query_index, group in result.groupby("query_dataset_index", sort=True)}
    frozen_selected = frozen_selections.loc[frozen_selections.association_state.eq("selected")].copy()
    rows = []
    for query_index, group in grouped.items():
        frozen_row = frozen_selections.loc[frozen_selections.query_dataset_index.eq(query_index)].iloc[0]
        if frozen_row.association_state == "selected":
            rows.append(selection_row(query_index, group, frozen_row, "selected", "frozen_accept"))
            continue
        selected, seed = select_aggregated_candidate(
            group,
            neighbor_seeds(frozen_selected, query_index, window),
            cluster_gap,
            rules,
        )
        if selected is None:
            rows.append(selection_row(query_index, group, None, "abstain", ""))
        else:
            rows.append(selection_row(query_index, group, selected, "selected", "temporal_aggregation", seed))
    return pd.DataFrame(rows)


def compare_with_frozen(frozen_selections, aggregated_selections):
    frozen_selected = frozen_selections.loc[frozen_selections.association_state.eq("selected")]
    kept = aggregated_selections.loc[aggregated_selections.aggregation_source.eq("frozen_accept")]
    if len(kept) != len(frozen_selected):
        raise RuntimeError("Aggregation changed the number of frozen accept associations.")
    merged = frozen_selected.merge(
        kept,
        on="query_dataset_index",
        suffixes=("_frozen", "_kept"),
    )
    if not merged.selected_pair_row_index_frozen.astype("Int64").eq(
        merged.selected_pair_row_index_kept.astype("Int64")
    ).all():
        raise RuntimeError("Aggregation overwrote one or more frozen accept associations.")
    promoted = aggregated_selections.loc[aggregated_selections.aggregation_source.eq("temporal_aggregation")]
    return {
        "frozen_accept_associations_kept": int(len(kept)),
        "queries_promoted_by_aggregation": int(len(promoted)),
        "promoted_positive": int(promoted.selected_label.eq("positive").sum()),
        "promoted_ignore": int(promoted.selected_label.eq("ignore").sum()),
        "promoted_negative": int(promoted.selected_label.eq("negative").sum()),
        "promoted_query_dataset_indices": [int(x) for x in promoted.query_dataset_index.tolist()],
    }


def main():
    args = parse_args()
    if args.aggregation_window < 1:
        raise ValueError("--aggregation_window must be positive.")
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
    result = add_raw_logits(result, model, features, standardizer)
    frozen_selections = build_selections(result, args.cluster_gap)
    aggregated_selections = aggregate_selections(
        result,
        frozen_selections,
        args.aggregation_window,
        args.cluster_gap,
        rules,
    )
    comparison = compare_with_frozen(frozen_selections, aggregated_selections)
    if comparison["promoted_negative"] != 0:
        raise RuntimeError("Aggregation produced an explicit negative association.")

    descriptor_data = torch.load(args.descriptors, map_location="cpu", weights_only=False)
    historical_positive_ids = historical_positive_query_ids(
        descriptor_data["center_indices"],
        descriptor_data["translations"],
        args.min_temporal_gap,
        args.positive_radius,
        args.distance_axes,
    )
    potential_predictions = build_potential_predictions(result, rules, args.cluster_gap)
    protocol = {
        "positive_radius_m": args.positive_radius,
        "minimum_temporal_gap": args.min_temporal_gap,
        "distance_axes": list(args.distance_axes),
        "temporal_cluster_gap_frames": args.cluster_gap,
        "ground_truth": "center-frame X/Z distance only; yaw and visual overlap are not used",
        "aggregation_rule": {
            "name": "frozen-accept temporal cluster inheritance",
            "applied_after": "frozen accept-only single-window selection",
            "window_queries": args.aggregation_window,
            "cluster_gap_frames": args.cluster_gap,
            "seed": "unchanged frozen accept associations",
            "eligibility": "same historical cluster; not reject; frozen visibility/track/residual gates",
            "does_not_change": [
                "UniPR descriptors",
                "VGGT geometry",
                "sequence-00 head",
                "sequence-02 Platt parameters",
                "accept probability 0.9471",
                "5m/25m labels",
            ],
        },
        "ignore_policy": (
            "strict precision treats ignore as incorrect; clear-label precision excludes ignore; "
            "explicit negative associations are forbidden"
        ),
    }
    inputs = {
        "descriptors": str(Path(args.descriptors)),
        "pairs": str(Path(args.pairs)),
        "evaluation_dir": str(Path(args.evaluation_dir)),
        "frozen_decision": str(Path(args.frozen_decision)),
    }
    frozen_report, _ = build_report(
        result=result,
        selections=frozen_selections,
        potential_predictions=potential_predictions,
        historical_positive_queries=len(historical_positive_ids),
        sequence_id=args.sequence_id,
        inputs=inputs,
        protocol={
            "positive_radius_m": args.positive_radius,
            "minimum_temporal_gap": args.min_temporal_gap,
            "distance_axes": list(args.distance_axes),
            "temporal_cluster_gap_frames": args.cluster_gap,
            "ground_truth": "center-frame X/Z distance only; yaw and visual overlap are not used",
        },
        frozen=frozen,
    )
    aggregated_report, curve = build_report(
        result=result,
        selections=aggregated_selections,
        potential_predictions=potential_predictions,
        historical_positive_queries=len(historical_positive_ids),
        sequence_id=args.sequence_id,
        inputs=inputs,
        protocol=protocol,
        frozen=frozen,
    )
    aggregated_report["protocol"]["candidate_selection"] = (
        "frozen accept copied unchanged; abstaining queries may inherit a neighboring accepted "
        "historical cluster using the same raw-logit tie-break"
    )
    aggregated_report["protocol"]["ignore_policy"] = protocol["ignore_policy"]
    aggregated_report["frozen_accept_only_baseline"] = {
        "queries_with_selected_association": frozen_report["queries_with_selected_association"],
        "correct_associations": frozen_report["correct_associations"],
        "gray_ignore_associations": frozen_report["gray_ignore_associations"],
        "explicitly_wrong_negative_associations": frozen_report["explicitly_wrong_negative_associations"],
        "association_precision_strict": frozen_report["association_precision_strict"],
        "association_recall_conditional_retrieval": frozen_report["association_recall_conditional_retrieval"],
        "association_recall_end_to_end": frozen_report["association_recall_end_to_end"],
    }
    aggregated_report["aggregation"] = comparison
    aggregated_report["note"] = (
        "This report adds a post-hoc temporal aggregation rule after frozen accept-only selection. "
        "It does not retune the sequence-00/02 model, overwrite frozen three-way chunks, or change 5m/25m labels."
    )

    frozen_selections.to_csv(output_dir / "frozen_accept_associations.csv", index=False)
    aggregated_selections.to_csv(output_dir / "selected_associations.csv", index=False)
    curve.to_csv(output_dir / "association_precision_recall_curve.csv", index=False)
    with (output_dir / "report.json").open("w") as handle:
        json.dump(aggregated_report, handle, indent=2)
    print(json.dumps(aggregated_report, indent=2))


if __name__ == "__main__":
    main()
