"""UniPR rank-1 + 02-frozen similarity threshold, no VGGT.

The only added variable versus descriptor rank-1 is abstention below a
descriptor-similarity threshold. That threshold is selected only on KITTI 02
to match the frozen three-way target_precision=0.99, using query-level strict
association precision (ignore counts as incorrect). Existing accept-only
reports and previous ablations are not modified.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_kitti_loop_association import (
    association_precision_recall_curve,
    historical_positive_query_ids,
)
from evaluate_kitti_loop_association_ablations import build_rank1_selections
from summarize_kitti_frozen_three_way import safe_divide


DEFAULT_TARGET_PRECISION = 0.99


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate UniPR rank-1 associations with a 02-frozen similarity threshold."
    )
    parser.add_argument("--threshold_file", default=None, help="Reuse a frozen 02 threshold JSON; do not recalibrate.")
    parser.add_argument("--calibration_pairs", default=None)
    parser.add_argument("--calibration_descriptors", default=None)
    parser.add_argument("--calibration_output_dir", default=None)
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--sequence_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_precision", type=float, default=DEFAULT_TARGET_PRECISION)
    parser.add_argument("--positive_radius", type=float, default=5.0)
    parser.add_argument("--min_temporal_gap", type=int, default=100)
    parser.add_argument("--distance_axes", type=int, nargs=2, default=(0, 2))
    return parser.parse_args()


def rank1_rows(pairs):
    ranked = pairs.loc[pairs.candidate_rank.eq(1)].copy()
    if ranked.empty:
        raise RuntimeError("No historical rank-1 rows found.")
    if ranked.duplicated("query_dataset_index").any():
        raise RuntimeError("Duplicate historical rank-1 rows for one query.")
    missing = set(pairs.query_dataset_index.unique()) - set(ranked.query_dataset_index.unique())
    if missing:
        raise RuntimeError(f"{len(missing)} retrieval queries have no rank-1 candidate.")
    return ranked


def calibrate_similarity_threshold(rank1, target_precision):
    if not 0.0 < target_precision <= 1.0:
        raise ValueError("--target_precision must be in (0, 1].")
    best = None
    for threshold in np.sort(rank1.descriptor_similarity.unique())[::-1]:
        accepted = rank1.loc[rank1.descriptor_similarity.ge(threshold)]
        selected_positive = int(accepted.label.eq("positive").sum())
        strict_precision = safe_divide(selected_positive, len(accepted))
        if strict_precision < target_precision:
            continue
        candidate = (
            selected_positive,
            -float(threshold),
            float(threshold),
            {
                "threshold": float(threshold),
                "target_precision": float(target_precision),
                "selected_queries": int(len(accepted)),
                "correct_associations": selected_positive,
                "explicitly_wrong_negative_associations": int(accepted.label.eq("negative").sum()),
                "gray_ignore_associations": int(accepted.label.eq("ignore").sum()),
                "association_precision_strict": strict_precision,
                "association_precision_clear_labels": safe_divide(
                    selected_positive,
                    selected_positive + int(accepted.label.eq("negative").sum()),
                ),
            },
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        raise RuntimeError(
            "No 02 rank-1 similarity threshold meets the requested strict precision."
        )
    return best[3]


def apply_similarity_threshold(rank1_selections, threshold):
    rows = []
    for row in rank1_selections.itertuples(index=False):
        base = {
            "query_dataset_index": int(row.query_dataset_index),
            "query_center_idx": int(row.query_center_idx),
            "retrieved_positive": bool(row.retrieved_positive),
            "score_name": "descriptor_similarity",
            "threshold": float(threshold),
        }
        if row.score >= threshold:
            rows.append(
                {
                    **base,
                    "association_state": "selected",
                    "selected_label": row.selected_label,
                    "selected_pair_row_index": row.selected_pair_row_index,
                    "candidate_dataset_index": int(row.candidate_dataset_index),
                    "candidate_center_idx": int(row.candidate_center_idx),
                    "candidate_rank": int(row.candidate_rank),
                    "gt_distance_m": float(row.gt_distance_m),
                    "score": float(row.score),
                    "raw_logit": float(row.score),
                    "candidate_count": int(row.candidate_count),
                    "cluster_count": int(row.cluster_count),
                    "selected_cluster_size": int(row.selected_cluster_size),
                }
            )
        else:
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
                    "score": float(row.score),
                    "raw_logit": np.nan,
                    "candidate_count": 0,
                    "cluster_count": 0,
                    "selected_cluster_size": 0,
                }
            )
    return pd.DataFrame(rows)


def count_historical_positives(descriptors_path, min_temporal_gap, positive_radius, distance_axes):
    descriptor_data = torch.load(descriptors_path, map_location="cpu", weights_only=False)
    return len(
        historical_positive_query_ids(
            descriptor_data["center_indices"],
            descriptor_data["translations"],
            min_temporal_gap,
            positive_radius,
            distance_axes,
        )
    )


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
        "method": "descriptor_rank1_threshold",
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
            "Ablation of association selection only. The similarity threshold is frozen from "
            "KITTI 02 rank-1 strict precision and is not tuned on the evaluation sequence. "
            "Accept-only reports and previous ablation reports are not modified."
        ),
    }, curve


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


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

    if args.threshold_file:
        with Path(args.threshold_file).open() as handle:
            threshold_stats = json.load(handle)
        if "threshold" not in threshold_stats:
            raise RuntimeError(f"Threshold file is missing 'threshold': {args.threshold_file}")
        calibration_report = {
            "sequence_id": "02",
            "method": "descriptor_rank1_threshold_reuse",
            "inputs": {"threshold_file": str(Path(args.threshold_file))},
            "note": "Reused the frozen 02 threshold; 02 calibration was not rewritten.",
            **threshold_stats,
        }
    else:
        if not args.calibration_pairs or not args.calibration_descriptors or not args.calibration_output_dir:
            raise ValueError("Calibration inputs are required unless --threshold_file is provided.")
        calibration_output_dir = Path(args.calibration_output_dir)
        if calibration_output_dir.exists() and any(calibration_output_dir.iterdir()):
            raise FileExistsError(f"Calibration output directory is not empty: {calibration_output_dir}")
        calibration_output_dir.mkdir(parents=True, exist_ok=True)
        calibration_pairs = pd.read_csv(args.calibration_pairs)
        calibration_rank1 = rank1_rows(calibration_pairs)
        threshold_stats = calibrate_similarity_threshold(calibration_rank1, args.target_precision)
        calibration_historical_positives = count_historical_positives(
            args.calibration_descriptors,
            args.min_temporal_gap,
            args.positive_radius,
            args.distance_axes,
        )
        calibration_report = {
        "sequence_id": "02",
        "method": "descriptor_rank1_threshold_calibration",
        "inputs": {
            "calibration_pairs": str(Path(args.calibration_pairs)),
            "calibration_descriptors": str(Path(args.calibration_descriptors)),
        },
        "protocol": {
            "positive_radius_m": args.positive_radius,
            "minimum_temporal_gap": args.min_temporal_gap,
            "distance_axes": list(args.distance_axes),
            "ground_truth": "center-frame X/Z distance only; yaw and visual overlap are not used",
            "candidate_selection": "historical UniPR rank-1; abstain below 02-frozen similarity threshold",
            "threshold_selection": (
                "lowest rank-1 descriptor similarity on KITTI 02 whose query-level strict "
                f"precision is at least {args.target_precision}; ignore counts as incorrect; "
                "ties prefer more true-positive associations"
            ),
            "target_precision": args.target_precision,
            "target_precision_source": (
                "same default as calibrate_kitti_three_way_decision.py; not tuned on 07/05/06/09"
            ),
            "unique_variable": "add a 02-frozen descriptor-similarity accept threshold to rank-1",
        },
        "retrieval_queries": int(calibration_pairs.query_dataset_index.nunique()),
        "historical_positive_queries": int(calibration_historical_positives),
        "retrieved_positive_queries": int(
            calibration_pairs.loc[calibration_pairs.label.eq("positive"), "query_dataset_index"].nunique()
        ),
        "rank1_label_counts": calibration_rank1.label.value_counts().to_dict(),
        **threshold_stats,
        "association_recall_conditional_retrieval": safe_divide(
            threshold_stats["correct_associations"],
            int(calibration_pairs.loc[calibration_pairs.label.eq("positive"), "query_dataset_index"].nunique()),
        ),
        "association_recall_end_to_end": safe_divide(
            threshold_stats["correct_associations"],
            calibration_historical_positives,
        ),
            "note": "Threshold is frozen from sequence 02 only and must not be retuned on 07.",
        }
        write_json(calibration_output_dir / "threshold.json", threshold_stats)
        write_json(calibration_output_dir / "report.json", calibration_report)

    pairs = pd.read_csv(args.pairs)
    historical_positive_queries = count_historical_positives(
        args.descriptors,
        args.min_temporal_gap,
        args.positive_radius,
        args.distance_axes,
    )
    selections = apply_similarity_threshold(
        build_rank1_selections(pairs),
        threshold_stats["threshold"],
    )
    protocol = {
        "positive_radius_m": args.positive_radius,
        "minimum_temporal_gap": args.min_temporal_gap,
        "distance_axes": list(args.distance_axes),
        "ground_truth": "center-frame X/Z distance only; yaw and visual overlap are not used",
        "candidate_selection": (
            "historical UniPR rank-1; abstain if descriptor_similarity < 02-frozen threshold; "
            "no VGGT geometry, no decision head"
        ),
        "ignore_policy": (
            "strict precision treats ignore as incorrect; clear-label precision excludes ignore"
        ),
        "descriptor_similarity_threshold": threshold_stats["threshold"],
        "target_precision": args.target_precision,
        "unique_variable": "add a 02-frozen descriptor-similarity accept threshold to rank-1",
    }
    inputs = {
        "threshold_file": None if args.threshold_file is None else str(Path(args.threshold_file)),
        "calibration_pairs": None if args.calibration_pairs is None else str(Path(args.calibration_pairs)),
        "calibration_descriptors": None if args.calibration_descriptors is None else str(Path(args.calibration_descriptors)),
        "calibration_output_dir": None if args.calibration_output_dir is None else str(Path(args.calibration_output_dir)),
        "descriptors": str(Path(args.descriptors)),
        "pairs": str(Path(args.pairs)),
        "mode": "descriptor_rank1_threshold",
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
    write_json(output_dir / "report.json", report)
    print(json.dumps({"calibration": calibration_report, "evaluation": report}, indent=2))


if __name__ == "__main__":
    main()
