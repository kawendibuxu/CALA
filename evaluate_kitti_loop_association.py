"""Evaluate one deterministic historical loop-window association per KITTI query.

This script is read-only with respect to frozen model outputs. It consumes the
completed three-way evaluation chunks, selects at most one accepted historical
window per query, and reports association-level precision and recall.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from summarize_kitti_frozen_three_way import load_evaluation, safe_divide, validate_evaluation
from train_kitti_decision_head import LogisticDecisionHead, MLPDecisionHead, Standardizer


REQUIRED_ASSOCIATION_COLUMNS = {
    "candidate_center_idx",
    "candidate_dataset_index",
    "candidate_rank",
    "gt_distance_m",
    "mean_track_confidence",
    "mean_visibility",
    "median_3d_residual",
    "query_center_idx",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select and evaluate one frozen loop-window association per KITTI query."
    )
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--evaluation_dir", required=True)
    parser.add_argument("--frozen_decision", required=True)
    parser.add_argument("--sequence_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--cluster_gap",
        type=int,
        default=5,
        help="Maximum center-frame gap joining accepted historical windows into one temporal cluster.",
    )
    parser.add_argument("--positive_radius", type=float, default=5.0)
    parser.add_argument("--min_temporal_gap", type=int, default=100)
    parser.add_argument("--distance_axes", type=int, nargs=2, default=(0, 2))
    return parser.parse_args()


def load_frozen_scoring_head(path):
    frozen = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = frozen["head_checkpoint"]
    features = checkpoint["feature_names"]
    standardizer = Standardizer(**checkpoint["standardizer"])
    if checkpoint["head"] == "logistic":
        model = LogisticDecisionHead(len(features))
    else:
        model = MLPDecisionHead(len(features), checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, features, standardizer, frozen["rules"], frozen


def add_raw_logits(result, model, features, standardizer):
    missing = set(features) - set(result.columns)
    if missing:
        raise RuntimeError(f"Evaluation is missing decision features: {sorted(missing)}.")
    values = result[features].to_numpy(dtype=np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(standardizer.transform(values))).numpy()
    result = result.copy()
    result["raw_logit"] = logits.astype(np.float64)
    return result


def add_temporal_clusters(candidates, cluster_gap):
    if cluster_gap < 0:
        raise ValueError("cluster_gap must be non-negative.")
    ordered = candidates.sort_values(
        ["candidate_center_idx", "candidate_rank", "pair_row_index"],
        kind="stable",
    ).copy()
    gaps = ordered["candidate_center_idx"].diff()
    ordered["temporal_cluster_id"] = gaps.gt(cluster_gap).fillna(False).cumsum().astype(int)
    return ordered


def select_one_candidate(candidates, cluster_gap):
    if candidates.empty:
        return None
    clustered = add_temporal_clusters(candidates, cluster_gap)
    ordered = clustered.sort_values(
        ["raw_logit", "candidate_rank", "candidate_center_idx", "pair_row_index"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    representatives = ordered.drop_duplicates("temporal_cluster_id", keep="first")
    selected = representatives.iloc[0].copy()
    selected["candidate_count"] = len(clustered)
    selected["cluster_count"] = int(clustered.temporal_cluster_id.nunique())
    selected["selected_cluster_size"] = int(
        clustered.temporal_cluster_id.eq(selected.temporal_cluster_id).sum()
    )
    return selected


def historical_positive_query_ids(
    center_indices,
    translations,
    min_temporal_gap,
    positive_radius,
    distance_axes,
):
    center_indices = np.asarray(center_indices)
    translations = np.asarray(translations)
    axes = np.asarray(distance_axes, dtype=int)
    if len(center_indices) != len(translations):
        raise RuntimeError("Descriptor center_indices and translations have different lengths.")
    positive_queries = set()
    for query_index, query_center in enumerate(center_indices):
        eligible = np.flatnonzero(center_indices <= query_center - min_temporal_gap)
        if not len(eligible):
            continue
        differences = translations[eligible][:, axes] - translations[query_index, axes]
        distances = np.linalg.norm(differences, axis=1)
        if np.any(distances <= positive_radius):
            positive_queries.add(query_index)
    return positive_queries


def build_selections(result, cluster_gap):
    rows = []
    for query_index, group in result.groupby("query_dataset_index", sort=True):
        accepted = group.loc[group.state.eq("accept")]
        selected = select_one_candidate(accepted, cluster_gap)
        base = {
            "query_dataset_index": int(query_index),
            "query_center_idx": int(group.query_center_idx.iloc[0]),
            "retrieved_positive": bool(group.label.eq("positive").any()),
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


def build_potential_predictions(result, rules, cluster_gap):
    strong = (
        result.mean_track_confidence.ge(rules["track_confidence_accept_min"])
        & result.mean_visibility.ge(rules["visibility_accept_min"])
        & result.median_3d_residual.le(rules["median_3d_residual_accept_max"])
    )
    rows = []
    for query_index, group in result.loc[strong].groupby("query_dataset_index", sort=True):
        selected = select_one_candidate(group, cluster_gap)
        rows.append(
            {
                "query_dataset_index": int(query_index),
                "raw_logit": float(selected.raw_logit),
                "label": selected.label,
            }
        )
    return pd.DataFrame(rows)


def association_precision_recall_curve(predictions, historical_positive_queries):
    if historical_positive_queries < 1:
        raise ValueError("historical_positive_queries must be positive.")
    if predictions.empty:
        return pd.DataFrame(
            columns=["raw_logit_threshold", "true_positives", "false_positives", "precision", "recall"]
        ), 0.0, 0.0
    ordered = predictions.sort_values(
        ["raw_logit", "query_dataset_index"],
        ascending=[False, True],
        kind="stable",
    ).copy()
    ordered["is_correct"] = ordered.label.eq("positive").astype(int)
    ordered["true_positives"] = ordered.is_correct.cumsum()
    ordered["false_positives"] = (~ordered.label.eq("positive")).astype(int).cumsum()
    ordered["precision"] = ordered.true_positives / (
        ordered.true_positives + ordered.false_positives
    )
    ordered["recall"] = ordered.true_positives / historical_positive_queries
    average_precision = float(
        ordered.loc[ordered.is_correct.eq(1), "precision"].sum() / historical_positive_queries
    )
    zero_false_positive = ordered.loc[ordered.false_positives.eq(0)]
    recall_at_100_precision = (
        float(zero_false_positive.recall.max()) if len(zero_false_positive) else 0.0
    )
    curve = ordered[
        ["raw_logit", "true_positives", "false_positives", "precision", "recall"]
    ].rename(columns={"raw_logit": "raw_logit_threshold"})
    return curve, average_precision, recall_at_100_precision


def frozen_raw_logit_threshold(frozen):
    probability = float(frozen["rules"]["accept_probability"])
    coefficient = float(np.asarray(frozen["platt_coef"]).reshape(-1)[0])
    intercept = float(np.asarray(frozen["platt_intercept"]).reshape(-1)[0])
    probability_logit = np.log(probability / (1.0 - probability))
    return float((probability_logit - intercept) / coefficient)


def build_report(
    result,
    selections,
    potential_predictions,
    historical_positive_queries,
    sequence_id,
    inputs,
    protocol,
    frozen,
):
    selected = selections.loc[selections.association_state.eq("selected")]
    selected_counts = selected.selected_label.value_counts().to_dict()
    selected_positive = int(selected.selected_label.eq("positive").sum())
    selected_negative = int(selected.selected_label.eq("negative").sum())
    selected_ignore = int(selected.selected_label.eq("ignore").sum())
    retrieved_positive_queries = int(result.loc[result.label.eq("positive"), "query_dataset_index"].nunique())
    curve, average_precision, recall_at_100_precision = association_precision_recall_curve(
        potential_predictions,
        historical_positive_queries,
    )
    report = {
        "sequence_id": str(sequence_id),
        "inputs": inputs,
        "protocol": {
            **protocol,
            "candidate_selection": (
                "accept-only; temporal clusters; highest uncalibrated raw logit; "
                "ties by lower candidate rank, center index, then pair_row_index"
            ),
            "ignore_policy": "reported separately; strict precision treats ignore as incorrect",
            "frozen_accept_raw_logit_threshold": frozen_raw_logit_threshold(frozen),
        },
        "retrieval_queries": int(result.query_dataset_index.nunique()),
        "historical_positive_queries": int(historical_positive_queries),
        "retrieved_positive_queries": retrieved_positive_queries,
        "retrieval_missed_positive_queries": int(
            historical_positive_queries - retrieved_positive_queries
        ),
        "queries_with_selected_association": int(len(selected)),
        "queries_abstained": int(selections.association_state.eq("abstain").sum()),
        "selected_label_counts": selected_counts,
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
        "abstention_rate_retrieval_queries": safe_divide(
            int(selections.association_state.eq("abstain").sum()),
            len(selections),
        ),
        "association_average_precision": average_precision,
        "association_recall_at_100_precision": recall_at_100_precision,
        "pr_curve_points": len(curve),
        "note": (
            "This report evaluates one concrete historical window per query. "
            "It does not retune the frozen sequence-00/02 model or rules."
        ),
    }
    return report, curve


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

    model, features, standardizer, rules, frozen = load_frozen_scoring_head(
        args.frozen_decision
    )
    result = add_raw_logits(result, model, features, standardizer)

    descriptor_data = torch.load(args.descriptors, map_location="cpu", weights_only=False)
    historical_positive_ids = historical_positive_query_ids(
        descriptor_data["center_indices"],
        descriptor_data["translations"],
        args.min_temporal_gap,
        args.positive_radius,
        args.distance_axes,
    )
    selections = build_selections(result, args.cluster_gap)
    potential_predictions = build_potential_predictions(result, rules, args.cluster_gap)
    protocol = {
        "positive_radius_m": args.positive_radius,
        "minimum_temporal_gap": args.min_temporal_gap,
        "distance_axes": list(args.distance_axes),
        "temporal_cluster_gap_frames": args.cluster_gap,
        "ground_truth": "center-frame X/Z distance only; yaw and visual overlap are not used",
    }
    inputs = {
        "descriptors": str(Path(args.descriptors)),
        "pairs": str(Path(args.pairs)),
        "evaluation_dir": str(Path(args.evaluation_dir)),
        "frozen_decision": str(Path(args.frozen_decision)),
    }
    report, curve = build_report(
        result=result,
        selections=selections,
        potential_predictions=potential_predictions,
        historical_positive_queries=len(historical_positive_ids),
        sequence_id=args.sequence_id,
        inputs=inputs,
        protocol=protocol,
        frozen=frozen,
    )

    selections.to_csv(output_dir / "selected_associations.csv", index=False)
    curve.to_csv(output_dir / "association_precision_recall_curve.csv", index=False)
    with (output_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
