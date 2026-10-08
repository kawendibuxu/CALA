"""Aggregate a completed frozen three-way KITTI evaluation without retuning it."""

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_RESULT_COLUMNS = {
    "pair_row_index",
    "query_dataset_index",
    "label",
    "state",
    "calibrated_probability",
}
VALID_LABELS = {"positive", "negative", "ignore"}
VALID_STATES = {"accept", "uncertain", "reject"}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize chunks from frozen three-way KITTI inference.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--evaluation_dir", required=True)
    parser.add_argument("--sequence_id", required=True, help="Evaluated KITTI sequence ID, for example 05 or 06.")
    parser.add_argument("--historical_positive_queries", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def load_evaluation(evaluation_dir):
    files = sorted((Path(evaluation_dir) / "chunks").glob("chunk_*.csv"))
    if not files:
        raise FileNotFoundError("No completed chunk CSVs found.")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def validate_evaluation(result, expected_pairs):
    missing_columns = REQUIRED_RESULT_COLUMNS - set(result.columns)
    if missing_columns:
        raise RuntimeError(f"Evaluation is missing required columns: {sorted(missing_columns)}.")
    if len(result) != expected_pairs:
        raise RuntimeError(f"Evaluation row count mismatch: found {len(result)} / {expected_pairs} rows.")
    if result.pair_row_index.duplicated().any():
        duplicates = int(result.pair_row_index.duplicated().sum())
        raise RuntimeError(f"Evaluation contains {duplicates} duplicate pair_row_index values.")
    expected_indices = set(range(expected_pairs))
    actual_indices = set(result.pair_row_index)
    if actual_indices != expected_indices:
        missing = len(expected_indices - actual_indices)
        unexpected = len(actual_indices - expected_indices)
        raise RuntimeError(
            f"Evaluation pair_row_index coverage mismatch: {missing} missing and {unexpected} unexpected indices."
        )
    invalid_labels = set(result.label.dropna().unique()) - VALID_LABELS
    if result.label.isna().any() or invalid_labels:
        raise RuntimeError(f"Evaluation contains invalid labels: {sorted(invalid_labels)}.")
    invalid_states = set(result.state.dropna().unique()) - VALID_STATES
    if result.state.isna().any() or invalid_states:
        raise RuntimeError(f"Evaluation contains invalid states: {sorted(invalid_states)}.")
    probabilities = pd.to_numeric(result.calibrated_probability, errors="coerce")
    if probabilities.isna().any() or not probabilities.between(0.0, 1.0).all():
        raise RuntimeError("Evaluation contains invalid calibrated_probability values.")
    result["calibrated_probability"] = probabilities


def build_report(result, historical_positive_queries, sequence_id, pairs_path, evaluation_dir):
    if historical_positive_queries < 1:
        raise ValueError("historical_positive_queries must be positive.")
    positive = result.label.eq("positive")
    negative = result.label.eq("negative")
    accepted = result.state.eq("accept")
    uncertain = result.state.eq("uncertain")
    rejected = result.state.eq("reject")
    accepted_tp = int((accepted & positive).sum())
    accepted_fp = int((accepted & negative).sum())
    groups = result.groupby("query_dataset_index", sort=False)
    positive_query_ids = set(result.loc[positive, "query_dataset_index"])
    accept_positive_queries = sum(group.loc[group.state.eq("accept"), "label"].eq("positive").any() for _, group in groups)
    top1_hits = sum(group.loc[group.calibrated_probability.idxmax(), "label"] == "positive" for _, group in groups)
    report = {
        "sequence_id": str(sequence_id),
        "inputs": {
            "pairs": str(Path(pairs_path)),
            "evaluation_dir": str(Path(evaluation_dir)),
        },
        "pairs": len(result), "queries": int(result.query_dataset_index.nunique()),
        "historical_positive_queries": historical_positive_queries,
        "retrieved_positive_queries": len(positive_query_ids),
        "descriptor_recall_at_50": safe_divide(len(positive_query_ids), historical_positive_queries),
        "state_counts": result.state.value_counts().to_dict(),
        "state_by_label": pd.crosstab(result.state, result.label).to_dict(),
        "accept_pair_precision": safe_divide(accepted_tp, accepted_tp + accepted_fp),
        "accept_pair_recall_conditional_top50": safe_divide(accepted_tp, int(positive.sum())),
        "accept_query_recall_conditional_top50": safe_divide(accept_positive_queries, len(positive_query_ids)),
        "accept_query_recall_end_to_end": safe_divide(accept_positive_queries, historical_positive_queries),
        "reranked_recall_at_1_end_to_end": safe_divide(top1_hits, historical_positive_queries),
        "false_positives_per_query": safe_divide(accepted_fp, int(result.query_dataset_index.nunique())),
        "uncertain_positive_pairs": int((uncertain & positive).sum()),
        "uncertain_negative_pairs": int((uncertain & negative).sum()),
        "reject_positive_pairs": int((rejected & positive).sum()),
        "reject_negative_pairs": int((rejected & negative).sum()),
        "note": (
            "All model, calibration, geometry, and three-way parameters are frozen from sequence 00/02 "
            f"before this sequence-{sequence_id} evaluation."
        ),
    }
    return report


def main():
    args = parse_args()
    all_pairs = pd.read_csv(args.pairs)
    result = load_evaluation(args.evaluation_dir)
    validate_evaluation(result, len(all_pairs))
    report = build_report(
        result=result,
        historical_positive_queries=args.historical_positive_queries,
        sequence_id=args.sequence_id,
        pairs_path=args.pairs,
        evaluation_dir=args.evaluation_dir,
    )
    with Path(args.output).open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
