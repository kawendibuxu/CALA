"""Aggregate a completed frozen three-way KITTI evaluation without retuning it."""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize chunks from frozen three-way KITTI inference.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--evaluation_dir", required=True)
    parser.add_argument("--historical_positive_queries", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    args = parse_args()
    all_pairs = pd.read_csv(args.pairs)
    files = sorted((Path(args.evaluation_dir) / "chunks").glob("chunk_*.csv"))
    if not files:
        raise FileNotFoundError("No completed chunk CSVs found.")
    result = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if result.pair_row_index.nunique() != len(all_pairs) or set(result.pair_row_index) != set(range(len(all_pairs))):
        raise RuntimeError(f"Evaluation incomplete: found {result.pair_row_index.nunique()} / {len(all_pairs)} pairs.")
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
        "pairs": len(result), "queries": int(result.query_dataset_index.nunique()),
        "historical_positive_queries": args.historical_positive_queries,
        "retrieved_positive_queries": len(positive_query_ids),
        "descriptor_recall_at_50": safe_divide(len(positive_query_ids), args.historical_positive_queries),
        "state_counts": result.state.value_counts().to_dict(),
        "state_by_label": pd.crosstab(result.state, result.label).to_dict(),
        "accept_pair_precision": safe_divide(accepted_tp, accepted_tp + accepted_fp),
        "accept_pair_recall_conditional_top50": safe_divide(accepted_tp, int(positive.sum())),
        "accept_query_recall_conditional_top50": safe_divide(accept_positive_queries, len(positive_query_ids)),
        "accept_query_recall_end_to_end": safe_divide(accept_positive_queries, args.historical_positive_queries),
        "reranked_recall_at_1_end_to_end": safe_divide(top1_hits, args.historical_positive_queries),
        "false_positives_per_query": safe_divide(accepted_fp, int(result.query_dataset_index.nunique())),
        "uncertain_positive_pairs": int((uncertain & positive).sum()),
        "uncertain_negative_pairs": int((uncertain & negative).sum()),
        "reject_positive_pairs": int((rejected & positive).sum()),
        "reject_negative_pairs": int((rejected & negative).sum()),
        "note": "All model, calibration, geometry, and three-way parameters are frozen from sequence 00/02 before this sequence-05 evaluation.",
    }
    with Path(args.output).open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
