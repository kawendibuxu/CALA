import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from kitti_loop.dataset import loop_label


def parse_args():
    parser = argparse.ArgumentParser(description="Build KITTI top-K loop candidate pairs from descriptors.")
    parser.add_argument("--descriptors", required=True, help="Path produced by extract_kitti_descriptors.py.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--min_temporal_gap", type=int, default=100)
    parser.add_argument("--positive_radius", type=float, default=5.0)
    parser.add_argument("--negative_radius", type=float, default=25.0)
    parser.add_argument("--distance_axes", nargs=2, type=int, default=(0, 2))
    parser.add_argument("--block_size", type=int, default=256)
    return parser.parse_args()


def default_output_path(descriptor_path, top_k):
    descriptor_path = Path(descriptor_path)
    return descriptor_path.with_name(descriptor_path.stem + f"_top{top_k}_pairs.csv")


def main():
    args = parse_args()
    descriptor_path = Path(args.descriptors)
    output_path = Path(args.output) if args.output else default_output_path(descriptor_path, args.top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = torch.load(descriptor_path, map_location="cpu", weights_only=False)
    descriptors = F.normalize(data["descriptors"].float(), dim=1)
    centers = data["center_indices"].long()
    translations = data["translations"].float()
    sequence_id = data["sequence_id"]

    rows = []
    recall_hits = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    recall_denominator = 0
    label_counts = {"positive": 0, "negative": 0, "ignore": 0}

    for start in tqdm(range(0, len(descriptors), args.block_size), desc="Building top-K pairs"):
        end = min(start + args.block_size, len(descriptors))
        query_desc = descriptors[start:end]
        similarities = query_desc @ descriptors.T

        for local_idx, query_row in enumerate(range(start, end)):
            query_center = int(centers[query_row])
            historical_mask = centers <= query_center - args.min_temporal_gap
            historical_mask[query_row] = False

            if not historical_mask.any():
                continue

            query_translation = translations[query_row]
            diffs = translations[:, list(args.distance_axes)] - query_translation[list(args.distance_axes)]
            distances = torch.linalg.norm(diffs, dim=1)

            historical_positive_mask = historical_mask & (distances <= args.positive_radius)
            has_historical_positive = bool(historical_positive_mask.any().item())
            if has_historical_positive:
                recall_denominator += 1

            candidate_scores = similarities[local_idx].clone()
            candidate_scores[~historical_mask] = -float("inf")
            k = min(args.top_k, int(historical_mask.sum().item()))
            scores, indices = torch.topk(candidate_scores, k=k, largest=True)

            top_candidate_indices = indices.tolist()
            if has_historical_positive:
                for recall_k in recall_hits:
                    if any(bool(historical_positive_mask[idx].item()) for idx in top_candidate_indices[:recall_k]):
                        recall_hits[recall_k] += 1

            for rank, (candidate_row_tensor, score_tensor) in enumerate(zip(indices, scores), start=1):
                candidate_row = int(candidate_row_tensor)
                candidate_center = int(centers[candidate_row])
                temporal_gap = abs(query_center - candidate_center)
                distance_m = float(distances[candidate_row].item())
                label = loop_label(
                    distance_m,
                    temporal_gap,
                    positive_radius=args.positive_radius,
                    negative_radius=args.negative_radius,
                    min_temporal_gap=args.min_temporal_gap,
                )
                label_counts[label] += 1
                rows.append(
                    {
                        "sequence_id": sequence_id,
                        "query_dataset_index": query_row,
                        "candidate_dataset_index": candidate_row,
                        "query_center_idx": query_center,
                        "candidate_center_idx": candidate_center,
                        "temporal_gap": temporal_gap,
                        "gt_distance_m": f"{distance_m:.6f}",
                        "label": label,
                        "candidate_rank": rank,
                        "descriptor_similarity": f"{float(score_tensor.item()):.8f}",
                    }
                )

    fieldnames = [
        "sequence_id",
        "query_dataset_index",
        "candidate_dataset_index",
        "query_center_idx",
        "candidate_center_idx",
        "temporal_gap",
        "gt_distance_m",
        "label",
        "candidate_rank",
        "descriptor_similarity",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved top-K pairs to {output_path}")
    print(f"Rows: {len(rows)}")
    print(f"Label counts: {label_counts}")
    print(f"Queries with at least one historical positive: {recall_denominator}")
    if recall_denominator > 0:
        for recall_k in sorted(recall_hits):
            if recall_k <= args.top_k:
                recall = recall_hits[recall_k] / recall_denominator
                print(f"Descriptor-only Recall@{recall_k}: {recall:.4f}")


if __name__ == "__main__":
    main()
