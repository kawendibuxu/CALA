import argparse
import csv
import random
from pathlib import Path

import torch
from tqdm import tqdm

from geometry_verification import VGGTGeometryVerifier
from geometry_verification.image_io import load_rgb_sequence


def parse_args():
    parser = argparse.ArgumentParser(description="Extract VGGT geometry features for KITTI top-K pairs.")
    parser.add_argument("--descriptors", required=True, help="Descriptor .pt from extract_kitti_descriptors.py.")
    parser.add_argument("--pairs", required=True, help="Top-K pair CSV from build_kitti_topk_pairs.py.")
    parser.add_argument("--vggt_ckpt", default="model/VGGT-model/model.pt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--image_size", nargs=2, type=int, default=(392, 518), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--max_positive", type=int, default=100)
    parser.add_argument("--max_negative", type=int, default=500)
    parser.add_argument("--max_ignore", type=int, default=0)
    parser.add_argument("--hard_negative_rank", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_rows", type=int, default=12)
    parser.add_argument("--grid_cols", type=int, default=16)
    parser.add_argument("--ransac_reproj_threshold", type=float, default=1.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    return parser.parse_args()


def default_output_path(pairs_path, max_positive, max_negative):
    pairs_path = Path(pairs_path)
    return pairs_path.with_name(
        pairs_path.stem + f"_geom_pos{max_positive}_neg{max_negative}.csv"
    )


def read_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def select_rows(rows, max_positive, max_negative, max_ignore, hard_negative_rank, seed):
    rng = random.Random(seed)
    positives = [row for row in rows if row["label"] == "positive"]
    negatives = [
        row
        for row in rows
        if row["label"] == "negative" and int(row["candidate_rank"]) <= hard_negative_rank
    ]
    ignores = [row for row in rows if row["label"] == "ignore"]

    rng.shuffle(positives)
    rng.shuffle(negatives)
    rng.shuffle(ignores)

    selected = (
        positives[:max_positive]
        + negatives[:max_negative]
        + ignores[:max_ignore]
    )
    rng.shuffle(selected)
    return selected


def main():
    args = parse_args()
    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(args.pairs, args.max_positive, args.max_negative)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor_data = torch.load(args.descriptors, map_location="cpu", weights_only=False)
    image_paths = descriptor_data["image_paths"]
    rows = read_rows(args.pairs)
    selected_rows = select_rows(
        rows,
        args.max_positive,
        args.max_negative,
        args.max_ignore,
        args.hard_negative_rank,
        args.seed,
    )

    verifier = VGGTGeometryVerifier(
        checkpoint_path=args.vggt_ckpt,
        device=args.device,
        image_size=tuple(args.image_size),
    )

    output_rows = []
    for row in tqdm(selected_rows, desc="Extracting geometry features"):
        query_idx = int(row["query_dataset_index"])
        candidate_idx = int(row["candidate_dataset_index"])
        query_seq = load_rgb_sequence(image_paths[query_idx], args.image_size)
        candidate_seq = load_rgb_sequence(image_paths[candidate_idx], args.image_size)

        result = verifier.verify_pair(
            query_seq,
            candidate_seq,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            ransac_reproj_threshold=args.ransac_reproj_threshold,
        )
        score = result.score
        output_rows.append(
            {
                **row,
                "num_matches": score.num_matches,
                "num_inliers": score.num_inliers,
                "inlier_ratio": f"{score.inlier_ratio:.8f}",
                "mean_sampson_error": f"{score.mean_sampson_error:.8f}",
                "mean_track_confidence": f"{score.mean_track_confidence:.8f}",
                "mean_visibility": f"{score.mean_visibility:.8f}",
                "accepted_by_geometry": int(score.accepted),
            }
        )

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved geometry features to {output_path}")
    print(f"Rows: {len(output_rows)}")
    if output_rows:
        counts = {}
        accepted = {}
        for row in output_rows:
            label = row["label"]
            counts[label] = counts.get(label, 0) + 1
            accepted[label] = accepted.get(label, 0) + int(row["accepted_by_geometry"])
        print(f"Label counts: {counts}")
        print(f"Accepted counts: {accepted}")


if __name__ == "__main__":
    main()
