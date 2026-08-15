"""Export frozen-VGGT explicit 3D cross-token features for retrieved KITTI pairs."""

import argparse
import csv
import random
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from geometry_verification import VGGTGeometryVerifier
from geometry_verification.image_io import load_rgb_sequence


def parse_args():
    parser = argparse.ArgumentParser(description="Extract explicit 3D cross-token KITTI features.")
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--selection_csv", default=None, help="Optional preselected pair CSV; bypasses default sampling.")
    parser.add_argument("--vggt_ckpt", default="model/VGGT-model/model.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image_size", nargs=2, type=int, default=(392, 518), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--max_positive", type=int, default=100)
    parser.add_argument("--max_negative", type=int, default=500)
    parser.add_argument("--max_ignore", type=int, default=0)
    parser.add_argument("--hard_negative_rank", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_rows", type=int, default=12)
    parser.add_argument("--grid_cols", type=int, default=16)
    parser.add_argument("--ransac_threshold_m", type=float, default=0.5)
    parser.add_argument("--ransac_iterations", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=("cuda", "cpu"))
    return parser.parse_args()


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_rows(rows, max_positive, max_negative, max_ignore, hard_negative_rank, seed):
    rng = random.Random(seed)
    groups = {
        "positive": [row for row in rows if row["label"] == "positive"],
        "negative": [
            row for row in rows
            if row["label"] == "negative" and int(row["candidate_rank"]) <= hard_negative_rank
        ],
        "ignore": [row for row in rows if row["label"] == "ignore"],
    }
    for group in groups.values():
        rng.shuffle(group)
    selected = (
        groups["positive"][:max_positive]
        + groups["negative"][:max_negative]
        + groups["ignore"][:max_ignore]
    )
    rng.shuffle(selected)
    return selected


def main():
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_data = torch.load(args.descriptors, map_location="cpu", weights_only=False)
    image_paths = descriptor_data["image_paths"]
    selected_rows = (
        read_rows(args.selection_csv)
        if args.selection_csv
        else select_rows(
            read_rows(args.pairs), args.max_positive, args.max_negative, args.max_ignore,
            args.hard_negative_rank, args.seed,
        )
    )
    verifier = VGGTGeometryVerifier(args.vggt_ckpt, device=args.device, image_size=tuple(args.image_size))
    output_rows = []
    for index, row in enumerate(tqdm(selected_rows, desc="Extracting cross-token 3D features")):
        query_sequence = load_rgb_sequence(image_paths[int(row["query_dataset_index"])], args.image_size)
        candidate_sequence = load_rgb_sequence(image_paths[int(row["candidate_dataset_index"])], args.image_size)
        result = verifier.verify_cross_token_3d_pair(
            query_sequence,
            candidate_sequence,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            ransac_threshold_m=args.ransac_threshold_m,
            ransac_iterations=args.ransac_iterations,
            seed=args.seed + index,
        )
        score = asdict(result.score)
        output_rows.append({
            **row,
            **{name: f"{value:.8f}" if isinstance(value, float) else value for name, value in score.items()},
        })
    if not output_rows:
        raise ValueError("No rows selected. Increase a --max_* argument.")
    with output_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    counts = {}
    for row in output_rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"Saved cross-token 3D features to {output_path}")
    print(f"Rows: {len(output_rows)}")
    print(f"Label counts: {counts}")


if __name__ == "__main__":
    main()
