"""Resumably export frozen-VGGT cross-token 3D features for every top-K pair.

This standalone script deliberately leaves UniPR untouched.  Each chunk is
atomically written so an interrupted multi-day extraction can be resumed.
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from geometry_verification import VGGTGeometryVerifier
from geometry_verification.image_io import load_rgb_sequence


def parse_args():
    parser = argparse.ArgumentParser(description="Extract every KITTI top-K cross-token 3D feature.")
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vggt_ckpt", default="model/VGGT-model/model.pt")
    parser.add_argument("--image_size", nargs=2, type=int, default=(392, 518), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--grid_rows", type=int, default=12)
    parser.add_argument("--grid_cols", type=int, default=16)
    parser.add_argument("--ransac_threshold_m", type=float, default=0.5)
    parser.add_argument("--ransac_iterations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=("cuda", "cpu"))
    return parser.parse_args()


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk_size must be positive.")
    output_dir = Path(args.output_dir)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.pairs)
    descriptor_data = torch.load(args.descriptors, map_location="cpu", weights_only=False)
    image_paths = descriptor_data["image_paths"]
    verifier = VGGTGeometryVerifier(args.vggt_ckpt, device=args.device, image_size=tuple(args.image_size))

    num_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
    completed = 0
    for chunk_index in tqdm(range(num_chunks), desc="Cross-token 3D chunks"):
        chunk_path = chunks_dir / f"chunk_{chunk_index:05d}.csv"
        if chunk_path.exists():
            continue
        if args.max_chunks is not None and completed >= args.max_chunks:
            break
        start = chunk_index * args.chunk_size
        output_rows = []
        for local_index, row in enumerate(tqdm(rows[start:start + args.chunk_size], desc=f"Chunk {chunk_index}", leave=False)):
            query_sequence = load_rgb_sequence(image_paths[int(row["query_dataset_index"])], args.image_size)
            candidate_sequence = load_rgb_sequence(image_paths[int(row["candidate_dataset_index"])], args.image_size)
            result = verifier.verify_cross_token_3d_pair(
                query_sequence, candidate_sequence, grid_rows=args.grid_rows, grid_cols=args.grid_cols,
                ransac_threshold_m=args.ransac_threshold_m, ransac_iterations=args.ransac_iterations,
                seed=args.seed + start + local_index,
            )
            score = asdict(result.score)
            output_rows.append({
                "pair_row_index": start + local_index,
                **row,
                **{name: f"{value:.8f}" if isinstance(value, float) else value for name, value in score.items()},
            })
        temporary_path = chunk_path.with_suffix(".tmp")
        with temporary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
        temporary_path.replace(chunk_path)
        completed += 1
        print(f"Completed chunk {chunk_index + 1}/{num_chunks}: {chunk_path}")
    print(f"Completed {completed} new chunks; output directory: {output_dir}")


if __name__ == "__main__":
    main()
