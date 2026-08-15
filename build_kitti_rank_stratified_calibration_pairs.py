"""Create a representative, rank-stratified calibration subset of KITTI top-K pairs."""

import argparse
import csv
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Sample rank-stratified KITTI calibration pairs.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def quota(label, rank):
    if label == "positive":
        if rank <= 10:
            return 12
        if rank <= 20:
            return 8
        return 4
    if label == "negative":
        return 25 if rank <= 10 else 20
    if label == "ignore":
        return 4
    return 0


def main():
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    with Path(args.pairs).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Pair CSV is empty.")

    rng = random.Random(args.seed)
    selected = []
    counts = {}
    for label in ("positive", "negative", "ignore"):
        for rank in range(1, 51):
            candidates = [
                row for row in rows if row["label"] == label and int(row["candidate_rank"]) == rank
            ]
            rng.shuffle(candidates)
            chosen = candidates[:quota(label, rank)]
            selected.extend(chosen)
            counts[(label, rank)] = len(chosen)
    rng.shuffle(selected)

    with output_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    label_counts = {}
    for row in selected:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
    print(f"Saved rank-stratified calibration pairs to {output_path}")
    print(f"Rows: {len(selected)}")
    print(f"Label counts: {label_counts}")
    print("Per-rank quotas: positive 12/8/4 for ranks 1-10/11-20/21-50; negative 25/20; ignore 4.")


if __name__ == "__main__":
    main()
