"""Run frozen 3D decisions on every retrieved KITTI top-K candidate."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from geometry_verification import VGGTGeometryVerifier
from geometry_verification.image_io import load_rgb_sequence
from train_kitti_decision_head import LogisticDecisionHead, MLPDecisionHead, Standardizer


def parse_args():
    parser = argparse.ArgumentParser(description="Apply a frozen three-way decision to every KITTI top-K pair.")
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--frozen_decision", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vggt_ckpt", default="model/VGGT-model/model.pt")
    parser.add_argument("--image_size", nargs=2, type=int, default=(392, 518), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--grid_rows", type=int, default=12)
    parser.add_argument("--grid_cols", type=int, default=16)
    parser.add_argument("--ransac_threshold_m", type=float, default=0.5)
    parser.add_argument("--ransac_iterations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--max_chunks", type=int, default=None, help="Optional cap for a scheduled run; resume later.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=("cuda", "cpu"))
    return parser.parse_args()


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_frozen_head(path):
    frozen = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = frozen["head_checkpoint"]
    features = checkpoint["feature_names"]
    standardizer = Standardizer(**checkpoint["standardizer"])
    model = LogisticDecisionHead(len(features)) if checkpoint["head"] == "logistic" else MLPDecisionHead(len(features), checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, features, standardizer, float(frozen["platt_coef"].reshape(-1)[0]), float(frozen["platt_intercept"].reshape(-1)[0]), frozen["rules"]


def classify(score, rules):
    strong = (
        score["mean_track_confidence"] >= rules["track_confidence_accept_min"]
        and score["mean_visibility"] >= rules["visibility_accept_min"]
        and score["median_3d_residual"] <= rules["median_3d_residual_accept_max"]
    )
    if score["calibrated_probability"] >= rules["accept_probability"] and strong:
        return "accept"
    weak = (
        score["mean_track_confidence"] <= rules["track_confidence_reject_max"]
        and score["mean_visibility"] <= rules["visibility_reject_max"]
    )
    if score["calibrated_probability"] <= rules["reject_probability"] and weak:
        return "reject"
    return "uncertain"


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
    model, features, standardizer, platt_coef, platt_intercept, rules = load_frozen_head(args.frozen_decision)
    verifier = VGGTGeometryVerifier(args.vggt_ckpt, device=args.device, image_size=tuple(args.image_size))

    num_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
    completed = 0
    for chunk_index in tqdm(range(num_chunks), desc="Frozen evaluation chunks"):
        chunk_path = chunks_dir / f"chunk_{chunk_index:05d}.csv"
        if chunk_path.exists():
            continue
        if args.max_chunks is not None and completed >= args.max_chunks:
            break
        start = chunk_index * args.chunk_size
        chunk_rows = rows[start:start + args.chunk_size]
        output_rows = []
        for local_index, row in enumerate(tqdm(chunk_rows, desc=f"Chunk {chunk_index}", leave=False)):
            query_sequence = load_rgb_sequence(image_paths[int(row["query_dataset_index"])], args.image_size)
            candidate_sequence = load_rgb_sequence(image_paths[int(row["candidate_dataset_index"])], args.image_size)
            result = verifier.verify_cross_token_3d_pair(
                query_sequence, candidate_sequence, grid_rows=args.grid_rows, grid_cols=args.grid_cols,
                ransac_threshold_m=args.ransac_threshold_m, ransac_iterations=args.ransac_iterations,
                seed=args.seed + start + local_index,
            )
            score = asdict(result.score)
            values = np.asarray([[float(score[name]) if name in score else float(row[name]) for name in features]], dtype=np.float32)
            with torch.no_grad():
                raw_logit = float(model(torch.from_numpy(standardizer.transform(values))).item())
            raw_score = 1.0 / (1.0 + np.exp(-raw_logit))
            clipped_logit = np.log(np.clip(raw_score, 1e-6, 1 - 1e-6) / np.clip(1 - raw_score, 1e-6, 1))
            score["raw_score"] = float(raw_score)
            score["calibrated_probability"] = float(1.0 / (1.0 + np.exp(-(platt_coef * clipped_logit + platt_intercept))))
            score["state"] = classify(score, rules)
            output_rows.append({"pair_row_index": start + local_index, **row, **{name: f"{value:.8f}" if isinstance(value, float) else value for name, value in score.items()}})
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
