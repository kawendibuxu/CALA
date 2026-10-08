"""Benchmark the frozen CALA inference stages without changing model outputs.

The script measures one five-frame UniPR descriptor, historical retrieval,
one frozen VGGT pair verification, and the lightweight calibrated decision
head.  It also reports a clearly marked serial Top-50 latency estimate because
the current extraction pipeline verifies candidates one by one.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from geometry_verification import VGGTGeometryVerifier
from geometry_verification.image_io import load_rgb_sequence
from run_multiframe_sample import build_model_from_checkpoint, load_sequence
from train_kitti_decision_head import LogisticDecisionHead, MLPDecisionHead, Standardizer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark frozen CALA inference stages.")
    parser.add_argument(
        "--descriptors",
        default="outputs/kitti_06_seq5_stride1_unipr_descriptors.pt",
    )
    parser.add_argument(
        "--pairs",
        default="outputs/kitti_06_seq5_stride1_unipr_descriptors_top50_pairs.csv",
    )
    parser.add_argument(
        "--feature_chunk",
        default="outputs/kitti_06_full_three_way_eval/chunks/chunk_00335.csv",
    )
    parser.add_argument("--unipr_ckpt", default="model/multi_model.ckpt")
    parser.add_argument("--vggt_ckpt", default="model/VGGT-model/model.pt")
    parser.add_argument(
        "--frozen_decision",
        default="outputs/kitti_cross_token_3d_full_three_way_calibration_02/"
        "frozen_three_way_decision.pt",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, nargs=2, default=(392, 518))
    parser.add_argument("--unipr_warmup", type=int, default=5)
    parser.add_argument("--unipr_repeats", type=int, default=30)
    parser.add_argument("--vggt_warmup", type=int, default=2)
    parser.add_argument("--vggt_repeats", type=int, default=10)
    parser.add_argument("--cpu_repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def summarize_ms(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "repeats": int(len(values)),
    }


def cuda_measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    return summarize_ms(samples), float(peak_mib)


def cpu_measure(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return summarize_ms(samples)


def load_pair_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_decision(path):
    frozen = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = frozen["head_checkpoint"]
    feature_names = checkpoint["feature_names"]
    standardizer = Standardizer(**checkpoint["standardizer"])
    if checkpoint["head"] == "logistic":
        model = LogisticDecisionHead(len(feature_names))
    else:
        model = MLPDecisionHead(len(feature_names), checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, feature_names, standardizer, frozen


def main():
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an available CUDA device.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    descriptor_data = torch.load(args.descriptors, map_location="cpu", weights_only=False)
    image_paths = descriptor_data["image_paths"]
    # Use one fixed query and its historical Rank-1 candidate. Runtime does not
    # depend on the pair label, while fixed inputs keep the benchmark repeatable.
    pair_rows = load_pair_rows(args.pairs)
    benchmark_pair = pair_rows[len(pair_rows) // 2]
    query_index = int(benchmark_pair["query_dataset_index"])
    candidate_index = int(benchmark_pair["candidate_dataset_index"])

    metadata = {
        "protocol": {
            "device": torch.cuda.get_device_name(0),
            "process_cuda_device": 0,
            "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "batch_size": 1,
            "image_size": list(args.image_size),
            "sequence_length": 5,
            "timing": "CUDA events with synchronize; model loading and disk I/O excluded",
            "top50_estimate": "serial estimate using 50 times measured one-pair VGGT latency",
            "seed": args.seed,
        },
        "benchmark_pair": {
            "sequence_id": benchmark_pair["sequence_id"],
            "query_center_idx": int(benchmark_pair["query_center_idx"]),
            "candidate_center_idx": int(benchmark_pair["candidate_center_idx"]),
            "candidate_rank": int(benchmark_pair["candidate_rank"]),
        },
    }

    # UniPR descriptor inference.
    unipr_input = load_sequence(image_paths[query_index], args.image_size).to(args.device)
    load_start = time.perf_counter()
    unipr = build_model_from_checkpoint(args.unipr_ckpt).eval().to(args.device)
    torch.cuda.synchronize()
    metadata["unipr_model_load_s"] = time.perf_counter() - load_start

    def run_unipr():
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            return unipr(unipr_input)["salad_pred"]

    unipr_stats, unipr_peak = cuda_measure(
        run_unipr, args.unipr_warmup, args.unipr_repeats
    )
    unipr_stats["peak_allocated_mib"] = unipr_peak
    metadata["unipr_descriptor"] = unipr_stats
    del unipr, unipr_input
    torch.cuda.empty_cache()

    # Retrieval uses the cached normalized historical descriptor database.
    descriptors = F.normalize(descriptor_data["descriptors"].float(), dim=1)
    centers = descriptor_data["center_indices"].long()
    query_desc = descriptors[query_index]
    history_mask = centers <= centers[query_index] - 100
    history = descriptors[history_mask]

    def retrieve_top1():
        return torch.topk(history @ query_desc, k=1)

    def retrieve_top50():
        return torch.topk(history @ query_desc, k=min(50, len(history)))

    metadata["retrieval_top1_cpu"] = cpu_measure(retrieve_top1, args.cpu_repeats)
    metadata["retrieval_top50_cpu"] = cpu_measure(retrieve_top50, args.cpu_repeats)

    # Frozen VGGT pair verification.
    query_sequence = load_rgb_sequence(image_paths[query_index], args.image_size)
    candidate_sequence = load_rgb_sequence(image_paths[candidate_index], args.image_size)
    load_start = time.perf_counter()
    verifier = VGGTGeometryVerifier(
        args.vggt_ckpt, device=args.device, image_size=tuple(args.image_size)
    )
    torch.cuda.synchronize()
    metadata["vggt_model_load_s"] = time.perf_counter() - load_start
    call_index = 0

    def run_vggt():
        nonlocal call_index
        result = verifier.verify_cross_token_3d_pair(
            query_sequence,
            candidate_sequence,
            grid_rows=12,
            grid_cols=16,
            ransac_threshold_m=0.5,
            ransac_iterations=256,
            seed=args.seed + call_index,
        )
        call_index += 1
        return result

    vggt_stats, vggt_peak = cuda_measure(
        run_vggt, args.vggt_warmup, args.vggt_repeats
    )
    vggt_stats["peak_allocated_mib"] = vggt_peak
    metadata["vggt_pair_verification"] = vggt_stats
    del verifier
    torch.cuda.empty_cache()

    # Lightweight head + Platt + frozen accept gates on a real 50-pair block.
    rows = load_pair_rows(args.feature_chunk)[:50]
    head, feature_names, standardizer, frozen = load_decision(args.frozen_decision)
    values = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    standardized = torch.from_numpy(standardizer.transform(values))
    coef = float(np.asarray(frozen["platt_coef"]).reshape(-1)[0])
    intercept = float(np.asarray(frozen["platt_intercept"]).reshape(-1)[0])
    rules = frozen["rules"]
    track_index = feature_names.index("mean_track_confidence")
    vis_index = feature_names.index("mean_visibility")
    residual_index = feature_names.index("median_3d_residual")

    def score_and_gate():
        with torch.inference_mode():
            logits = head(standardized).numpy()
            probabilities = 1.0 / (1.0 + np.exp(-(logits * coef + intercept)))
            strong = (
                (values[:, track_index] >= rules["track_confidence_accept_min"])
                & (values[:, vis_index] >= rules["visibility_accept_min"])
                & (values[:, residual_index] <= rules["median_3d_residual_accept_max"])
            )
            accepted = strong & (probabilities >= rules["accept_probability"])
            if accepted.any():
                np.flatnonzero(accepted)[np.argmax(logits[accepted])]

    metadata["decision_50_pairs_cpu"] = cpu_measure(score_and_gate, args.cpu_repeats)

    # Derived deployment-level numbers under the current serial implementation.
    descriptor_ms = metadata["unipr_descriptor"]["mean_ms"]
    top1_ms = metadata["retrieval_top1_cpu"]["mean_ms"]
    top50_ms = metadata["retrieval_top50_cpu"]["mean_ms"]
    pair_ms = metadata["vggt_pair_verification"]["mean_ms"]
    decision_ms = metadata["decision_50_pairs_cpu"]["mean_ms"]
    metadata["derived_serial_latency"] = {
        "descriptor_threshold_ms_per_query": descriptor_ms + top1_ms,
        "cala_top50_ms_per_query": descriptor_ms + top50_ms + 50 * pair_ms + decision_ms,
        "cala_top50_vggt_component_ms": 50 * pair_ms,
        "note": "Derived from measured stage means; excludes image decode, model loading, and disk I/O.",
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved benchmark to {output}")


if __name__ == "__main__":
    main()
