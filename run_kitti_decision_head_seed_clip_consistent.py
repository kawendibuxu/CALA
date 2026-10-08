"""Evaluate predeclared seeds 18/19 with clip-consistent 02 thresholds."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import torch


CODE_ROOT = Path("/data1/jiaming/UniPR-3D-main")
OFFICIAL_FROZEN = (
    CODE_ROOT / "outputs/kitti_cross_token_3d_full_three_way_calibration_02/frozen_three_way_decision.pt"
)
OFFICIAL_REPORTS = {
    "05": CODE_ROOT / "outputs/kitti_05_loop_association_eval/report.json",
    "06": CODE_ROOT / "outputs/kitti_06_loop_association_eval/report.json",
    "07": CODE_ROOT / "outputs/kitti_07_loop_association_eval/report.json",
}
CAL_02_DIR = CODE_ROOT / "outputs/kitti_02_top50_cross_token_3d_full"
PAIRS_02 = CODE_ROOT / "outputs/kitti_02_seq5_stride1_unipr_descriptors_top50_pairs.csv"
OUTPUT_ROOT = Path("/data3/xiaoke/UniPR-3D-main-loop/outputs/kitti_seed_ablation")
PYTHON = "/data1/jiaming/.conda/envs/py311/bin/python"
SEEDS = (18, 19)
SEQUENCES = ("07", "05", "06")


def run(script, extra):
    command = [PYTHON, str(CODE_ROOT / script), *extra]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(CODE_ROOT))


def metrics_from_report(report):
    counts = report.get("selected_label_counts", {})
    return {
        "queries_with_selected_association": report["queries_with_selected_association"],
        "queries_abstained": report["queries_abstained"],
        "positive": int(counts.get("positive", report.get("correct_associations", 0))),
        "ignore": int(counts.get("ignore", report.get("gray_ignore_associations", 0))),
        "negative": int(counts.get("negative", report.get("explicitly_wrong_negative_associations", 0))),
        "association_precision_strict": report["association_precision_strict"],
        "association_recall_end_to_end": report["association_recall_end_to_end"],
        "explicitly_wrong_negative_associations": report["explicitly_wrong_negative_associations"],
    }


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(var)


def summarize():
    official_frozen = torch.load(OFFICIAL_FROZEN, map_location="cpu", weights_only=False)
    per_seed = {
        17: {
            "status": "official_frozen_accept_only",
            "accept_probability": float(official_frozen["rules"]["accept_probability"]),
            "selection_rule": "official_max_recall_at_weighted_precision_0.99",
            "sequences": {},
        }
    }
    for sequence_id, path in OFFICIAL_REPORTS.items():
        per_seed[17]["sequences"][sequence_id] = metrics_from_report(json.loads(path.read_text()))
        per_seed[17]["sequences"][sequence_id]["report"] = str(path.resolve())

    for seed in SEEDS:
        freeze_report = json.loads((OUTPUT_ROOT / f"seed_{seed}" / "clip_consistent_02" / "report.json").read_text())
        per_seed[seed] = {
            "status": "new_head_clip_consistent_02",
            "accept_probability": freeze_report["rules"]["accept_probability"],
            "selection_rule": freeze_report["selection_rule"],
            "selected_02_point": freeze_report["selected_02_point"],
            "sequences": {},
        }
        for sequence_id in SEQUENCES:
            report_path = OUTPUT_ROOT / f"seed_{seed}" / f"kitti_{sequence_id}_clip_consistent_eval" / "report.json"
            report = json.loads(report_path.read_text())
            per_seed[seed]["sequences"][sequence_id] = metrics_from_report(report)
            per_seed[seed]["sequences"][sequence_id]["report"] = str(report_path.resolve())

    seeds = (17, *SEEDS)
    aggregates = {}
    for sequence_id in SEQUENCES:
        aggregates[sequence_id] = {}
        for key in (
            "association_precision_strict",
            "association_recall_end_to_end",
            "positive",
            "ignore",
            "negative",
            "queries_with_selected_association",
        ):
            values = [per_seed[seed]["sequences"][sequence_id][key] for seed in seeds]
            mean, std = mean_std(values)
            aggregates[sequence_id][key] = {"mean": mean, "std": std, "values": values, "seeds": list(seeds)}
    summary = {
        "unique_variable": "00 logistic training seed",
        "held_fixed": [
            "frozen UniPR descriptors and top-50 pairs",
            "frozen VGGT 7-d features",
            "official 00 training recipe except seed",
            "02-only threshold selection with eval-consistent Platt probabilities",
            "accept-only association protocol",
        ],
        "successful_seeds": list(seeds),
        "per_seed": {str(seed): per_seed[seed] for seed in seeds},
        "mean_std": aggregates,
        "note": (
            "Seed 17 is the official accept-only report. Seeds 18/19 use clip-consistent "
            "02 selection so the accept threshold is reachable at eval. Do not replace "
            "the frozen baseline with this mean."
        ),
    }
    path = OUTPUT_ROOT / "seed_summary_clip_consistent.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    if "kitti_seed_ablation" not in str(OUTPUT_ROOT.resolve()):
        raise RuntimeError("Refusing to write outside kitti_seed_ablation.")
    for seed in SEEDS:
        head = OUTPUT_ROOT / f"seed_{seed}" / "head_00" / "decision_head.pt"
        freeze_dir = OUTPUT_ROOT / f"seed_{seed}" / "clip_consistent_02"
        if not (freeze_dir / "frozen_three_way_decision.pt").exists():
            run(
                "freeze_kitti_seed_head_clip_consistent.py",
                [
                    "--head_checkpoint",
                    str(head),
                    "--calibration_dir",
                    str(CAL_02_DIR),
                    "--full_pairs",
                    str(PAIRS_02),
                    "--output_dir",
                    str(freeze_dir),
                    "--seed",
                    str(seed),
                ],
            )
        for sequence_id in SEQUENCES:
            eval_dir = OUTPUT_ROOT / f"seed_{seed}" / f"kitti_{sequence_id}_clip_consistent_eval"
            if (eval_dir / "report.json").exists():
                continue
            run(
                "evaluate_kitti_loop_association_rescored_head.py",
                [
                    "--descriptors",
                    str(CODE_ROOT / f"outputs/kitti_{sequence_id}_seq5_stride1_unipr_descriptors.pt"),
                    "--pairs",
                    str(CODE_ROOT / f"outputs/kitti_{sequence_id}_seq5_stride1_unipr_descriptors_top50_pairs.csv"),
                    "--evaluation_dir",
                    str(CODE_ROOT / f"outputs/kitti_{sequence_id}_full_three_way_eval"),
                    "--frozen_decision",
                    str(freeze_dir / "frozen_three_way_decision.pt"),
                    "--sequence_id",
                    sequence_id,
                    "--output_dir",
                    str(eval_dir),
                    "--seed",
                    str(seed),
                ],
            )
    summarize()


if __name__ == "__main__":
    main()
