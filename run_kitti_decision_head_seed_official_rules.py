"""Evaluate predeclared seeds 18/19 under official 0.947 rules.

Requires already-trained heads in kitti_seed_ablation/seed_{18,19}/head_00.
Does not retrain, does not rewrite official outputs, and does not use the
brittle 0.99 reselection path.
"""

from __future__ import annotations

import argparse
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
DEFAULT_OUT = Path("/data3/xiaoke/UniPR-3D-main-loop/outputs/kitti_seed_ablation")
SEEDS = (18, 19)
SEQUENCES = ("07", "05", "06")
FORBIDDEN_SUBSTRINGS = (
    "kitti_cross_token_3d_full_decision_head_00",
    "kitti_cross_token_3d_full_three_way_calibration_02",
    "kitti_05_loop_association_eval",
    "kitti_06_loop_association_eval",
    "kitti_07_loop_association_eval",
    "kitti_05_full_three_way_eval",
    "kitti_06_full_three_way_eval",
    "kitti_07_full_three_way_eval",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate seeds 18/19 with official frozen rules.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUT))
    parser.add_argument("--python", default="/data1/jiaming/.conda/envs/py311/bin/python")
    return parser.parse_args()


def assert_new_output(path):
    resolved = str(Path(path).resolve())
    for token in FORBIDDEN_SUBSTRINGS:
        if token in resolved and "kitti_seed_ablation" not in resolved:
            raise RuntimeError(f"Refusing to write into official output: {resolved}")


def run(python, script, extra):
    command = [python, str(CODE_ROOT / script), *extra]
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


def summarize(output_root):
    official_frozen = torch.load(OFFICIAL_FROZEN, map_location="cpu", weights_only=False)
    per_seed = {
        17: {
            "status": "official_frozen_accept_only",
            "accept_probability": float(official_frozen["rules"]["accept_probability"]),
            "sequences": {},
        }
    }
    for sequence_id, path in OFFICIAL_REPORTS.items():
        per_seed[17]["sequences"][sequence_id] = metrics_from_report(json.loads(path.read_text()))
        per_seed[17]["sequences"][sequence_id]["report"] = str(path.resolve())

    for seed in SEEDS:
        frozen_path = output_root / f"seed_{seed}" / "official_rules_02" / "frozen_three_way_decision.pt"
        frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
        per_seed[seed] = {
            "status": "new_head_02_platt_official_rules",
            "accept_probability": float(frozen["rules"]["accept_probability"]),
            "sequences": {},
        }
        for sequence_id in SEQUENCES:
            report_path = output_root / f"seed_{seed}" / f"kitti_{sequence_id}_official_rules_eval" / "report.json"
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
            aggregates[sequence_id][key] = {
                "mean": mean,
                "std": std,
                "values": values,
                "seeds": list(seeds),
            }
    summary = {
        "unique_variable": "00 logistic training seed",
        "operating_point": "official frozen accept_probability and geometry gates",
        "held_fixed": [
            "frozen UniPR descriptors and top-50 pairs",
            "frozen VGGT 7-d features",
            "official 00 training recipe except seed",
            "official accept_probability 0.9470927362616622",
            "official geometry hard gates",
            "accept-only association protocol",
        ],
        "induced_by_seed": "logistic weights, standardizer and 02 weighted Platt",
        "successful_seeds": list(seeds),
        "per_seed": {str(seed): per_seed[seed] for seed in seeds},
        "mean_std": aggregates,
        "note": (
            "Seed 17 is the official accept-only report. Seeds 18/19 reuse the official "
            "0.947 rules and only change the 00 head plus its 02 Platt. The 0.99 "
            "reselection diagnostics are kept separately and are not used here. "
            "Do not replace the frozen baseline with this mean."
        ),
    }
    path = output_root / "seed_summary_official_rules.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return path


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    assert_new_output(output_root)
    if "kitti_seed_ablation" not in str(output_root):
        raise RuntimeError("Output root must contain kitti_seed_ablation.")
    python = args.python
    for seed in SEEDS:
        head = output_root / f"seed_{seed}" / "head_00" / "decision_head.pt"
        if not head.exists():
            raise FileNotFoundError(f"Missing trained head for seed {seed}: {head}")
        freeze_dir = output_root / f"seed_{seed}" / "official_rules_02"
        assert_new_output(freeze_dir)
        if not (freeze_dir / "frozen_three_way_decision.pt").exists():
            run(
                python,
                "freeze_kitti_seed_head_official_rules.py",
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
            eval_dir = output_root / f"seed_{seed}" / f"kitti_{sequence_id}_official_rules_eval"
            assert_new_output(eval_dir)
            if (eval_dir / "report.json").exists():
                continue
            run(
                python,
                "evaluate_kitti_loop_association_rescored_head.py",
                [
                    "--descriptors",
                    str(CODE_ROOT / f"outputs/kitti_{sequence_id}_seq5_stride1_unipr_descriptors.pt"),
                    "--pairs",
                    str(
                        CODE_ROOT
                        / f"outputs/kitti_{sequence_id}_seq5_stride1_unipr_descriptors_top50_pairs.csv"
                    ),
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
    summarize(output_root)


if __name__ == "__main__":
    main()
