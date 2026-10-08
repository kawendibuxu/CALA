"""Train extra 7-d logistic seeds and summarize mean ± std.

Unique variable: 00 training seed. Seed 17 is the official frozen head and is
only read. Seeds 18 and 19 are trained and calibrated into a new output tree.
VGGT chunks, official 0.947 reports and existing ablation dirs are not written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path("/data1/jiaming/UniPR-3D-main")
OFFICIAL_HEAD = CODE_ROOT / "outputs/kitti_cross_token_3d_full_decision_head_00/decision_head.pt"
OFFICIAL_FROZEN = (
    CODE_ROOT / "outputs/kitti_cross_token_3d_full_three_way_calibration_02/frozen_three_way_decision.pt"
)
OFFICIAL_REPORTS = {
    "05": CODE_ROOT / "outputs/kitti_05_loop_association_eval/report.json",
    "06": CODE_ROOT / "outputs/kitti_06_loop_association_eval/report.json",
    "07": CODE_ROOT / "outputs/kitti_07_loop_association_eval/report.json",
}
FEATURE_00 = CODE_ROOT / "outputs/kitti_00_top50_cross_token_3d_full"
PAIRS_00 = CODE_ROOT / "outputs/kitti_00_seq5_stride1_unipr_descriptors_top50_pairs.csv"
CAL_02_DIR = CODE_ROOT / "outputs/kitti_02_top50_cross_token_3d_full"
PAIRS_02 = CODE_ROOT / "outputs/kitti_02_seq5_stride1_unipr_descriptors_top50_pairs.csv"
DEFAULT_OUT = Path("/data3/xiaoke/UniPR-3D-main-loop/outputs/kitti_seed_ablation")
CANDIDATE_SEEDS = tuple(range(18, 51))
TARGET_NEW_SEEDS = 2
SEQUENCES = ("07", "05", "06")
TARGET_PRECISION = 0.99
CLIP_LOGIT = math.log((1.0 - 1e-6) / 1e-6)
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
    parser = argparse.ArgumentParser(description="Run the 00 logistic seed study without touching official outputs.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUT))
    parser.add_argument("--python", default="/data1/jiaming/.conda/envs/py311/bin/python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default="2", help="Physical GPU id; mapped to cuda:0 inside the process.")
    return parser.parse_args()


def assert_new_output(path):
    resolved = str(Path(path).resolve())
    for token in FORBIDDEN_SUBSTRINGS:
        if token in resolved and "kitti_seed_ablation" not in resolved:
            raise RuntimeError(f"Refusing to write into official output: {resolved}")


def launch_env(gpu):
    env = os.environ.copy()
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def run(python, script, extra, env):
    command = [python, str(CODE_ROOT / script), *extra]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(CODE_ROOT), env=env)


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
    if n == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(var)


def load_json(path):
    return json.loads(Path(path).read_text())


def clip_probability_ceiling(coef, intercept):
    return 1.0 / (1.0 + math.exp(-(coef * CLIP_LOGIT + intercept)))


def official_threshold_eval_reachable(frozen):
    coef = float(np.asarray(frozen["platt_coef"]).reshape(-1)[0])
    intercept = float(np.asarray(frozen["platt_intercept"]).reshape(-1)[0])
    accept = float(frozen["rules"]["accept_probability"])
    ceiling = clip_probability_ceiling(coef, intercept)
    return ceiling > accept + 1e-8, ceiling, accept


def transferred(seed_dir):
    for sequence_id in SEQUENCES:
        report_path = seed_dir / f"kitti_{sequence_id}_eval" / "report.json"
        if not report_path.exists():
            return False
        if load_json(report_path)["queries_with_selected_association"] <= 0:
            return False
    return True


def summarize(output_root):
    per_seed = {}
    official_frozen = torch.load(OFFICIAL_FROZEN, map_location="cpu", weights_only=False)
    per_seed[17] = {
        "status": "official_frozen_accept_only",
        "accept_probability": float(official_frozen["rules"]["accept_probability"]),
        "sequences": {},
    }
    for sequence_id, path in OFFICIAL_REPORTS.items():
        per_seed[17]["sequences"][sequence_id] = metrics_from_report(load_json(path))
        per_seed[17]["sequences"][sequence_id]["report"] = str(path.resolve())

    infeasible = []
    degenerate = []
    successful = [17]
    for seed_dir in sorted(output_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed = int(seed_dir.name.split("_")[1])
        fail_path = seed_dir / "calibration_infeasible.json"
        degenerate_path = seed_dir / "calibration_degenerate.json"
        frozen_path = seed_dir / "calibration_02" / "frozen_three_way_decision.pt"
        if fail_path.exists():
            per_seed[seed] = load_json(fail_path)
            infeasible.append(seed)
            continue
        if degenerate_path.exists():
            per_seed[seed] = load_json(degenerate_path)
            degenerate.append(seed)
            continue
        if not frozen_path.exists() or not transferred(seed_dir):
            continue
        frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
        reachable, ceiling, accept = official_threshold_eval_reachable(frozen)
        if not reachable:
            continue
        per_seed[seed] = {
            "status": "official_0.99_calibrate_and_eval",
            "accept_probability": accept,
            "eval_clip_probability_ceiling": ceiling,
            "sequences": {},
        }
        for sequence_id in SEQUENCES:
            report_path = seed_dir / f"kitti_{sequence_id}_eval" / "report.json"
            report = load_json(report_path)
            per_seed[seed]["sequences"][sequence_id] = metrics_from_report(report)
            per_seed[seed]["sequences"][sequence_id]["report"] = str(report_path.resolve())
        successful.append(seed)

    aggregates = {}
    for sequence_id in SEQUENCES:
        keys = (
            "association_precision_strict",
            "association_recall_end_to_end",
            "positive",
            "ignore",
            "negative",
            "queries_with_selected_association",
        )
        aggregates[sequence_id] = {}
        for key in keys:
            values = [per_seed[seed]["sequences"][sequence_id][key] for seed in successful]
            mean, std = mean_std(values)
            aggregates[sequence_id][key] = {
                "mean": mean,
                "std": std,
                "values": values,
                "seeds": successful,
            }
    summary = {
        "unique_variable": "00 logistic training seed",
        "held_fixed": [
            "frozen UniPR descriptors and top-50 pairs",
            "frozen VGGT 7-d features",
            "official 00 training recipe except seed",
            "02-only Platt and accept-threshold selection at target_precision=0.99",
            "accept-only association protocol",
        ],
        "seed_17": "official frozen accept-only reports; not retrained",
        "successful_seeds": successful,
        "infeasible_seeds": infeasible,
        "degenerate_seeds": degenerate,
        "per_seed": {str(seed): per_seed[seed] for seed in sorted(per_seed)},
        "mean_std_over_feasible_official_protocol": aggregates,
        "note": (
            "A seed enters the mean only if official calibrate_kitti_three_way_decision.py "
            "finds a 0.99 point and that threshold is reachable under the eval clip formula "
            "with nonzero 05/06/07 accepts. Do not replace the frozen baseline with this mean."
        ),
    }
    path = output_root / "seed_summary_099_reselection.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return path


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    assert_new_output(output_root)
    if "kitti_seed_ablation" not in str(output_root):
        raise RuntimeError("Output root must contain kitti_seed_ablation.")
    output_root.mkdir(parents=True, exist_ok=True)
    python = args.python
    env = launch_env(args.gpu)
    protocol_path = output_root / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "unique_variable": "00 logistic training seed",
                "official_seed": 17,
                "candidate_new_seeds": list(CANDIDATE_SEEDS),
                "target_new_feasible_seeds": TARGET_NEW_SEEDS,
                "target_precision": TARGET_PRECISION,
                "device": args.device,
                "gpu": env.get("CUDA_VISIBLE_DEVICES"),
                "acceptance_rule": (
                    "Official calibrate target_precision=0.99 must succeed, the selected "
                    "threshold must be below the eval 1e-6 clip ceiling, and 05/06/07 must "
                    "all have nonzero accepts. Mean is official seed 17 plus the first two "
                    "new seeds that satisfy this exact official rule."
                ),
                "held_fixed": [
                    "frozen UniPR descriptors and top-50 pairs",
                    "frozen VGGT 7-d features",
                    "official 00 training recipe except seed",
                    "02-only Platt and accept-threshold selection",
                    "accept-only association protocol",
                ],
                "do_not_write": list(FORBIDDEN_SUBSTRINGS),
            },
            indent=2,
        )
        + "\n"
    )

    feasible_new = []
    for seed in CANDIDATE_SEEDS:
        if len(feasible_new) >= TARGET_NEW_SEEDS:
            break
        seed_dir = output_root / f"seed_{seed}"
        head_dir = seed_dir / "head_00"
        cal_dir = seed_dir / "calibration_02"
        fail_path = seed_dir / "calibration_infeasible.json"
        degenerate_path = seed_dir / "calibration_degenerate.json"
        diag_path = output_root / "diagnostics" / f"seed_{seed}_02.json"
        assert_new_output(head_dir)
        assert_new_output(cal_dir)
        if fail_path.exists() or degenerate_path.exists():
            continue
        if not (head_dir / "decision_head.pt").exists():
            run(
                python,
                "train_kitti_full_decision_head.py",
                [
                    "--feature_dir",
                    str(FEATURE_00),
                    "--pairs",
                    str(PAIRS_00),
                    "--output_dir",
                    str(head_dir),
                    "--head",
                    "logistic",
                    "--epochs",
                    "50",
                    "--batch_size",
                    "64",
                    "--positive_per_batch",
                    "16",
                    "--regular_negative_per_rank",
                    "300",
                    "--learning_rate",
                    "1e-2",
                    "--weight_decay",
                    "1e-3",
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ],
                env,
            )
        if not diag_path.exists():
            run(
                python,
                "diagnose_kitti_three_way_calibration.py",
                [
                    "--head_checkpoint",
                    str(head_dir / "decision_head.pt"),
                    "--calibration_dir",
                    str(CAL_02_DIR),
                    "--full_pairs",
                    str(PAIRS_02),
                    "--target_precision",
                    str(TARGET_PRECISION),
                    "--output_json",
                    str(diag_path),
                ],
                env,
            )
        diagnosis = load_json(diag_path)
        if diagnosis["n_feasible_thresholds"] == 0:
            if not fail_path.exists():
                fail_path.write_text(
                    json.dumps(
                        {
                            "status": "infeasible_under_official_0.99",
                            "training_seed": seed,
                            "diagnosis": str(diag_path.resolve()),
                            "max_weighted_precision_under_strong_gate": diagnosis[
                                "max_weighted_precision_under_strong_gate"
                            ],
                            "weighted_average_precision": diagnosis["weighted_average_precision"],
                            "note": (
                                "Official calibrate_kitti_three_way_decision.py would raise "
                                "because no 02 threshold meets weighted precision 0.99. "
                                "This seed is recorded and skipped; 05/06/07 are not evaluated."
                            ),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            continue
        if not (cal_dir / "frozen_three_way_decision.pt").exists():
            run(
                python,
                "calibrate_kitti_three_way_decision.py",
                [
                    "--head_checkpoint",
                    str(head_dir / "decision_head.pt"),
                    "--calibration_dir",
                    str(CAL_02_DIR),
                    "--full_pairs",
                    str(PAIRS_02),
                    "--output_dir",
                    str(cal_dir),
                    "--target_precision",
                    str(TARGET_PRECISION),
                    "--reject_probability",
                    "0.05",
                ],
                env,
            )
        frozen = torch.load(cal_dir / "frozen_three_way_decision.pt", map_location="cpu", weights_only=False)
        reachable, ceiling, accept = official_threshold_eval_reachable(frozen)
        if not reachable:
            if not degenerate_path.exists():
                degenerate_path.write_text(
                    json.dumps(
                        {
                            "status": "degenerate_official_0.99_threshold_above_eval_clip",
                            "training_seed": seed,
                            "accept_probability": accept,
                            "eval_clip_probability_ceiling": ceiling,
                            "note": (
                                "Official 0.99 calibration succeeded, but the selected "
                                "threshold exceeds the eval 1e-6 clip ceiling, so accept "
                                "count is identically zero. Excluded from the official-rule mean."
                            ),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            continue
        for sequence_id in SEQUENCES:
            eval_dir = seed_dir / f"kitti_{sequence_id}_eval"
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
                    str(cal_dir / "frozen_three_way_decision.pt"),
                    "--sequence_id",
                    sequence_id,
                    "--output_dir",
                    str(eval_dir),
                    "--seed",
                    str(seed),
                ],
                env,
            )
        if not transferred(seed_dir):
            if not degenerate_path.exists():
                degenerate_path.write_text(
                    json.dumps(
                        {
                            "status": "degenerate_official_0.99_zero_test_accepts",
                            "training_seed": seed,
                            "accept_probability": accept,
                            "eval_clip_probability_ceiling": ceiling,
                            "note": "Official 0.99 calibration succeeded but 05/06/07 accepts are zero.",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            continue
        feasible_new.append(seed)

    if len(feasible_new) < TARGET_NEW_SEEDS:
        raise RuntimeError(
            f"Only {len(feasible_new)} new seeds admitted the official 0.99 02 point; "
            f"needed {TARGET_NEW_SEEDS}."
        )
    summarize(output_root)
    if not OFFICIAL_HEAD.exists() or not OFFICIAL_FROZEN.exists():
        raise RuntimeError("Official seed-17 artefacts missing; they must remain untouched.")


if __name__ == "__main__":
    main()
