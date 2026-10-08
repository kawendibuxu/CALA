#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=/data1/jiaming/UniPR-3D-main
OUT_ROOT=/data3/xiaoke/UniPR-3D-main-loop
DATA_ROOT=/data3/xiaoke/UniPR-3D-main-loop/data-kitti-08/dataset
cd "$CODE_ROOT"

DESCRIPTORS="$OUT_ROOT/outputs/kitti_08_seq5_stride1_unipr_descriptors.pt"
PAIRS="$OUT_ROOT/outputs/kitti_08_seq5_stride1_unipr_descriptors_top50_pairs.csv"
EVALUATION_DIR="$OUT_ROOT/outputs/kitti_08_full_three_way_eval"
ASSOCIATION_DIR="$OUT_ROOT/outputs/kitti_08_loop_association_eval"
FROZEN_DECISION="$CODE_ROOT/outputs/kitti_cross_token_3d_full_three_way_calibration_02/frozen_three_way_decision.pt"
PYTHON=/data1/jiaming/.conda/envs/py311/bin/python

check_hash() {
    local expected="$1"
    local path="$2"
    printf '%s  %s\n' "$expected" "$path" | sha256sum --check --status -
}

check_hash "2b7d00a4227b8e17361694d5b6df56c6e6f553880c1d55a81d190e3c7b59f738" "$CODE_ROOT/model/multi_model.ckpt"
check_hash "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0" "$CODE_ROOT/model/VGGT-model/model.pt"
check_hash "14db2d7f37de45cdf1e31ca550ce436c06dc9c8563308d3c98680628db3d4a08" "$FROZEN_DECISION"
check_hash "3289d017ff71932ff74e8d006691f71b2a370acef934d736ec95779e2caa48b5" "$CODE_ROOT/extract_kitti_descriptors.py"
check_hash "bdd4f5a45b0786e0ba1c86329a2ef6b5555a906ecf0b54d21477532e463773ce" "$CODE_ROOT/build_kitti_topk_pairs.py"
check_hash "a79b2869eceebd00815113a6676536cd286ef746ac7f96465f384d0dc608190b" "$CODE_ROOT/evaluate_kitti_frozen_three_way.py"
check_hash "de718016b062c6066aad3d0141949bf2c16b166eea4978ea49cf25f10ffbc450" "$CODE_ROOT/summarize_kitti_frozen_three_way.py"
check_hash "e2c121ee69b2568d3ec2f2a9b5b36d0b9066ad3886ba5fe47869d768e9da970a" "$CODE_ROOT/evaluate_kitti_loop_association.py"
check_hash "cd7177170c7d7ba98cdbfe9417f97bd9586da5c70cbd5ccefa5db6bf88a5fe88" "$DATA_ROOT/poses/08.txt"
check_hash "4abedb688d60249c36353b9893570f993ebfdc5ef7938f52f3a7b9e1edbfffb1" "$DATA_ROOT/sequences/08/calib.txt"
check_hash "65359db9033c8716d2150c90c6fabc6f93db640531d3370e3eb7056d82590e9f" "$DATA_ROOT/sequences/08/times.txt"

image_count="$(find "$DATA_ROOT/sequences/08/image_2" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)"
if [[ "$image_count" -ne 4071 ]]; then
    echo "KITTI08_PREFLIGHT_FAILED: expected 4071 image_2 frames, found $image_count"
    exit 1
fi

if [[ -e "$ASSOCIATION_DIR/report.json" ]]; then
    echo "KITTI08_PREFLIGHT_FAILED: association report already exists"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
echo "KITTI08_GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
mkdir -p "$OUT_ROOT/outputs"

if [[ ! -e "$DESCRIPTORS" ]]; then
    "$PYTHON" extract_kitti_descriptors.py \
        --root "$DATA_ROOT" \
        --sequence_id 08 \
        --ckpt model/multi_model.ckpt \
        --output "$DESCRIPTORS" \
        --seq_len 5 \
        --stride 1 \
        --image_size 392 518 \
        --batch_size 4 \
        --num_workers 4 \
        --device cuda
fi

if [[ ! -e "$PAIRS" ]]; then
    "$PYTHON" build_kitti_topk_pairs.py \
        --descriptors "$DESCRIPTORS" \
        --output "$PAIRS" \
        --top_k 50 \
        --min_temporal_gap 100 \
        --positive_radius 5.0 \
        --negative_radius 25.0 \
        --distance_axes 0 2 \
        --block_size 256
fi

"$PYTHON" evaluate_kitti_frozen_three_way.py \
    --descriptors "$DESCRIPTORS" \
    --pairs "$PAIRS" \
    --frozen_decision "$FROZEN_DECISION" \
    --output_dir "$EVALUATION_DIR" \
    --vggt_ckpt model/VGGT-model/model.pt \
    --image_size 392 518 \
    --grid_rows 12 \
    --grid_cols 16 \
    --ransac_threshold_m 0.5 \
    --ransac_iterations 256 \
    --seed 17 \
    --chunk_size 100 \
    --device cuda

historical_positive_queries="$(
DESCRIPTORS_PATH="$DESCRIPTORS" "$PYTHON" - <<'PY'
import os
import torch

data = torch.load(os.environ["DESCRIPTORS_PATH"], map_location="cpu", weights_only=False)
centers = data["center_indices"].long()
translations = data["translations"].float()
count = 0
for query_row, query_center in enumerate(centers):
    historical = centers <= int(query_center) - 100
    distances = torch.linalg.norm(
        translations[:, [0, 2]] - translations[query_row, [0, 2]],
        dim=1,
    )
    count += int(bool((historical & (distances <= 5.0)).any().item()))
print(count)
PY
)"

"$PYTHON" summarize_kitti_frozen_three_way.py \
    --pairs "$PAIRS" \
    --evaluation_dir "$EVALUATION_DIR" \
    --sequence_id 08 \
    --historical_positive_queries "$historical_positive_queries" \
    --output "$EVALUATION_DIR/report.json"

"$PYTHON" evaluate_kitti_loop_association.py \
    --descriptors "$DESCRIPTORS" \
    --pairs "$PAIRS" \
    --evaluation_dir "$EVALUATION_DIR" \
    --frozen_decision "$FROZEN_DECISION" \
    --sequence_id 08 \
    --output_dir "$ASSOCIATION_DIR" \
    --cluster_gap 5 \
    --positive_radius 5.0 \
    --min_temporal_gap 100 \
    --distance_axes 0 2

echo "KITTI08_FROZEN_ACCEPT_ONLY_COMPLETE historical_positive_queries=$historical_positive_queries"
