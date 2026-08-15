# UniPR-3D / KITTI Cross-Token 3D Loop Closure 项目交接

更新时间：2026-08-15（Asia/Shanghai，已按当前代码和产物二次复核）  
项目根目录：`/data1/jiaming/UniPR-3D-main`

> 本文以当前工作区中的实际代码、模型文件、实验产物和进程状态为准。无法由当前工作区确认的内容均标记为“待确认”。2026-08-15 已在当前目录初始化新的本地 Git 仓库，用于保存当前代码快照；原上游仓库的 `.git` 历史此前缺失，因此当前 Git 历史不能恢复原项目 commit、branch、remote 或本地改动相对上游的准确 diff。原 remote/基准 commit：待确认。

## 1. 项目目标

原项目 UniPR-3D 是一个基于 VGGT、LoRA 和 SALAD 的视觉地点识别（VPR）项目，支持单帧和多帧全局 descriptor 检索。

当前扩展目标是在不修改原 UniPR 主流程的前提下，实现 KITTI 多帧回环检测流水线：

```text
KITTI 5 帧 RGB 序列
  -> 冻结 UniPR-3D 多帧模型提取全局 descriptor
  -> historical top-50 coarse retrieval
  -> 冻结官方 VGGT 对 query/candidate 的两组 5 帧 RGB 联合推理
  -> track/point/depth/camera 输出构造 cross-token 显式 3D 几何特征
  -> 轻量 decision head
  -> Platt 概率校准
  -> accept / uncertain / reject 三态回环判定
```

实验设计强调：错误回环对 SLAM 的风险高于漏检，因此三态规则以高 precision 和低 false positive 为主要目标。

长期研究方向还包括“轻量 3D token 检索”和更高效的 token bank，但当前代码已经完成的是“UniPR 全局 descriptor 检索 + VGGT cross-token 3D 验证 + 不确定性三态判定”。当前并未实现一个替代 UniPR descriptor 的轻量 3D token coarse retriever。

## 2. 当前整体架构

### 2.1 原 UniPR descriptor 路径

```text
[B, S=5, 3, 392, 518]
  -> VGGTPR_LoRA
  -> VGGT LoRA aggregator
  -> geometry-token SALAD + DINO-token SALAD
  -> concatenate two SALAD outputs
  -> [B, 17152] global descriptor
```

加载入口为 `run_multiframe_sample.py:build_model_from_checkpoint()`，正式 KITTI descriptor 提取入口为 `extract_kitti_descriptors.py`。使用权重 `model/multi_model.ckpt`。

已直接读取该 checkpoint 的 `hyper_parameters`：`with_dinov2_features=True`、`with_camera_pose=False`、`lora_frame_attn=True`、`lora_global_attn=True`、`lora_patch_embed=False`。加载脚本另外固定 `with_geo_features=True`，因此 17152 维 `salad_pred` 是 geometry SALAD 与 DINO SALAD 的拼接。虽然 KITTI descriptor `.pt` 中保存了 pose，当前 UniPR descriptor 前向只调用 `model(images)`，不会把 KITTI pose 输入模型。

### 2.2 KITTI retrieval 与标签

`build_kitti_topk_pairs.py` 对 L2-normalized descriptor 做 cosine similarity 检索，仅允许历史候选：

```text
candidate_center <= query_center - 100
```

这里的 top-50 是“最多 50 个”：序列最前面的 query 可能只有 1-49 个满足时间间隔的历史候选，因此 pair 总数不一定等于 query 数乘以 50。

标签由 KITTI GT pose 的 X/Z 平面距离产生，仅用于训练、校准和评估，不进入正式推理模型：

```text
temporal_gap < 100           -> 不进入 historical retrieval
distance <= 5 m             -> positive
distance >= 25 m            -> negative
5 m < distance < 25 m       -> ignore
```

### 2.3 Frozen VGGT cross-token 3D 验证

对每个 top-50 query/candidate pair：

1. query 5 帧和 candidate 5 帧拼接为 10 帧，联合输入官方 VGGT。
2. query 中心帧上生成固定 `12 x 16 = 192` 个网格 query points。
3. VGGT track head 将这些 token/point 跟踪到 candidate 的 5 帧。
4. 在跟踪位置采样 `world_points`、point/depth confidence、visibility、track confidence。
5. 使用置信度乘积作为权重，执行 weighted 3D RANSAC + Kabsch 刚体配准。
6. 将 token-level、多帧 3D 证据聚合为 pair-level 特征。

主要特征：

- `descriptor_similarity`
- `weighted_3d_inlier_ratio`
- `median_3d_residual`
- `p90_3d_residual`
- `mean_track_confidence`
- `mean_visibility`
- `temporal_consistency`
- 另有 `camera_rotation_consistency_deg`、`camera_translation_consistency`、`frame_support`、point/depth confidence 等被导出，但当前正式 decision head 未使用这些额外字段。

### 2.4 Decision head 与三态规则

正式 head 是 7 维输入的 logistic linear head：

```text
7 features -> median fill + standardization -> Linear(7, 1) -> raw loop score
```

训练序列为 KITTI 00；序列 02 仅做 Platt calibration 和三态阈值选择；序列 05、06 是不调参独立测试。

最终状态规则：

- `accept`：校准概率达到阈值，并满足 track confidence、visibility、median 3D residual 的强几何门控。
- `reject`：校准概率低，并同时满足弱 track/visibility 条件。
- 其他情况为 `uncertain`。

## 3. 当前代码目录结构

```text
UniPR-3D-main/
├── README.md                         # 原项目说明
├── DOCKERFILE                        # 原项目 Docker 环境
├── main_ft.py                        # 原单帧 fine-tune 入口
├── main_lora.py                      # 原单帧 LoRA 训练入口
├── main_lora_multiframe.py           # 原多帧 LoRA 训练入口
├── eval_lora.py                      # 原 VPR 评估入口
├── vpr_vggt_ft.py                    # 原 Lightning fine-tune 模型
├── vpr_vggt_lora.py                  # 原单帧 LoRA Lightning 模型
├── vpr_vggt_lora_multiframe.py       # 原多帧 LoRA Lightning 模型
├── run_multiframe_sample.py          # 从 multi_model.ckpt 跑多帧 descriptor 示例
├── dataloaders/                      # 原 MSLS/GSV/Pittsburgh/Nordland/RobotCar 数据层
├── datasets/                         # 原项目部分 benchmark 元数据/少量数据
├── training/                         # 上游 VGGT 训练代码和配置
├── utils/                            # 原项目工具
├── vggt/                             # 本地 VGGT / VGGT-PR / LoRA / SALAD 实现
├── model/
│   ├── multi_model.ckpt              # UniPR 多帧权重
│   └── VGGT-model/model.pt           # 官方 VGGT-1B geometry 权重
├── kitti_loop/
│   └── dataset.py                    # 新增 KITTI 5 帧窗口、pose、标签逻辑
├── geometry_verification/
│   ├── vggt_verifier.py              # 冻结官方 VGGT 包装器
│   ├── cross_token_3d.py              # 3D token 对齐、RANSAC/Kabsch 与特征
│   ├── scoring.py                    # 早期 2D fundamental/Sampson 基线
│   └── image_io.py                   # RGB 序列加载和网格点生成
├── extract_kitti_descriptors.py      # KITTI -> UniPR descriptor
├── build_kitti_topk_pairs.py         # descriptor historical top-K 与 GT 标签
├── extract_kitti_geometry_features.py# 早期 2D geometry 特征提取
├── extract_kitti_cross_token_3d_features.py # 小样本 3D 特征提取
├── extract_kitti_cross_token_3d_full.py     # 全量、chunk 化、可恢复 3D 特征提取
├── train_kitti_decision_head.py      # 早期 600 对 decision head
├── train_kitti_full_decision_head.py # 正式 00 全量训练
├── build_kitti_rank_stratified_calibration_pairs.py # 早期分层校准子集
├── calibrate_kitti_three_way_decision.py    # Platt + 三态规则冻结
├── evaluate_kitti_frozen_three_way.py       # 冻结模型全量独立测试
├── summarize_kitti_frozen_three_way.py      # 完整 chunks 指标汇总
├── outputs/                          # descriptor、pairs、chunks、模型和报告
└── PROJECT_HANDOFF.md                # 本交接文档
```

## 4. 已经完成的功能

- KITTI odometry 彩色 `image_2` 的中心帧优先 5 帧窗口数据集。
- 读取 KITTI pose 和 calibration；pose translation 用于离线标签。
- 使用原 UniPR 多帧 checkpoint 提取 17152 维 descriptor。
- historical top-50 retrieval、Recall@K 计算和 5m/25m 标签生成。
- 冻结官方 VGGT camera/depth/point/track heads 的联合 10 帧推理。
- 基于固定 query 网格、跨帧 track 和 world point 的 cross-token 3D 验证。
- weighted 3D RANSAC/Kabsch、残差、内点率、跨帧一致性和相机一致性特征。
- 小样本 600 对可行性基线。
- 00 完整 top-50 cross-token 3D 特征提取。
- 固定 1:3 batch 正负比例的正式 logistic decision head 训练。
- 02 完整 top-50 特征提取、全量 Platt 校准和三态规则冻结。
- 05 完整 top-50 的正式不调参独立测试与报告。
- 06 descriptor 与 top-50 retrieval；06 frozen VGGT 全量测试已部分完成并支持续跑。
- 长任务使用每 100 对一个 CSV chunk，并通过临时文件 rename 原子写入；中断后会跳过已完成 chunk。

## 5. 每个主要模块的作用

### 原项目模块

- `vggt/models/vggtpr_lora.py`：LoRA 版 UniPR/VGGT-PR descriptor 网络，输出 `salad_pred`。
- `vggt/heads/salad_head.py`：SALAD 全局聚合。
- `vpr_vggt_lora_multiframe.py`：原多帧 Lightning 训练/验证逻辑。
- `dataloaders/`：原 MSLS、GSV-Cities、Nordland、RobotCar、Pittsburgh 等数据层。
- `main_lora_multiframe.py`：原多帧训练入口，但包含作者机器绝对路径，不能直接用于当前 KITTI 扩展。
- `eval_lora.py`：原 benchmark 评估入口，也包含硬编码 checkpoint 和数据路径。

### KITTI 与新增几何模块

- `kitti_loop/dataset.py`：构造 5 帧样本；默认 frame 顺序为 `[center, center-2, center-1, center+1, center+2]`；加载 `image_2`、pose、calib。
- `extract_kitti_descriptors.py`：冻结 UniPR 前向并保存 descriptor、中心帧、pose、图像路径等。
- `build_kitti_topk_pairs.py`：cosine historical retrieval、GT 标签和 descriptor Recall@K。
- `geometry_verification/vggt_verifier.py`：实例化官方 `VGGT(enable_camera/point/depth/track=True)`，加载 `model.pt` 并冻结推理。
- `geometry_verification/cross_token_3d.py`：3D 对应采样、置信度加权、RANSAC/Kabsch 和 pair feature 汇总。
- `extract_kitti_cross_token_3d_full.py`：对完整 top-50 提特征；每个 chunk 结束才写盘；同路径重复运行可续跑。
- `train_kitti_full_decision_head.py`：检查 00 chunks 完整性；保留全部正例，负例池为所有 rank 1-10 negative 加 rank 11-50 每 rank 300 个；每 batch 16 正 + 48 负。
- `calibrate_kitti_three_way_decision.py`：加载冻结 head，在 02 上做 Platt scaling，按目标 precision 选择 accept threshold，并从正负尾部分位数构造几何门控。它兼容分层子集的 `(rank,label)` population weight；正式 02 使用完整 top-50，因此这些权重实际全部为 1。
- `evaluate_kitti_frozen_three_way.py`：对测试序列完整 top-50 运行 frozen VGGT、head、Platt 和三态分类；标签只随结果写出，不参与推理。
- `summarize_kitti_frozen_three_way.py`：要求所有 pair index 完整后，汇总 pair/query/三态指标。

## 6. 当前开发进度

正式实验划分：

| 序列 | 角色 | 当前状态 |
|---|---|---|
| KITTI 00 | decision head 训练 | 完成 |
| KITTI 02 | 全量校准和三态规则 | 完成 |
| KITTI 05 | 第一独立测试 | 完成 |
| KITTI 06 | 第二独立测试 | 进行中，当前停止在 345/487 chunks |

当前最重要状态（2026-08-15 审计时）：

```text
outputs/kitti_06_full_three_way_eval/chunks/
已完成 345 / 487 chunks
已完整覆盖 pair_row_index 0..34499
剩余 142 chunks（约 14,125 pairs）
当前没有 evaluate_kitti_frozen_three_way.py 进程在运行
```

注意：06 有 997 个进入 historical retrieval 的 query；`487` 是 `ceil(48,625 pairs / 100)` 得到的 chunk 数，不是 query 数。其中 266 个 query 在 GT 中存在历史正例，并且全部在 top-50 中召回了正例。

因此下一位 Agent 首先需要恢复 06 测试，而不是重新训练或重新校准。

## 7. 最近修改过的文件以及修改内容

当前本地 Git 仓库是 2026-08-15 新初始化的代码快照。以下“最近修改”仍根据文件时间和实际代码内容整理，无法提供相对原上游仓库的 Git diff。

### 2026-07-27

- `run_multiframe_sample.py`：增加从 `multi_model.ckpt` 构建模型并对少量多帧 RGB 跑 descriptor 的独立入口。
- `geometry_verification/image_io.py`、`scoring.py`、`vggt_verifier.py`：增加 frozen VGGT 2D track/fundamental geometry 验证原型。
- `run_vggt_geometry_verification_sample.py`：几何验证 smoke/sample 入口。

### 2026-07-28

- `kitti_loop/dataset.py`、`inspect_kitti_sequence_dataset.py`：增加 KITTI 5 帧数据集和检查工具。
- `extract_kitti_descriptors.py`：增加 KITTI UniPR descriptor 提取。
- `build_kitti_topk_pairs.py`：增加 historical top-K、Recall@K 和标签。
- `extract_kitti_geometry_features.py`：增加早期 2D geometry 特征批量提取。
- `train_kitti_decision_head.py`：增加早期 logistic/MLP decision head、标准化和 sampled metrics。

### 2026-07-29

- `geometry_verification/cross_token_3d.py`：增加显式 3D token 对齐、weighted RANSAC/Kabsch 和特征。
- `geometry_verification/vggt_verifier.py`：增加 `verify_cross_token_3d_pair()`。
- `extract_kitti_cross_token_3d_features.py`：增加 100 positive + 500 negative 等小样本提取。
- `build_kitti_rank_stratified_calibration_pairs.py`：增加 02 rank 分层校准子集。

### 2026-07-30

- `evaluate_kitti_frozen_three_way.py`：增加可恢复的完整测试序列三态评估。
- `summarize_kitti_frozen_three_way.py`：增加完整 chunks 汇总。

### 2026-08-01

- `extract_kitti_cross_token_3d_full.py`：增加正式全量、chunk 化、原子落盘和续跑。
- `train_kitti_full_decision_head.py`：增加正式 00 训练协议和完整性检查。
- `calibrate_kitti_three_way_decision.py`：增加读取完整 chunk 目录和完整性检查。

原 UniPR 核心文件的时间主要为 2026-06-20，当前 KITTI/cross-token 扩展没有直接修改上述原始核心训练代码。

## 8. 当前正在解决的问题

1. 完成 KITTI 06 的冻结、不调参第二独立测试，以判断 05 的安全-召回表现能否跨序列泛化。
2. 06 完成后生成 `outputs/kitti_06_full_three_way_eval/report.json`，并与 05 正式结果对比。
3. 当前 VGGT 验证吞吐低：每个 query-candidate pair 都重新联合运行 10 帧 VGGT，同一个 query 对 top-50 被重复计算。
4. 后续研究需要决定是否实现真正的轻量 3D token bank coarse retrieval；当前仍使用 17152 维 UniPR 全局 descriptor 做 coarse retrieval。

## 9. 已知 Bug 和限制

### 明确已知问题

- **原 Git 历史缺失**：当前已初始化新的本地 Git 仓库并保存代码快照，但原 remote、基准 commit、branch 和上游历史仍不可恢复。后续若找回原仓库，需要人工对比并迁移当前提交。
- **前台长任务随终端/服务器断连停止**：已经多次发生。脚本支持 chunk 续跑，但不会自动守护或自动重启。建议使用 `tmux`/`screen`/作业调度器。
- **续跑进度条显示误导**：脚本跳过已存在 chunk 时没有更新 tqdm 计数，外层可能显示 `0/487`，但看到 `Chunk 345` 即表示在正确续跑。
- **descriptor 提取不支持 chunk/续跑**：`extract_kitti_descriptors.py` 完成全部样本后才一次性 `torch.save`；中断会整段重跑。
- **`summarize_kitti_frozen_three_way.py` 的 note 硬编码 sequence-05**：用于 06 汇总时 JSON note 仍会写 “before this sequence-05 evaluation”。指标不受影响，但文案错误。
- **全量测试非常慢**：当前实现逐 pair 重跑 VGGT，05 131,625 pairs 约两天；06 48,625 pairs 约十几小时到一天，实际取决于 GPU 和中断。
- **原训练/评估入口有硬编码作者路径**：例如 `/nas0/dataset/...`、`/home/vggt-pr/...` 和不存在的 checkpoint 路径。原 `main_*`/`eval_lora.py` 不能直接在当前机器复现实验，需显式改配置；本次任务要求未修改原 UniPR 代码。
- **没有 requirements/environment lock 文件**：环境可通过当前 conda `py311` 的包版本复现，但项目本身未锁版本。
- **正式 02 校准报告中的 ignore 状态不是部署行为**：`calibrate_kitti_three_way_decision.py` 在设置 state 时使用 `trainable` mask，因此所有 ignore 行在校准 CSV 中都会被强制保留为 `uncertain`。测试时 `evaluate_kitti_frozen_three_way.py` 不使用标签 mask，ignore 可以被 accept/reject/uncertain。不要用 02 校准报告中的 ignore state 分布推断部署表现。

### 方法和实现限制

- `camera_rotation_consistency_deg` 和 `camera_translation_consistency` 已提取但未输入正式 decision head。
- `mean_match_similarity` 当前固定为 `0.0`，因为官方 track API 不暴露 token descriptor。
- 正式所谓 hard negatives 实际定义为所有 rank 1-10 negative；没有根据训练后模型分数或几何混淆度做二次 online hard-negative mining。
- 正式训练使用固定 1:3 batch，但没有按真实部署先验加 sample weight；概率含义依赖后续 02 Platt calibration。
- 正式训练池保存全部 11,985 个正例，但训练循环只处理完整的 16-positive batch；`11985 % 16 = 1`，因此每个 epoch 会随机跳过 1 个正例。由于每轮重新 shuffle，不是固定遗漏同一条，但严格来说并非每个 epoch 使用全部正例。
- 02 使用完整 top-50 时 `(rank,label)` population/sample 权重实际为 1；因此正式报告虽然字段名写作 `weighted_metrics`，数值实际上等同于未加权的完整 02 population 指标。保留权重代码是为了兼容早期分层子集。
- 训练只有序列 00，没有训练内独立 sequence validation 或 early stopping；选择 logistic head 是为了降低过拟合风险。
- 标签仅使用 KITTI X/Z 平面距离，不考虑 yaw、视场重叠、道路拓扑或真正可见性。5m-25m 统一为 ignore。
- `ignore` 不参与 BCE 和 Platt 二分类拟合；测试时仍可能被 accept/uncertain/reject，需要单独解释。
- 05 的 `accept_pair_precision=1.0` 只针对明确 positive/negative；ignore accept 不计入 false positive。正式 05 中有 221 个 ignore 被 accept。
- 冻结 VGGT checkpoint 使用 `strict=False` 加载；当前代码会返回 missing/unexpected keys，但批量提取产物没有集中保存或报告这些 key。实际加载是否完全匹配：待确认，建议下一位 Agent 做一次 smoke 并打印。
- 当前 `VGGTGeometryVerifier` 的 autocast 判断依赖 `self.device == "cuda"`；现有命令使用 `--device cuda` 正常。若未来传入 `cuda:0`，autocast 会被关闭。
- 当前读取完整 chunks 时大量使用 Python `list(csv.DictReader(...))`，内存效率一般。
- 正式训练固定了 Python/NumPy/Torch seed，但没有开启 PyTorch deterministic algorithms；GPU 训练结果不保证逐 bit 可复现。
- `extract_kitti_descriptors.py` 和 `build_kitti_topk_pairs.py` 没有“拒绝覆盖”保护，传入已有输出路径会覆盖文件；`train_kitti_full_decision_head.py` 和 `calibrate_kitti_three_way_decision.py` 则会拒绝覆盖已有输出目录。运行前必须区分。

### 环境审计限制

- 当前 Codex 只读沙箱内 `nvidia-smi` 无法连接驱动，且 import torch/torchvision 时临时目录不可写；这不等同于实际训练终端环境损坏。此前所有 GPU 任务使用授权 shell 正常运行。
- 机器历史信息来自实际运行时用户提供的 `nvidia-smi`：4 张 RTX 4090、driver `580.126.09`、`nvidia-smi` 显示 CUDA `13.0`。2026-08-15 当前瞬时 GPU 状态：待确认。

## 10. 已完成的实验和结果

### 10.1 Descriptor / top-50 数据规模

| 序列 | descriptor shape | retrieval queries | top-50 pairs | positive | negative | ignore |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 4537 x 17152 | 4,437 | 220,625 | 11,985 | 183,100 | 25,540 |
| 02 | 4657 x 17152 | 4,557 | 226,625 | 2,488 | 215,203 | 8,934 |
| 05 | 2757 x 17152 | 2,657 | 131,625 | 5,489 | 111,676 | 14,460 |
| 06 | 1097 x 17152 | 997 | 48,625 | 2,629 | 36,028 | 9,968 |

每个序列前 100 个 descriptor center 没有满足 `temporal_gap >= 100` 的历史候选，因此不进入 pair CSV；其中最早的 49 个可检索 query 还不足 50 个历史候选。

06 descriptor retrieval：

```text
Queries with at least one historical positive: 266
Recall@1  = 0.9812
Recall@5  = 0.9887
Recall@10 = 0.9962
Recall@20 = 1.0000
Recall@50 = 1.0000
```

05 descriptor retrieval：

```text
Historical-positive queries: 448
Retrieved-positive queries: 427
Recall@50 = 0.953125
```

### 10.2 早期 600 对可行性基线

- 00 train：100 positive + 500 negative。
- 02 validation：100 positive + 500 negative。
- 7 个 descriptor/cross-token 特征，logistic head，weighted BCE。
- 02 sampled validation：ROC-AUC 1.0，AP 1.0，precision 0.9901，recall 1.0。
- 该结果只证明特征可分，不代表完整 top-50 部署性能。

早期 rank-stratified 02 校准后，在 05 的旧基线测试：

```text
Descriptor Recall@50                  0.953125
accept pair precision                 1.0000
accept pair recall | top-50           0.6990
accept query recall | retrieved       0.8806
accept query recall end-to-end        0.8393
reranked Recall@1 end-to-end          0.8906
false positives/query                 0.0
```

对应产物：`outputs/kitti_05_frozen_three_way_eval/report.json`。

### 10.3 正式 00 全量训练

完整 00 3D 特征：`2207/2207` chunks，`220,625` pairs。

训练协议：

- 全部 11,985 positive。
- 负例池 47,865：所有 rank 1-10 negative + rank 11-50 每 rank 300 negative。
- batch 64：16 positive + 48 negative。
- logistic head，50 epochs。
- 7 个正式输入特征。

训练池拟合检查：

```text
ROC-AUC = 0.9991186709630988
AP      = 0.9979808236769145
```

产物：

- `outputs/kitti_cross_token_3d_full_decision_head_00/decision_head.pt`
- `outputs/kitti_cross_token_3d_full_decision_head_00/report.json`

### 10.4 正式 02 全量校准

完整 02 特征：`2267/2267` chunks，`226,625` pairs。

校准结果：

```text
ROC-AUC             0.9982079481
AP                  0.9552780062
Brier               0.0020805370
accept TP/FP/FN     1734 / 17 / 754
accept precision    0.9902912621
accept recall       0.6969453376
```

冻结规则：

```text
accept_probability                  0.9470927362616622
reject_probability                  0.05
track_confidence_accept_min         0.10930951356887818
visibility_accept_min               0.1663115668296814
median_3d_residual_accept_max       0.34197994112968433
track_confidence_reject_max         0.3135078063607182
visibility_reject_max               0.16327825635671586
```

产物：`outputs/kitti_cross_token_3d_full_three_way_calibration_02/`。

### 10.5 正式 05 独立测试

完整 `1317/1317` chunks，`131,625` pairs。不使用 05 调参。

```text
Descriptor Recall@50                  0.953125
accept pair precision                 1.0000
accept pair recall | top-50           0.5135726
accept query recall | retrieved       0.8407494
accept query recall end-to-end        0.8013393
reranked Recall@1 end-to-end          0.8928571
false positives/query                 0.0

states:
  accept       3,040
  uncertain   12,399
  reject     116,186

accepted by label:
  positive  2,819
  negative      0
  ignore      221
```

产物：`outputs/kitti_05_full_three_way_eval/report.json`。

结论边界：05 上未观察到明确 negative 被 accept，安全性强；端到端安全 accept query recall 为约 80.13%，模型较保守。是否跨序列泛化必须等 06 完成后判断。

指标定义以 `summarize_kitti_frozen_three_way.py` 为准：

- `accept_pair_precision` 只把明确 negative accept 计为 FP，ignore 不进入分母。
- `accept_query_recall_conditional_top50` 的分母是 top-50 中实际出现 positive 的 query 数。
- `accept_query_recall_end_to_end` 的分母是通过 GT 扫描得到的全部 historical-positive query 数。
- `reranked_recall_at_1_end_to_end` 对每个 retrieval query 选择 `calibrated_probability` 最大的候选，再以 historical-positive query 总数为分母；该指标不要求 top-1 候选状态必须是 accept。
- `false_positives_per_query` 的分母是 pair CSV 中全部 retrieval query 数，不是 historical-positive query 数。

### 10.6 正式 06 独立测试

- descriptor 和 top-50 已完成。
- 目标总数：48,625 pairs，487 chunks。
- retrieval query 数：997；historical-positive query 数：266；retrieved-positive query 数：266。
- 当前完成：345 chunks / 34,500 pairs。
- 尚无最终 report，不能给出泛化结论。

## 11. 数据集、模型和权重

### KITTI 数据

实际根目录：

```text
/data1/jiaming/data-0102/dataset
```

已确认存在：

```text
sequences/00/image_2 + calib.txt
sequences/02/image_2 + calib.txt
sequences/05/image_2 + calib.txt
sequences/06/image_2 + calib.txt
poses/00.txt
poses/02.txt
poses/05.txt
poses/06.txt
```

当前实验只使用 KITTI odometry 彩色 `image_2`、calibration 和 GT pose；没有使用 Velodyne 点云作为模型输入，也没有使用 KITTI 外部 depth。

### 原项目数据

- `datasets/` 中存在 MSLS/Nordland/Pittsburgh/SPED 的部分元数据或少量内容。
- 用户早期只下载过 MSLS images volume 1；其完整性和当前是否仍在工作区：待确认。
- 原训练数据路径在代码中多为作者机器 `/nas0/dataset/...`，当前不可直接使用。

### 权重

```text
model/multi_model.ckpt
  size: 3,703,481,315 bytes
  UniPR-3D multi-frame checkpoint

model/VGGT-model/model.pt
  size: 5,026,874,952 bytes
  官方 facebook/VGGT-1B geometry checkpoint

outputs/kitti_cross_token_3d_full_decision_head_00/decision_head.pt
  正式 00 logistic head

outputs/kitti_cross_token_3d_full_three_way_calibration_02/frozen_three_way_decision.pt
  正式冻结 head + Platt + accept/uncertain/reject rules
```

VGGT model card 标注 license 为 `cc-by-nc-4.0`；原 UniPR README 标注 MIT。发布或商业使用前需要确认组合许可边界。

## 12. Python / CUDA / PyTorch 环境信息

实际用于当前流水线的 Python：

```text
/data1/jiaming/.conda/envs/py311/bin/python
Python 3.11.15 (conda-forge)
Linux 5.15.0-176-generic x86_64, glibc 2.35
```

关键包（由当前 conda 包元数据读取）：

```text
torch                  2.5.1+cu121
torchvision            0.20.1+cu121
torchaudio             2.5.1+cu121
numpy                  1.26.4
scikit-learn           1.5.2
pandas                 2.2.3
Pillow                 12.2.0
einops                 0.8.0
lightning              2.6.0
pytorch-lightning      2.6.5
tqdm                   4.69.1
opencv-python-headless 4.10.0.84
faiss-gpu-cu12         1.9.0.post1
xformers               0.0.28.post3
```

GPU/driver 历史确认：

```text
4 x NVIDIA GeForce RTX 4090, each 24 GB
NVIDIA driver 580.126.09
nvidia-smi CUDA version 13.0
PyTorch build CUDA 12.1
```

注意：`nvidia-smi` 的 CUDA 13.0 是驱动支持上限；当前 PyTorch wheel 是 cu121，这种组合此前已正常运行。通常通过：

```bash
CUDA_VISIBLE_DEVICES=2 ... --device cuda
```

使用物理 GPU 2；进程内部该卡映射为 `cuda:0`。

项目另有根目录 `.venv`，但正式实验实际使用 conda `py311`；`.venv` 的用途和完整性：待确认。

## 13. 项目运行方法

所有命令从项目根目录执行。

### 13.1 激活环境

```bash
conda activate /data1/jiaming/.conda/envs/py311
```

或始终使用绝对 Python：

```bash
/data1/jiaming/.conda/envs/py311/bin/python
```

### 13.2 单个 5 帧序列跑 UniPR descriptor

```bash
CUDA_VISIBLE_DEVICES=2 /data1/jiaming/.conda/envs/py311/bin/python \
  run_multiframe_sample.py \
  --ckpt model/multi_model.ckpt \
  --image_dir /path/to/images \
  --seq_len 5 \
  --image_size 392 518 \
  --device cuda
```

### 13.3 KITTI descriptor

```bash
CUDA_VISIBLE_DEVICES=2 /data1/jiaming/.conda/envs/py311/bin/python \
  extract_kitti_descriptors.py \
  --root /data1/jiaming/data-0102/dataset \
  --sequence_id 06 \
  --ckpt model/multi_model.ckpt \
  --output outputs/kitti_06_seq5_stride1_unipr_descriptors.pt \
  --seq_len 5 --stride 1 \
  --image_size 392 518 \
  --batch_size 4 --num_workers 4 \
  --device cuda
```

### 13.4 Historical top-50

```bash
/data1/jiaming/.conda/envs/py311/bin/python \
  build_kitti_topk_pairs.py \
  --descriptors outputs/kitti_06_seq5_stride1_unipr_descriptors.pt \
  --output outputs/kitti_06_seq5_stride1_unipr_descriptors_top50_pairs.csv \
  --top_k 50 \
  --min_temporal_gap 100 \
  --positive_radius 5 \
  --negative_radius 25 \
  --distance_axes 0 2 \
  --block_size 256
```

### 13.5 全量 cross-token 3D 特征提取（训练/校准序列）

```bash
CUDA_VISIBLE_DEVICES=2 /data1/jiaming/.conda/envs/py311/bin/python \
  extract_kitti_cross_token_3d_full.py \
  --descriptors outputs/kitti_00_seq5_stride1_unipr_descriptors.pt \
  --pairs outputs/kitti_00_seq5_stride1_unipr_descriptors_top50_pairs.csv \
  --output_dir outputs/kitti_00_top50_cross_token_3d_full \
  --vggt_ckpt model/VGGT-model/model.pt \
  --image_size 392 518 \
  --grid_rows 12 --grid_cols 16 \
  --ransac_threshold_m 0.5 \
  --ransac_iterations 256 \
  --seed 17 --chunk_size 100 \
  --device cuda
```

重复同一命令会跳过已有完整 chunk。

### 13.6 正式 00 decision head 训练

```bash
CUDA_VISIBLE_DEVICES=2 /data1/jiaming/.conda/envs/py311/bin/python \
  train_kitti_full_decision_head.py \
  --feature_dir outputs/kitti_00_top50_cross_token_3d_full \
  --pairs outputs/kitti_00_seq5_stride1_unipr_descriptors_top50_pairs.csv \
  --output_dir outputs/kitti_cross_token_3d_full_decision_head_00 \
  --head logistic \
  --epochs 50 \
  --batch_size 64 --positive_per_batch 16 \
  --regular_negative_per_rank 300 \
  --learning_rate 1e-2 --weight_decay 1e-3 \
  --seed 17 --device cuda
```

不要重复运行到现有同名输出目录；脚本会拒绝覆盖。

### 13.7 正式 02 校准

```bash
/data1/jiaming/.conda/envs/py311/bin/python \
  calibrate_kitti_three_way_decision.py \
  --head_checkpoint outputs/kitti_cross_token_3d_full_decision_head_00/decision_head.pt \
  --calibration_dir outputs/kitti_02_top50_cross_token_3d_full \
  --full_pairs outputs/kitti_02_seq5_stride1_unipr_descriptors_top50_pairs.csv \
  --output_dir outputs/kitti_cross_token_3d_full_three_way_calibration_02 \
  --target_precision 0.99 \
  --reject_probability 0.05
```

### 13.8 冻结测试（05/06）

06 当前续跑命令：

```bash
CUDA_VISIBLE_DEVICES=2 /data1/jiaming/.conda/envs/py311/bin/python \
  evaluate_kitti_frozen_three_way.py \
  --descriptors outputs/kitti_06_seq5_stride1_unipr_descriptors.pt \
  --pairs outputs/kitti_06_seq5_stride1_unipr_descriptors_top50_pairs.csv \
  --frozen_decision outputs/kitti_cross_token_3d_full_three_way_calibration_02/frozen_three_way_decision.pt \
  --output_dir outputs/kitti_06_full_three_way_eval \
  --vggt_ckpt model/VGGT-model/model.pt \
  --image_size 392 518 \
  --grid_rows 12 --grid_cols 16 \
  --ransac_threshold_m 0.5 \
  --ransac_iterations 256 \
  --seed 17 --chunk_size 100 \
  --device cuda
```

建议在 `tmux` 中运行。查看进度：

```bash
find outputs/kitti_06_full_three_way_eval/chunks \
  -name 'chunk_*.csv' | wc -l
```

目标为 `487`。

### 13.9 汇总 06

06 完成后：

```bash
/data1/jiaming/.conda/envs/py311/bin/python \
  summarize_kitti_frozen_three_way.py \
  --pairs outputs/kitti_06_seq5_stride1_unipr_descriptors_top50_pairs.csv \
  --evaluation_dir outputs/kitti_06_full_three_way_eval \
  --historical_positive_queries 266 \
  --output outputs/kitti_06_full_three_way_eval/report.json
```

脚本会验证全部 48,625 个 `pair_row_index` 是否完整。注意生成的 `note` 会错误提到 sequence 05，见已知 Bug；不要修改指标，只需在报告解释中注明。

## 14. 下一步应该做什么

按优先级：

1. 在 `tmux` 或调度器中用 13.8 的原命令恢复 06，从 `chunk_00345` 继续到 `486`。
2. 确认 `487` 个 chunks 后运行 13.9 汇总，生成 06 正式报告。
3. 将 06 与 05 比较，重点看：
   - descriptor Recall@50；
   - accept pair precision；
   - 明确 negative accept 数；
   - accept query recall end-to-end；
   - reranked Recall@1；
   - false positives/query；
   - positive/negative/ignore 的三态分布。
4. 不得根据 06 结果回调 00 head、02 Platt 或三态阈值；否则 06 不再是独立测试。
5. 形成正式实验表：600 对基线、00 全量正式模型、05/06 独立测试。
6. 若 05/06 都安全但 recall 偏低，下一研究阶段应处理 uncertain，而不是直接用测试集调阈值：可在新的验证序列上研究更多帧、retrieval margin、候选熵、yaw/pose embedding 或后端二次验证。
7. 若继续优化速度，应优先消除同一 query 对 top-50 重复 VGGT 前向，设计可缓存/复用的 query/candidate token 或只对 top-N/uncertain 候选运行重型 VGGT。
8. 找回原上游 Git remote/基准 commit，并补充依赖锁文件，再开始较大代码重构。当前新仓库仅代表本机代码快照。

## 15. 需要特别注意的地方

- 不要修改原 UniPR 主流程，除非用户明确改变这一约束。当前所有 KITTI/cross-token 工作都是独立新增模块。
- 00 是训练、02 是校准、05/06 是独立测试；不要混用序列职责。
- GT pose、`gt_distance_m` 和 `label` 只用于训练监督和离线指标；`evaluate_kitti_frozen_three_way.py` 推理输入只使用 descriptor similarity、RGB-derived VGGT geometry 和冻结 head/rules。
- 05/06 的 `ignore` 是 5m-25m 灰区，不应自动算作明确 false positive，但必须单独报告 accept 数。
- 当前正式 frozen artifact 是：
  `outputs/kitti_cross_token_3d_full_three_way_calibration_02/frozen_three_way_decision.pt`。
  不要误用早期 600 对版本：
  `outputs/kitti_cross_token_3d_three_way_calibration_02/frozen_three_way_decision.pt`。
- 正式 05 输出目录是 `outputs/kitti_05_full_three_way_eval`；旧基线是 `outputs/kitti_05_frozen_three_way_eval`。
- KITTI descriptor 文件中保存 pose/translation 是为了生成标签和便于分析；`extract_kitti_descriptors.py` 的模型前向没有传入 pose，且该 checkpoint 的 `with_camera_pose=False`。
- 完整 chunk 的存在才表示该 100 对已完成；进程在 chunk 中间中断会在续跑时重算最多一个 chunk，不会覆盖已完成 chunk。
- `CUDA_VISIBLE_DEVICES=2` 后，物理 GPU 2 在进程内部是 `cuda:0`；命令参数仍使用 `--device cuda`，不要改成 `cuda:2`。
- 当前脚本前台运行会随终端或服务器断连终止，必须使用 `tmux`/`screen`/scheduler 才能可靠完成长任务。
- `outputs/`、`model/multi_model.ckpt` 和 `model/VGGT-model/model.pt` 已加入 `.gitignore`，不会进入普通 Git。它们需要 Git LFS、artifact storage 或独立下载说明。
- 本文创建时未修改任何现有项目代码，仅新增 `PROJECT_HANDOFF.md`。
