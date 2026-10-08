# CALA：校准弃权的回环关联

本仓库是 **CALA（Calibrated Abstention Loop Association）** 的实验代码，不是 UniPR-3D 的官方发布。

CALA 解决的问题是：给定当前 5 帧窗口，是否输出**一个**历史 5 帧窗口作为回环。没有足够证据时必须弃权。错误关联的风险高于漏检，因此每个 query 最多接受一个历史窗口。本工作停在回环关联前端，不估计相对位姿，也不做 pose graph。

描述子和几何骨干都是冻结的第三方模型：

- 全局描述子来自 [UniPR-3D](https://github.com/dtc111111/UniPR-3D)（Deng et al., ECCV 2026，arXiv:2512.21078）。官方仓库负责视觉地点识别模型本身。
- 几何证据来自冻结的 [VGGT](https://github.com/facebookresearch/vggt)。

这里新增的是 KITTI 上的历史检索、cross-token 3D 特征、Logistic 决策头、Platt 校准，以及 accept / uncertain / reject 的单一窗口选择。

## 协议

KITTI `image_2`，5 帧 center-first 窗口，输入分辨率 392×518。标签只用中心帧的 X/Z 距离，并且只用于训练、校准和评估：

| 条件 | 标签 |
|---|---|
| 中心帧时间差 < 100 | 不进入历史检索 |
| 距离 ≤ 5 m | positive |
| 5 m < 距离 < 25 m | ignore |
| 距离 ≥ 25 m | negative |

数据划分在实验开始前固定，测试序列不参与选阈值：

| 序列 | 用途 |
|---|---|
| 00 | 训练 7 维 Logistic |
| 02 | Platt 校准，并按 pair 级加权精度 ≥ 0.99 选择接受阈值 |
| 05、06 | 独立测试 |
| 07 | 难例分析，不用于选阈值 |
| 09 | 冻结后的确认集（historical-positive query 只有 17 个） |

正式冻结点为 seed 17。序列 02 上的接受概率阈值是 0.9470927362616622。几何硬门同样只在序列 02 上确定：track confidence ≥ 0.10931，visibility ≥ 0.16631，median 3D residual ≤ 0.34198 m。选窗使用时间聚类后的未校准 `raw_logit`。

Strict precision 把 ignore 记为错误。端到端召回的分母是全部 historical-positive query。

## 正式结果

显式负关联均为 0。数字来自冻结输出，不是在 05/06/07/09 上重新调阈值得到的。

| 序列 | Strict P (%) | 端到端召回 (%) | 输出（正 / ignore / 负） |
|---|---:|---:|---|
| 05 | 99.72 | 80.13 | 359 / 1 / 0，共 360 |
| 06 | 99.61 | 95.11 | 253 / 1 / 0，共 254 |
| 07 | 100.00 | 31.67 | 19 / 0 / 0 |
| 09 | 100.00 | 70.59 | 12 / 0 / 0 |

同一协议下，描述子阈值基线 UniPR-R1+τ₀₂（τ = 0.86845386，由序列 02 的 query 级 Strict P ≥ 0.99 确定）在召回上持平或更高，例如 05 为 100.00 / 81.25，06 为 98.84 / 96.24，07 为 100.00 / 46.67。CALA 相对该基线的可见差别，主要是序列 06 上更严的边界控制，以及一条可以逐项拆开的证据链。不把 Rank-1 无阈值检索当作安全基线，它会产生大量显式负关联。

单组件消融、种子 17/21/23 和效率测量写在 `UniPR-3D_ablation_tables.md` 与 `UniPR-result.md`。几何硬门在该冻结点上没有独立增量。种子均值只包含能够完整满足官方 0.99 规则并完成迁移的 17、21、23。

## 仓库里有什么

包含 KITTI 关联实验的脚本、测试和冻结记录。不包含：

- `model/multi_model.ckpt` 与 `model/VGGT-model/model.pt`
- `outputs/` 下的描述子、pair 表、三态评估和决策头权重

这些文件留在实验机器上。没有它们时，仓库里的脚本不能直接重跑完整实验。

主要入口：

| 文件 | 作用 |
|---|---|
| `extract_kitti_descriptors.py` | 用冻结 UniPR-3D 提取 5 帧描述子 |
| `build_kitti_topk_pairs.py` | 历史 top-50 检索 |
| `extract_kitti_cross_token_3d_full.py` | 冻结 VGGT 的 cross-token 3D 特征 |
| `train_kitti_decision_head.py` | 在序列 00 上训练决策头 |
| `evaluate_kitti_loop_association.py` | 正式 accept-only 评估 |
| `evaluate_kitti_loop_association_descriptor_threshold.py` | UniPR-R1+τ₀₂ 基线 |
| `benchmark_kitti_cala_efficiency.py` | 效率测量，不改冻结输出 |
| `kitti_frozen_baseline_2026-08-18/` | 2026-08-18 冻结记录 |

上游 UniPR-3D 的训练和 VPR 评测入口仍保留在仓库中，例如 `main_lora_multiframe.py`。那些脚本复现的是 UniPR-3D 自己的地点识别实验，不是 CALA 的 KITTI 回环关联结果。

## 引用骨干

如果使用其中的 UniPR-3D 描述子代码，请引用原论文：

```bibtex
@inproceedings{deng2026unipr3d,
  title     = {UniPR-3D: Towards Universal Visual Place Recognition with 3D Visual Geometry Grounded Transformer},
  author    = {Deng, Tianchen and Chen, Xun and Li, Ziming and Shen, Hongming and Wang, Danwei and Civera, Javier and Wang, Hesheng},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```
