# 冻结规则与完整性验证

## 1. 冻结对象

本冻结版本固定以下对象：

- UniPR-3D 多帧 checkpoint：`model/multi_model.ckpt`
- 官方 VGGT checkpoint：`model/VGGT-model/model.pt`
- 00 正式 decision head：`outputs/kitti_cross_token_3d_full_decision_head_00/decision_head.pt`
- 02 正式冻结决策：`outputs/kitti_cross_token_3d_full_three_way_calibration_02/frozen_three_way_decision.pt`
- 00/02/05/06 descriptors 和 top-50 pairs
- 00/02 全量 cross-token 3D feature chunks
- 05/06 正式 frozen evaluation chunks 和 reports
- 生成这些结果的关键源码快照

具体文件、大小、修改时间和 SHA-256 见 `integrity_manifest.json`。

## 2. 冻结后禁止事项

在仍使用“冻结基线”名称时，禁止：

1. 根据 05/06 指标修改 00 decision head。
2. 根据 05/06 指标重新拟合 02 Platt 参数。
3. 根据 05/06 指标修改 accept/reject probability。
4. 根据 05/06 指标修改 track、visibility 或 residual 几何门控。
5. 覆盖现有 05/06 report 而不创建新版本。
6. 将 ignore 自动并入 positive 或 negative 后继续沿用当前指标名称。
7. 修改输入帧数、图像尺寸、top-K、时间间隔、标签半径或特征集合后仍声称是同一基线。

## 3. 允许事项

不改变冻结结果的前提下，可以：

- 对 05/06 结果做描述性分析；
- 修复不影响数值结果的文档问题；
- 增强 chunk 续跑完整性校验；
- 修复进度显示；
- 为现有产物增加备份和校验；
- 开发新版本并与本基线比较。

如果根据 05/06 失败案例设计了新方法，05/06 可以继续作为“已消费测试集上的历史对比”，但不能再作为新方法完全无偏的最终测试集。

## 4. 新版本规则

任何影响模型输出或实验口径的变化必须：

1. 使用新的版本标识，例如 `kitti-baseline-v2`。
2. 使用新的输出目录，不覆盖当前产物。
3. 重新记录源码、权重、数据和报告哈希。
4. 明确新的开发/验证数据。
5. 使用未参与方法设计的新独立测试序列。
6. 与本冻结基线并列报告，不能替换或回写本版本。

## 5. 完整性校验

从项目根目录执行：

```bash
sha256sum -c kitti_frozen_baseline_2026-08-18/SHA256SUMS
```

该命令校验核心模型、descriptor、pairs、正式/早期产物和关键源码。

`COLLECTION_SHA256SUMS` 使用集合哈希，计算定义为：

```text
SHA256(sorted(relative_path + NUL + size_bytes + NUL + file_sha256 + newline))
```

集合哈希覆盖：

- 00 正式 feature chunks
- 02 正式 feature chunks
- 早期 05 evaluation chunks
- 正式 05 evaluation chunks
- 正式 06 evaluation chunks

集合的文件数、数据行数、总字节数、首末文件和 aggregate SHA-256 均保存在 `integrity_manifest.json`。

冻结目录自身可使用：

```bash
sha256sum -c kitti_frozen_baseline_2026-08-18/FREEZE_FOLDER_SHA256SUMS
```

## 6. Git 状态限制

冻结时：

- branch：`main`
- HEAD：`9c3a2eb5876025e7a3b1536307e2fe7b4692aa0d`
- 工作区：非 clean

因此 Git commit 只表示原始快照，不能单独代表本次冻结源码。当前冻结源码应以 `integrity_manifest.json` 中 13 个 `source_files` SHA-256 为准。

如果之后经用户明确授权提交冻结材料，应记录新的 commit，但不得重新生成或静默替换本冻结清单。

## 7. 备份要求

本目录保存报告快照和完整性证据，但没有复制约 8.7 GB 的两个基础模型，也没有复制 descriptor、pairs 和 chunks。建议将以下内容备份到独立磁盘或 artifact storage：

- `model/multi_model.ckpt`
- `model/VGGT-model/model.pt`
- `outputs/kitti_cross_token_3d_full_decision_head_00/`
- `outputs/kitti_cross_token_3d_full_three_way_calibration_02/`
- `outputs/kitti_05_full_three_way_eval/`
- `outputs/kitti_06_full_three_way_eval/`
- 四个序列的 descriptor 和 pair CSV
- 00/02 全量 feature chunks
- 本冻结目录

只有校验值而没有原文件，无法恢复丢失的产物。
