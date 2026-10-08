# KITTI 05/06 冻结基线包

冻结标识：`kitti-frozen-baseline-2026-08-18`

本目录用于固定 UniPR-3D + VGGT 多帧安全回环检测的正式实验口径、05/06 独立测试结果和关键产物完整性。冻结不复制大型模型或中间数据；原始产物保留在项目原路径，本目录通过 SHA-256 对其进行唯一标识，并保存关键报告快照。

## 目录内容

- `FORMAL_EXPERIMENT_REPORT.md`：正式实验流程、结果表、结果解读和结论边界。
- `FREEZE_POLICY.md`：冻结范围、禁止事项、后续版本规则和校验方法。
- `experiment_baseline.json`：机器可读的实验划分、正式参数和指标。
- `metrics.csv`：便于导入表格或绘图工具的主要指标。
- `integrity_manifest.json`：核心模型、数据、报告、源码和 chunk 集合的完整性清单。
- `SHA256SUMS`：27 个核心产物和 13 个关键源码文件的 SHA-256。
- `COLLECTION_SHA256SUMS`：00/02 特征 chunks、早期 05 测试 chunks、正式 05/06 测试 chunks 的集合校验值。
- `report_snapshots/`：早期基线及正式 00/02/05/06 报告的只读快照。
- `FREEZE_FOLDER_SHA256SUMS`：本冻结目录自身文件的校验值，不包含该校验文件自身。

## 冻结状态

- 正式 decision head：序列 00 训练，已冻结。
- 正式 Platt 与三态规则：序列 02 校准，已冻结。
- 序列 05：第一独立测试，已完成并冻结。
- 序列 06：第二独立测试，已完成并冻结。
- 05/06 不得用于回调 00 head、02 Platt 参数、三态阈值或几何门控。

## 重要限制

冻结时 Git `HEAD` 为 `9c3a2eb5876025e7a3b1536307e2fe7b4692aa0d`，但工作区不是 clean。冻结清单记录了当时的 Git 状态，并逐文件保存关键源码 SHA-256。因此，本冻结包的源码身份应以 `integrity_manifest.json` 中的 `source_files` 哈希为准，而不能只依赖 Git commit。

本目录提供完整性证明，不等同于大型产物备份。模型、descriptor、pairs 和 chunks 仍应另行备份。
