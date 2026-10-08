# KITTI 07 关联对照 2：UniPR 相似度阈值

本文件只记录对照 2。不修改冻结 accept-only、对照 1、对照 3 的结论。

冻结 07 accept-only 报告哈希（运行前后不变）：

`ee9c9bd9fedd89e76b16ba17b553068a0acad745f9bde3f1eee14b2c39b37aa1`

## 1. 实验变量

相对对照 1 的唯一变量：给 historical UniPR rank-1 加上一个 **只在 02 上选定、不再改** 的相似度阈值；低于阈值弃权。无 VGGT、无决策头。

阈值规则（02 only，与 `calibrate_kitti_three_way_decision.py` 默认 `target_precision=0.99` 对齐）：

- 每个 02 retrieval query 取 rank-1
- 选最低的 `descriptor_similarity`，使 query 级严格 precision ≥ 0.99（ignore 算错）
- 并列时保留更多正例

02 冻结结果：

- 阈值 `0.86845386`
- 02 上选出 239：237 正 / 2 ignore / 0 负，strict P = 99.16%

产物：

- `outputs/kitti_02_descriptor_similarity_threshold/`
- `outputs/kitti_07_loop_association_descriptor_threshold_eval/`
- 脚本：`evaluate_kitti_loop_association_descriptor_threshold.py`

## 2. 07 结果

| 指标 | 对照 2 阈值 | 对照 1 rank-1（参考） | 冻结 accept-only（参考） |
|---|---:|---:|---:|
| 输出关联 / 弃权 | 28 / 969 | 997 / 0 | 19 / 978 |
| 正 / ignore / 负 | 28 / 0 / 0 | 39 / 136 / 822 | 19 / 0 / 0 |
| Strict precision | 100% | 3.91% | 100% |
| 条件召回 | 60.87%（28/46） | 84.78% | 41.30% |
| 端到端召回 | 46.67%（28/60） | 65.00% | 31.67% |

只读核对：

- 冻结已 accept 的 19 个 query，对照 2 全部仍输出正例；同一历史窗口只有 2/19。
- 对照 1 的 822 个负例、136 个 ignore，相似度均低于 0.868，对照 2 全部弃权。
- 对照 1 多出的 11 个正例被阈值裁掉，距离 3.7–4.9m，相似度 0.723–0.868。
- 对照 2 相对冻结多出的 9 个正例，距离 3.1–4.3m，仍是 rank-1 窗口。
