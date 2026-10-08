# KITTI 单一历史回环窗口关联评估

## 1. 目标

已有三态评估回答“某个 query-candidate pair 是否可以安全 accept”。本评估进一步回答：

> 对每个当前 query 5 帧窗口，系统最终选择的一个具体历史 5 帧窗口是否属于真实回环窗口集合。

本阶段只读取已完成的 05/06 frozen evaluation chunks，不重新运行 VGGT，不修改 00 decision head、02 Platt 参数、三态阈值或几何门控。

## 2. 固定协议

### GT 回环窗口

- historical temporal gap：至少 100 帧；
- query/candidate 中心帧 X/Z 平面距离：不大于 5m；
- 当前版本不使用 yaw 或视觉重叠；
- 5–25m 的 ignore 在严格关联 precision 中视为未正确关联，同时单独报告。

### 单一候选选择

1. 每个 query 只考虑状态为 `accept` 的候选。
2. candidate 中心帧差不超过 5 帧的相邻窗口归入同一 temporal cluster。
3. 每个 cluster 选择未校准、未饱和的 `raw_logit` 最高候选。
4. 再从 cluster representatives 中选择 `raw_logit` 最高者。
5. 并列时依次选择 candidate rank 更低、candidate center 更小、`pair_row_index` 更小者。
6. 没有 accept 候选时输出 `abstain`。

选择 `raw_logit` 而不是 `calibrated_probability`，是因为后者在高分区域饱和为相同数值，会产生顺序相关的并列。

## 3. 核心指标

- `association_precision_strict`：选中 positive 的 query 数 / 所有输出了关联的 query 数；ignore 按未正确关联处理。
- `association_precision_clear_labels`：只在 positive/negative 中计算关联 precision；ignore 不进入分母。
- `association_recall_conditional_retrieval`：选中 positive 的 query 数 / top-50 中存在 positive 的 query 数。
- `association_recall_end_to_end`：选中 positive 的 query 数 / 全部 historical-positive query 数。
- `association_average_precision`：在通过冻结 accept 几何门控的单一候选上按 raw logit 扫描得到；ignore 和 negative 都视为错误关联。
- `association_recall_at_100_precision`：严格无错误关联前提下可达到的最大端到端 recall。
- `abstention_rate_retrieval_queries`：所有 retrieval query 中没有输出 accept 关联的比例。该指标包含本来就没有真实回环的 query，不能单独解释为漏检率。

## 4. 正式结果

| 指标 | KITTI 05 | KITTI 06 |
|---|---:|---:|
| Historical-positive queries | 448 | 266 |
| Top-50 retrieved-positive queries | 427 | 266 |
| 输出单一关联的 queries | 360 | 254 |
| 正确 positive associations | 359 | 253 |
| 明确错误 negative associations | 0 | 0 |
| 灰区 ignore associations | 1 | 1 |
| Clear-label association precision | 1.000000 | 1.000000 |
| Strict association precision | 0.997222 | 0.996063 |
| Conditional association recall | 0.840749 | 0.951128 |
| End-to-end association recall | 0.801339 | 0.951128 |
| Association AP | 0.867245 | 0.984602 |
| Association RP100 | 0.790179 | 0.943609 |

## 5. 结果解释

- 05 最终输出 360 条单一 query-to-history 关联，其中 359 条落在 5m GT 正例范围，1 条落在 ignore 灰区。
- 06 最终输出 254 条单一关联，其中 253 条为 positive，1 条为 ignore。
- 两个序列均未选择距离不小于 25m 的明确 negative。
- 单一候选选择后，端到端 recall 与原来的“query 中存在任意 positive accept”一致，说明当前 mixed positive/ignore accept query 中，最高 raw logit 代表均选择了 positive。
- 严格 precision 不是 100%，原因是两个序列各有一个最终选择落在 5–25m 灰区。
- 当前GT只使用中心位置距离。若论文要求“视觉上确实是同一地点且可建立约束”，还需要在新版本中预先定义 yaw 和视觉重叠标准。

## 6. 产物

实现：

- `evaluate_kitti_loop_association.py`

测试：

- `tests/test_evaluate_kitti_loop_association.py`

05：

- `outputs/kitti_05_loop_association_eval/report.json`
- `outputs/kitti_05_loop_association_eval/selected_associations.csv`
- `outputs/kitti_05_loop_association_eval/association_precision_recall_curve.csv`

06：

- `outputs/kitti_06_loop_association_eval/report.json`
- `outputs/kitti_06_loop_association_eval/selected_associations.csv`
- `outputs/kitti_06_loop_association_eval/association_precision_recall_curve.csv`

## 7. 结论边界

本评估完成了“具体历史窗口关联”的闭环，但仍属于回环检测前端，不输出相对 SE(3) 位姿，也不执行 pose graph correction。05/06 保持冻结测试身份；不得根据本报告回调模型或阈值。
