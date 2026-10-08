# KITTI 07 关联对照 4：只用冻结几何门控

本文件只记录对照 4。不修改冻结 accept-only 以及对照 1/2/3 的结论。

冻结 07 accept-only 报告哈希（运行前后不变）：

`ee9c9bd9fedd89e76b16ba17b553068a0acad745f9bde3f1eee14b2c39b37aa1`

## 1. 实验变量

相对冻结 accept-only 的唯一变量：去掉 00 logistic 与 02 概率阈值，只保留 02 冻结的 vis / track / residual 门。过门候选按 `candidate_rank` 选一个；全部不过则弃权。不使用 `raw_logit`。

门控（只读自冻结决策，未改）：

- `mean_track_confidence ≥ 0.1093`
- `mean_visibility ≥ 0.1663`
- `median_3d_residual ≤ 0.3420`

产物：

- `outputs/kitti_07_loop_association_geometry_only_eval/`
- 脚本：`evaluate_kitti_loop_association_geometry_only.py`

## 2. 07 结果

| 指标 | 对照 4 仅几何门 | 冻结 accept-only（参考） |
|---|---:|---:|
| 输出关联 / 弃权 | 100 / 897 | 19 / 978 |
| 正 / ignore / 负 | 36 / 64 / 0 | 19 / 0 / 0 |
| Clear-label precision | 100% | 100% |
| Strict precision | 36.00% | 100% |
| 条件召回 | 78.26%（36/46） | 41.30% |
| 端到端召回 | 60.00%（36/60） | 31.67% |

只读核对：

- 明确 ≥25m 负例为 0；64 个错误全是 5.0–12.5m ignore。
- 冻结 19 个 accept query 在对照 4 仍都输出正例，但同一历史窗口只有 2/19。
- 多出的 17 个正例距离 3.1–4.8m，三态全是 uncertain。
- 选出窗口的三态：accept 15、uncertain 85；64 个 ignore 全是 uncertain。
