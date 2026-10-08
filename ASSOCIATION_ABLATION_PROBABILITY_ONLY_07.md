# KITTI 07 关联对照 5：只要冻结概率阈值

本文件只记录对照 5。不修改冻结 accept-only 以及对照 1–4 的结论。

冻结 07 accept-only 报告哈希（运行前后不变）：

`ee9c9bd9fedd89e76b16ba17b553068a0acad745f9bde3f1eee14b2c39b37aa1`

## 1. 实验变量

相对冻结 accept-only 的唯一变量：去掉 vis / track / residual 硬门控，只保留 `calibrated_probability ≥ 0.9470927362616622`（02 冻结，未改）。过阈值的候选仍按同一套 `raw_logit` + cluster gap=5 选一个。

产物：

- `outputs/kitti_07_loop_association_probability_only_eval/`
- 脚本：`evaluate_kitti_loop_association_probability_only.py`

## 2. 07 结果

| 指标 | 对照 5 仅概率 | 冻结 accept-only（参考） |
|---|---:|---:|
| 输出关联 / 弃权 | 19 / 978 | 19 / 978 |
| 正 / ignore / 负 | 19 / 0 / 0 | 19 / 0 / 0 |
| Strict precision | 100% | 100% |
| 条件召回 | 41.30%（19/46） | 41.30% |
| 端到端召回 | 31.67%（19/60） | 31.67% |

只读核对：

- 19 个 query、19 条 `pair_row_index` 与冻结 accept-only **完全相同**。
- 这 19 条三态都是 accept，且都过冻结几何门。
- 978 个弃权 query 的 top-50 里，没有任何候选 `calibrated_probability ≥ 0.947`。
- 07 上不存在“概率已过、只被几何门单独挡掉”的关联。
