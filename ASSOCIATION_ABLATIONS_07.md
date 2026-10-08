# KITTI 07 关联选择对照（1 与 3）

本文件只记录新增对照，不修改冻结 accept-only 结论。正式工作点仍是 `outputs/kitti_07_loop_association_eval/`。

冻结 07 accept-only 报告哈希（运行前后不变）：

`ee9c9bd9fedd89e76b16ba17b553068a0acad745f9bde3f1eee14b2c39b37aa1`

## 1. 实验变量

共同协议（与冻结关联评估相同，未改）：

- GT：historical gap ≥ 100，中心帧 X/Z ≤5m 正、5–25m ignore、≥25m 负
- 每个 retrieval query 最多一个历史窗口
- 描述子、VGGT chunks、00 head、02 Platt 一律只读
- 07 只作开发对照；样本小（60 个 historical-positive query），不能当主表

新增脚本：`evaluate_kitti_loop_association_ablations.py`

| 对照 | 唯一变量 | 选择规则 |
|---|---|---|
| 1 `descriptor_rank1` | 去掉几何验证与决策头 | 每个 query 取 historical UniPR rank-1 |
| 3 `score_rerank` | 去掉冻结 `accept` 门控 | 对全部 top-50 用同一套 `raw_logit` + cluster gap=5 选最高分 |
| 冻结 accept-only | 参考，不是本次变量 | 只在 `state==accept` 中选；无 accept 则弃权 |

对照 1 不读 VGGT。对照 3 只读已有 `outputs/kitti_07_full_three_way_eval/chunks/`，不重跑推理。

## 2. 07 结果

| 指标 | 对照 1 rank-1 | 对照 3 分数重排 | 冻结 accept-only（参考） |
|---|---:|---:|---:|
| 输出关联 / 弃权 | 997 / 0 | 997 / 0 | 19 / 978 |
| 正 / ignore / 负 | 39 / 136 / 822 | 36 / 136 / 825 | 19 / 0 / 0 |
| Clear-label precision | 4.53% | 4.18% | 100% |
| Strict precision | 3.91% | 3.61% | 100% |
| 条件召回（46 retrieved-positive） | 84.78% | 78.26% | 41.30% |
| 端到端召回（60 historical-positive） | 65.00% | 60.00% | 31.67% |

AP / RP100 在对照里是对每个 query 的唯一选出窗口按该对照分数扫描，与冻结报告里的几何门控 AP 不可比，故不列入上表。

产物：

- `outputs/kitti_07_loop_association_descriptor_rank1_eval/`
- `outputs/kitti_07_loop_association_score_rerank_eval/`

## 3. 只读核对（不改变对照口径）

- 对照 1 选出的 rank 全是 1。
- 对照 3 在冻结已 accept 的 19 个 query 上，选出的历史窗口与冻结完全相同，且这 19 个仍是 positive。
- 对照 3 选出窗口的三态分布：accept 19、uncertain 107、reject 871。精度崩溃来自强制给 978 个原弃权 query 各塞一个最高分窗口，不是因为那 19 个 accept 被改坏。
- 在 46 个 retrieved-positive query 上：对照 1 为 39 正 / 7 ignore / 0 负；对照 3 为 36 正 / 10 ignore / 0 负。明确 negative 几乎全部来自本来没有 top-50 正例的 query。

## 4. 结论边界

- 冻结 accept-only 的 07 数字不因本次对照而改写。
- 只靠 UniPR rank-1 不能当安全关联：召回更高，但会输出 822 个 ≥25m 负例。
- 冻结分数重排若去掉 accept 门控，并不能代替三态；它几乎不改变已 accept 的 19 条，却会把弃权全部变成低精度关联。
- 07 正例少，百分点波动大；主表对照仍应在规则冻结后只读打 05/06。
