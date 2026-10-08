# UniPR-3D + VGGT 多帧安全回环检测正式冻结实验报告

冻结标识：`kitti-frozen-baseline-2026-08-18`  
冻结日期：2026-08-18（Asia/Shanghai）  
项目：UniPR-3D / KITTI Cross-Token 3D Loop Closure

## 1. 冻结目的

本报告固定当前已完成的实验闭环：

```text
KITTI 5-frame RGB
  -> frozen UniPR-3D global descriptor
  -> historical top-50 retrieval
  -> frozen VGGT cross-token 3D verification
  -> sequence-00 logistic decision head
  -> sequence-02 Platt calibration and three-way rules
  -> sequence-05 / sequence-06 independent tests
```

本次冻结以后，05/06 结果只作为已消费的独立测试基线，不得用于修改 00 decision head、02 Platt 参数、accept/reject 阈值或几何门控。

## 2. 固定实验口径

### 2.1 数据划分

| KITTI 序列 | 固定角色 | 是否可调参 |
|---|---|---|
| 00 | 正式 decision head 训练 | 仅用于训练 |
| 02 | 全量 Platt 校准和三态规则选择 | 仅用于校准 |
| 05 | 第一独立测试 | 否 |
| 06 | 第二独立测试 | 否 |

### 2.2 检索与标签

- 输入：KITTI `image_2` 中心帧优先的 5 帧 RGB，顺序为 `[t, t-2, t-1, t+1, t+2]`。
- 输入尺寸：`392 x 518`。
- descriptor：冻结 UniPR-3D 输出的 17152 维全局 descriptor。
- coarse retrieval：L2-normalized descriptor cosine similarity，historical top-50。
- 历史约束：`candidate_center <= query_center - 100`。
- positive：X/Z 平面距离不大于 5 m。
- negative：X/Z 平面距离不小于 25 m。
- ignore：距离位于 5 m 与 25 m 之间。
- GT pose、距离和标签只用于训练、校准及离线评估，不进入正式测试推理。

### 2.3 冻结模型与特征

正式 head 是 7 维输入的线性 logistic head：

1. `descriptor_similarity`
2. `weighted_3d_inlier_ratio`
3. `median_3d_residual`
4. `p90_3d_residual`
5. `mean_track_confidence`
6. `mean_visibility`
7. `temporal_consistency`

序列 00 的训练池保留 11,985 个 positive；negative pool 为 47,865 对。每 batch 固定 16 positive + 48 negative，共训练 50 epochs。

### 2.4 正式冻结规则

- Platt coefficient：`0.5642800416793863`
- Platt intercept：`-4.512386596854704`
- accept probability：`0.9470927362616622`
- reject probability：`0.05`
- accept track confidence minimum：`0.10930951356887818`
- accept visibility minimum：`0.1663115668296814`
- accept median 3D residual maximum：`0.34197994112968433`
- reject track confidence maximum：`0.3135078063607182`
- reject visibility maximum：`0.16327825635671586`

正式 02 使用完整 top-50 population，因此校准代码虽然兼容 `(rank, label)` 权重，正式运行中的 population/sample 权重实际均为 1。

## 3. 数据和产物规模

| 序列 | descriptor shape | retrieval queries | top-50 pairs | positive | negative | ignore |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 4537 x 17152 | 4,437 | 220,625 | 11,985 | 183,100 | 25,540 |
| 02 | 4657 x 17152 | 4,557 | 226,625 | 2,488 | 215,203 | 8,934 |
| 05 | 2757 x 17152 | 2,657 | 131,625 | 5,489 | 111,676 | 14,460 |
| 06 | 1097 x 17152 | 997 | 48,625 | 2,629 | 36,028 | 9,968 |

完整 chunk 集合：

- 00 全量训练特征：2207 chunks / 220,625 pairs。
- 02 全量校准特征：2267 chunks / 226,625 pairs。
- 05 正式测试：1317 chunks / 131,625 pairs。
- 06 正式测试：487 chunks / 48,625 pairs。

## 4. 正式实验总表

`N/A` 表示该阶段不定义该指标。02 的 ignore 在校准 CSV 中因 trainable mask 被保留为 uncertain，因此不能与部署测试中的 ignore state 比较。

| 阶段 | 性质 | accept pair precision | 明确 negative accept | ignore accept | 端到端 accept query recall | reranked Recall@1 |
|---|---|---:|---:|---:|---:|---:|
| 600 对早期模型 -> 05 | 早期可行性基线，非正式模型 | 1.000000 | 0 | 1,228 | 0.839286 | 0.890625 |
| 00 正式训练 | 训练池拟合检查 | N/A | N/A | N/A | N/A | N/A |
| 02 正式校准 | 校准集，不是独立测试 | 0.990291 | 17 | 不可比较 | N/A | N/A |
| 05 正式测试 | 第一独立测试 | 1.000000 | 0 | 221 | 0.801339 | 0.892857 |
| 06 正式测试 | 第二独立测试 | 1.000000 | 0 | 143 | 0.951128 | 0.981203 |

补充指标：

| 阶段 | ROC-AUC | AP | accept pair recall / top-50 | descriptor Recall@50 | false positives/query |
|---|---:|---:|---:|---:|---:|
| 600 对 02 sampled validation | 1.000000 | 1.000000 | 1.000000 | N/A | sampled: 0.001802 |
| 00 正式训练池 | 0.999119 | 0.997981 | N/A | N/A | N/A |
| 02 正式校准 | 0.998208 | 0.955278 | 0.696945 | N/A | N/A |
| 05 正式测试 | N/A | N/A | 0.513573 | 0.953125 | 0.000000 |
| 06 正式测试 | N/A | N/A | 0.698745 | 1.000000 | 0.000000 |

## 5. 05 正式独立测试

- pairs：131,625
- queries：2,657
- historical-positive queries：448
- retrieved-positive queries：427
- descriptor Recall@50：0.953125
- accept：3,040
- uncertain：12,399
- reject：116,186
- accepted positive：2,819
- accepted negative：0
- accepted ignore：221
- accept pair precision：1.0
- accept pair recall conditional top-50：0.5135726
- accept query recall conditional top-50：0.8407494
- accept query recall end-to-end：0.8013393
- reranked Recall@1 end-to-end：0.8928571
- false positives/query：0.0

05 上没有明确 negative 被 accept，但 descriptor 阶段有 21 个 historical-positive query 未在 top-50 找回正例。正式系统的安全 accept 比较保守。

## 6. 06 正式独立测试

- pairs：48,625
- queries：997
- historical-positive queries：266
- retrieved-positive queries：266
- descriptor Recall@50：1.0
- accept：1,980
- uncertain：6,090
- reject：40,555
- accepted positive：1,837
- accepted negative：0
- accepted ignore：143
- accept pair precision：1.0
- accept pair recall conditional top-50：0.6987448
- accept query recall conditional top-50：0.9511278
- accept query recall end-to-end：0.9511278
- reranked Recall@1 end-to-end：0.9812030
- false positives/query：0.0

06 上所有具有历史正例的 query 都在 top-50 找回了正例，冻结三态规则在保持明确 negative 零误接受的同时获得了更高召回。

## 7. 结果解读

1. **安全性在两个独立序列上保持一致。** 05 和 06 的明确 negative accept 均为 0，按当前 positive/negative 口径计算的 accept pair precision 均为 1.0。
2. **06 的召回明显高于 05。** 06 的 descriptor Recall@50、accept pair recall、端到端 accept query recall 和 reranked Recall@1 均高于 05。
3. **05 的瓶颈同时存在于检索和三态验证。** 21 个 historical-positive query 没有正例进入 top-50；进入 top-50 后仍有较多 positive pair 被分到 uncertain。
4. **ignore 必须单独解释。** 05/06 分别有 221/143 个 5–25 m 灰区样本被 accept。它们不计入当前 accept precision 的 false positive，但不能被隐藏或自动解释为安全回环。
5. **不能外推为普遍零误检。** 当前结论只覆盖 KITTI 05/06、当前距离标签和当前候选分布，不能直接代表其他序列、真实 SLAM 拓扑或更长时间运行。

## 8. 指标口径

- `accept_pair_precision = accepted_positive / (accepted_positive + accepted_negative)`；ignore 不进入分母。
- `accept_pair_recall_conditional_top50` 的分母是 top-50 pairs 中全部 positive。
- `accept_query_recall_conditional_top50` 的分母是 top-50 中实际出现 positive 的 query。
- `accept_query_recall_end_to_end` 的分母是全部 historical-positive query。
- `reranked_recall_at_1_end_to_end` 按每个 retrieval query 的最大 calibrated probability 候选计算，不要求该候选状态为 accept。
- `false_positives_per_query` 只统计明确 negative accept，分母为全部 retrieval query。

## 9. 冻结结论

当前实验闭环已经完成。正式冻结基线为：

- 00 全量 logistic decision head；
- 02 全量 Platt calibration 和三态规则；
- 05/06 不调参独立测试；
- 05/06 明确 negative accept 均为 0；
- 05/06 端到端 accept query recall 分别为 0.801339 和 0.951128。

后续可以分析 uncertain、增强 chunk 续跑可靠性或开发新方法，但不得覆盖本冻结版本。任何根据 05/06 结果形成的方法改动都必须建立新版本，并使用新的验证和独立测试数据。
