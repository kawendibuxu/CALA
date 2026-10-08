# 冻结 accept 之后的时间聚合规则 v2

本规则是后处理，不改变 00 训练、02 校准、05/06/07 冻结三态评估和原单一窗口关联报告。

## 输入

- 冻结 VGGT 三态 chunks
- 冻结 accept-only 单一窗口选择

## 规则

1. 已有 `accept` 关联原样保留。
2. 仅对当前弃权的 query，查看 ±3 个相邻 query 中已经接受的历史窗口。
3. 历史簇：候选中心帧差 ≤ 5。
4. 当前 query 在该簇中须存在非 `reject`、且通过冻结 vis / track / residual 门控的候选。
5. 用与原协议相同的 `raw_logit` 并列规则选一个窗口。
6. 不降低 accept 概率阈值，不改 5m/25m 标签，不改 top-50。

## 指标

- 主指标：`association_precision_strict`（ignore 算错误关联）
- 硬约束：明确 negative 关联必须为 0
- 次指标：clear-label precision、条件召回、端到端召回、新增 ignore 条数
