# UniPR-3D + VGGT 多帧安全回环检测实验记录

## 0. 目标与总体思路

实验目标是把 **UniPR-3D 的多帧全局检索能力**扩展为更安全的回环检测系统：

```text
KITTI RGB 多帧序列
    ↓
冻结 UniPR-3D 提取全局 descriptor
    ↓
historical top-50 retrieval
    ↓
冻结 VGGT 做 cross-token 显式 3D 验证
    ↓
decision head 输出回环概率
    ↓
校准为 accept / uncertain / reject
    ↓
在未参与训练的序列上端到端评估
```

核心原则：

- UniPR-3D 和 VGGT **全部保持冻结**。
- 只训练新增的轻量 **decision head**。
- 不修改或微调原 UniPR 的 LoRA、多帧路径和 SALAD descriptor 模块。
- `05/06` 作为独立测试集，标签不能反向用于修改模型、阈值或校准器。

---

## 1. 数据划分

| 序列 | 角色 | 用途 |
|---|---|---|
| `00` | 正式训练集 | 训练 decision head |
| `02` | 验证与校准集 | Platt 校准、选择三态规则 |
| `05` | 第一独立测试集 | 仅最终评估 |
| `06` | 第二独立测试集 | 仅最终评估 |

> **重要：** `05/06` 的标签不能用于模型训练、阈值修改或校准，否则不再属于独立测试。

---

## 2. 构建多帧输入

每个 KITTI 样本以中心帧为基准，构造长度为 5 的 RGB 序列：

```text
center = t
frames = [t, t-2, t-1, t+1, t+2]
```

输入设置：

- 输入类型：KITTI `image_2` 彩色图像
- 输入帧数：5
- 输入尺寸：`392 × 518`
- 中心帧：`t`

KITTI pose 和 calibration **不作为 VGGT 输入**，只用于：

1. 离线生成监督标签；
2. 生成实验分析结果；
3. 计算真实几何距离。

---

## 3. 冻结 UniPR-3D 提取 descriptor

每个 5 帧 RGB 序列输入原 UniPR 多帧模型：

```text
sequence_images
    ↓
UniPR LoRA multi-frame model
    ↓
SALAD global descriptor
```

得到每个中心帧的高维全局 descriptor。

已有 descriptor 产物：

```text
outputs/kitti_00_seq5_stride1_unipr_descriptors.pt
outputs/kitti_02_seq5_stride1_unipr_descriptors.pt
outputs/kitti_05_seq5_stride1_unipr_descriptors.pt
```

这一阶段只负责 **coarse retrieval**，不负责最终几何判定。

---

## 4. Historical Top-50 检索与标签生成

### 4.1 历史候选约束

对于 query 帧 `t`，只在历史数据库中检索：

```text
candidate_center <= query_center - 100
```

即：

```text
temporal_gap >= 100
```

这样可以避免：

- 相邻帧被误认为回环；
- 短时间重复观测被误认为真实回环。

### 4.2 Top-50 检索

使用 UniPR descriptor 的 cosine similarity：

```text
query descriptor
    ↓
historical database
    ↓
cosine similarity
    ↓
Top-50 candidates
```

### 4.3 GT 标签

利用 KITTI GT pose 的平面位置距离生成监督标签：

```text
distance <= 5 m
    → positive：真实回环

distance >= 25 m
    → negative：明确非回环

5 m < distance < 25 m
    → ignore：边界 / 灰区样本

temporal_gap < 100
    → 不进入检索候选
```

这套标签只用于：

- decision head 训练；
- 校准；
- 测试指标计算。

推理阶段 decision head **不会看到**：

- `gt_distance_m`
- pose
- ground-truth label

---

## 5. 00 序列 Descriptor-only 基线

00 完整 Top-50 的规模：

```text
总对数：220,625
positive：11,985
negative：183,100
ignore：25,540
```

05 的 descriptor-only 基线：

```text
Descriptor Recall@50 = 95.31%
```

含义：

- 共有 448 个具有历史真实回环的 query；
- 其中 427 个至少在 Top-50 中找回一个正候选；
- 21 个真实回环没有进入 Top-50。

因此：

> 后续几何验证无法找回这 21 个没有进入 Top-50 的回环，这构成 coarse retrieval 阶段的理论上限。

---

## 6. Cross-token 显式 3D 几何验证

对于 Top-50 中的每一对：

```text
query 5-frame RGB
        +
candidate 5-frame RGB
        ↓
frozen VGGT joint inference
        ↓
track / point / depth / camera outputs
        ↓
token-level 3D correspondence
        ↓
3D geometric verification
```

### 6.1 VGGT 输入与输出

VGGT 输入仍然是 RGB，不需要外部 depth。

VGGT 内部预测或输出几何表示，包括：

- 跨图 track
- world point
- depth
- depth confidence
- point confidence
- visibility
- camera pose encoding

### 6.2 Query token 采样

从 query 中心帧选取固定网格 token，例如：

```text
12 × 16 = 192 patch/token
```

每个 query token：

```text
query token
    ↓
VGGT track
    ↓
candidate 的 5 帧
    ↓
对应像素位置
    ↓
采样 3D point / depth / confidence / visibility
```

### 6.3 加权 3D RANSAC / Kabsch

对 token correspondence 执行：

```text
query token 3D points
        ↕
candidate tracked token 3D points
        ↓
estimated rigid transform
        ↓
3D residuals
        ↓
inlier mask
```

### 6.4 Pair-level 特征

最终每个 query-candidate pair 压缩为以下特征：

| 特征 | 含义 |
|---|---|
| `descriptor_similarity` | UniPR coarse retrieval 相似度 |
| `weighted_3d_inlier_ratio` | 根据 track、visibility、point/depth confidence 加权后的 3D 内点比例 |
| `median_3d_residual` | 典型 3D 对齐残差，越低越好 |
| `p90_3d_residual` | 残差长尾质量，防止少量坏 token 被中位数掩盖 |
| `mean_track_confidence` | 参与 3D 内点的跨序列 track 置信度 |
| `mean_visibility` | token 在 candidate 多帧中的可见性 |
| `temporal_consistency` | token 几何支持是否跨 candidate 5 帧稳定 |
| `camera_rotation_consistency_deg` | VGGT camera 输出的跨帧旋转一致性 |
| `camera_translation_consistency` | VGGT camera 输出的跨帧平移一致性 |

这里称为 **cross-token 3D**，是因为它不是简单比较两个全局 descriptor，而是：

```text
query patch token
    ↓
candidate patch token
    ↓
跨 candidate 多帧的 track 对齐
    ↓
point / depth
    ↓
显式 3D 一致性验证
```

---

## 7. 600 对可行性基线实验

最初从 00 和 02 各自抽取：

```text
100 positive
+
500 rank <= 10 negative
```

用于验证 cross-token 3D 特征是否具备区分能力，并训练小样本 baseline head。

### 7.1 为什么不是最终正式模型

600 对样本存在明显局限：

- 样本量太小；
- 负例主要来自 rank 1–10；
- 无法覆盖完整 Top-50 的真实负例分布；
- 正负比例与真实部署比例差异很大。

因此：

> 600 对模型的原始输出不能解释为真实回环概率。

### 7.2 05 小样本基线结果

旧版冻结模型在 05 的端到端基线中得到：

```text
accept pair precision = 100%
accept query recall end-to-end = 83.93%
reranked Recall@1 = 89.06%
明确 negative accept 数 = 0
```

说明：

> Cross-token 3D 验证 + decision head 的整体方案具有可行性。

但上述结果仅对应 **600 对小样本训练 baseline**，不是最终正式实验结果。

---

## 8. 00 全量特征提取

正式训练前，已经对 00 的全部 Top-50 pairs 提取 frozen VGGT cross-token 3D 特征：

```text
outputs/kitti_00_top50_cross_token_3d_full/chunks/
```

完整结果：

```text
2207 / 2207 chunks
220,625 / 220,625 pairs
```

数据检查：

- 数据完整；
- 索引连续；
- 无重复。

### 8.1 正负特征分布

例如：

```text
mean_track_confidence:
positive median = 0.633
negative median = 0.0025

mean_visibility:
positive median = 0.387
negative median = 0.021

median_3d_residual:
positive median = 0.019 m
negative median = 0.272 m
```

可以看到正负样本在 3D 几何证据上具有明显差异。

灰区样本通常介于正负样本之间，因此：

> `5 m < distance < 25 m` 的 ignore 样本不适合直接强制视为 negative。

---

## 9. 正式 00 Decision Head 训练

### 9.1 模型选择

正式 decision head 使用：

```text
Logistic Regression
```

而不是 MLP。

原因：

1. 当前只有一个训练序列；
2. 线性 head 更稳定；
3. 更容易解释；
4. 降低对 00 场景特性的过拟合风险；
5. 当前目标主要是验证已有 UniPR + VGGT 几何特征的判别能力。

### 9.2 训练特征

正式训练使用：

```text
[
    descriptor_similarity,
    weighted_3d_inlier_ratio,
    median_3d_residual,
    p90_3d_residual,
    mean_track_confidence,
    mean_visibility,
    temporal_consistency
]
```

### 9.3 负例采样策略

不是直接把全部 183,100 个 negative 与 positive 混合进行 BCE。

正式负例池：

```text
所有 11,985 个 positive 保留

+

rank 1-10 negative
作为 hard negatives

+

rank 11-50
每个 rank 抽取 300 个普通 negatives
```

最终：

```text
negative pool = 47,865 pairs
```

每个 batch 固定：

```text
16 positive + 48 negative
```

即：

```text
positive : negative = 1 : 3
```

这样可以：

- 保持稳定的正例梯度；
- 保留 rank 1–50 的真实候选覆盖；
- 避免大量 easy negative 主导训练。

`ignore` 不进入 BCE，因为 5–25 m 的几何状态本身具有模糊性。

### 9.4 正式模型产物

训练完成：

```text
outputs/kitti_cross_token_3d_full_decision_head_00/decision_head.pt
```

训练池内结果：

```text
ROC-AUC = 0.9991
AP = 0.9980
```

注意：

> 这是 00 训练池上的拟合检查，不代表泛化性能。

真正的泛化结论必须来自：

```text
02
05
06
```

---

## 10. 02 全量特征提取与校准

当前阶段正在执行 02 完整 Top-50 的 frozen VGGT 特征提取：

```text
outputs/kitti_02_top50_cross_token_3d_full/chunks/
```

02 数据规模：

```text
226,625 pairs
2267 chunks
```

服务器中断后，已从第 2048 个 chunk 后恢复。

### 10.1 下一步

02 特征提取完成后：

```text
冻结 00 decision head
        ↓
02 完整 Top-50 特征
        ↓
weighted Platt scaling
        ↓
calibrated P(loop | retrieval, geometry)
        ↓
固定 accept / uncertain / reject 规则
```

### 10.2 为什么需要 Platt Calibration

正式 decision head 的训练 batch 使用人为固定的：

```text
positive : negative = 1 : 3
```

但真实 Top-50 部署分布中：

```text
positive << negative
```

因此未经校准的：

```text
score = 0.8
```

只能理解为：

> 在训练采样分布下，这个样本很像 positive。

不能直接理解为：

> 这个候选有 80% 的真实回环概率。

因此需要在 02 上进行 Platt calibration，将 decision score 映射为更符合真实部署分布的概率：

```text
decision score
    ↓
weighted Platt scaling
    ↓
calibrated P(loop)
```

### 10.3 校准权重

校准时需要按照：

```text
(candidate_rank, label)
```

进行部署分布加权，避免 rank 分层采样改变概率含义。

---

## 11. 三态回环判定

最终不采用简单的：

```text
accept / reject
```

而采用：

```text
accept
uncertain
reject
```

### 11.1 Accept

需要同时具备较强的综合证据：

```text
高 calibrated probability
+
足够 track confidence
+
足够 visibility
+
足够低的 3D residual
```

→ `accept`

### 11.2 Reject

如果：

```text
低 calibrated probability
+
低 track confidence
+
低 visibility
```

→ `reject`

### 11.3 Uncertain

如果出现：

- 概率处于边界区域；
- 几何证据相互冲突；
- track confidence 不足；
- visibility 不足；
- 3D residual 较高；
- 当前证据不足以安全接受或拒绝；

→ `uncertain`

`uncertain` 并不是错误类别，而表示：

> 当前 5 帧 RGB 的证据不足以让 SLAM 前端安全接受或强制拒绝。

实际系统可以将 `uncertain` 交给：

- 后端图优化；
- 更多帧验证；
- 里程计一致性检查；
- 延迟决策。

### 11.4 阈值原则

三态阈值：

```text
只能在 02 上确定
```

确定后：

```text
冻结
```

不能再根据 05/06 的结果调参。

---

## 12. 最终独立测试

02 校准完成后，形成一套冻结产物：

```text
formal decision head
+
Platt calibration parameters
+
accept / reject probability thresholds
+
geometry evidence gates
```

之后不再修改任何模型和阈值。

### 12.1 05 第一独立测试

```text
05 full top-50
    ↓
frozen cross-token 3D verification
    ↓
frozen calibrated three-way decision
    ↓
final report
```

### 12.2 06 第二独立测试

```text
06 descriptor extraction
    ↓
06 historical top-50
    ↓
frozen cross-token 3D verification
    ↓
same frozen calibrated three-way decision
    ↓
second independent final report
```

### 12.3 为什么 05 必须重新跑

此前 05 使用的是：

```text
600 对小样本 baseline head
```

现在正式实验使用：

```text
00 全量 decision head
+
02 Platt calibration
+
02 确定的三态阈值
```

两套模型和决策规则不同。

因此：

> **05 必须重新进行正式评估，不能复用旧的 05 baseline 结果作为最终实验结果。**

---

## 13. 最终实验指标

最终至少分为四个层次进行报告。

### 13.1 Retrieval Level

```text
Descriptor Recall@1
Descriptor Recall@5
Descriptor Recall@10
Descriptor Recall@20
Descriptor Recall@50

历史正例 query 数
Top-50 未召回正例数
```

重点回答：

> UniPR-3D 的 coarse retrieval 能否把真实回环召回到 Top-K？

---

### 13.2 Geometry / Pair Level

```text
ROC-AUC
AP

accept precision
accept recall

明确 negative 被 accept 的数量

ignore 被 accept / uncertain / reject 的分布
```

重点回答：

> VGGT cross-token 3D 几何证据能否有效区分真实回环和非回环？

---

### 13.3 Query Level

```text
accept query recall conditional on top-50 retrieval
end-to-end accept query recall
reranked Recall@1
false positives per query
```

重点回答：

> 从 query 级别来看，整个系统最终能安全发现多少真实回环？

其中：

```text
conditional recall
```

主要衡量：

> 已经进入 Top-50 的真实回环，有多少最终被接受。

而：

```text
end-to-end recall
```

衡量：

> 从原始 query 开始，到最终 accept，整个系统能找回多少真实回环。

---

### 13.4 Uncertainty Level

```text
accept / uncertain / reject 比例

positive 在三态中的分布
negative 在三态中的分布
ignore 在三态中的分布

uncertain 覆盖的真实正例比例
uncertain 覆盖的 hard negative 比例
```

重点回答：

> 三态决策是否真的能够把“证据不足”的样本安全地隔离出来，而不是强制二分类。

---

## 14. 当前实验状态

整个实验链路已经完成的部分：

```text
✓ 多帧 RGB 输入构建
✓ UniPR-3D 冻结 descriptor 提取
✓ Historical Top-50 retrieval
✓ GT 标签生成
✓ Cross-token 3D 验证方案实现
✓ 600 对 feasibility baseline
✓ 00 全量 Top-50 几何特征提取
✓ 00 正式 decision head 训练
```

当前正在进行：

```text
→ 02 全量 Top-50 cross-token 3D 特征提取
```

---

## 15. 最终剩余工作

最终实验链路还差三大阶段：

### 阶段 A：完成 02

```text
[进行中]
02 全量特征提取
    ↓
weighted Platt calibration
    ↓
确定三态阈值
    ↓
冻结 calibration + thresholds
```

### 阶段 B：正式评估 05

```text
05 full top-50
    ↓
frozen VGGT cross-token 3D
    ↓
00 formal decision head
    ↓
02 frozen calibration
    ↓
02 frozen three-way thresholds
    ↓
最终 05 结果
```

### 阶段 C：完成 06

```text
06 descriptor extraction
    ↓
historical top-50
    ↓
frozen VGGT cross-token 3D
    ↓
frozen decision head
    ↓
frozen calibration
    ↓
frozen three-way thresholds
    ↓
最终 06 结果
```

---

## 16. 最终实验闭环

最终论文/实验应形成如下完整链路：

```text
                  ┌─────────────────────┐
                  │   KITTI RGB 5-frame │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │  Frozen UniPR-3D   │
                  │  SALAD Descriptor   │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Historical Top-50   │
                  │ Coarse Retrieval    │
                  └──────────┬──────────┘
                             ↓
              ┌──────────────────────────────┐
              │ Frozen VGGT Cross-token 3D  │
              │ Track + Point + Depth       │
              │ + Visibility + Camera       │
              └──────────────┬───────────────┘
                             ↓
                  ┌─────────────────────┐
                  │  Pair-level Feature │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Decision Head (00)  │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Platt Calibration   │
                  │       (02)          │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Three-way Decision  │
                  │ Accept / Uncertain  │
                  │ / Reject            │
                  └──────────┬──────────┘
                             ↓
                    ┌────────┴────────┐
                    ↓                 ↓
                 KITTI 05          KITTI 06
               Independent        Independent
                  Test 1             Test 2
```

最终实验的核心思想可以概括为：

> **UniPR-3D 负责“把可能的回环召回来”，VGGT cross-token 3D 负责“验证这些候选在显式 3D 几何上是否真的一致”，decision head 负责融合检索与几何证据，Platt calibration 将模型分数转换为更可靠的回环概率，最终通过 accept / uncertain / reject 三态机制提高回环检测的安全性。**
