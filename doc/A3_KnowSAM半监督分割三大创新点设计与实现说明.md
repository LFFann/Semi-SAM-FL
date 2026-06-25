# 面向 A3 超声外侧裂半监督分割的 KnowSAM 框架级优化方案

更新时间：2026-04-23

框架名称建议：A3-RCP-KnowSAM，Anatomy-aware Reliability-Calibrated Prompt Co-evolution KnowSAM。

## 1. 核心判断

本方案不应被表述为“在 KnowSAM 上叠加若干 loss 或模块”。更合理的顶会级创新定位是：针对 KnowSAM 的关键框架瓶颈，重构 SAM 与 SGDL 的交互方式，使 baseline 从“单向提示 + 单向蒸馏”升级为“可靠性校准的提示-蒸馏闭环协同框架”。

KnowSAM baseline 的核心路径为：

```text
Input A3 image
  -> SGDL 生成 UNet / VNet / fusion_map
  -> fusion_map 直接作为 mask prompt 输入 SAM
  -> SAM 输出 mask
  -> SAM mask 作为教师，等权 KL 蒸馏给 UNet / VNet
```

该路径的主要问题不是缺少某个 loss，而是信息流存在结构性缺陷：

1. `fusion_map` 被直接当作 SAM prompt，但它本身可能包含半监督噪声。
2. SAM 被默认视为可靠教师，但超声边界低对比区域上 SAM 可能不稳定。
3. 伪标签被等价使用，没有利用 A3 外侧裂的解剖尺度与细结构特征做质量控制。

因此，本文档将创新点重新组织为三个框架级贡献。

## 2. 相关克隆模型的启发与边界

已克隆模型包括 `PH-Net`、`CPAC-SAM`、`CPC-SAM`、`SemiSAM`、`E-BayesSAM`、`AD-MT`、`ABD`、`DiffRect`、`BCSI`、`CMC`、`SFR`、`SAM-MedUS` 等。它们提供了若干研究趋势：

| 趋势 | 代表代码/论文 | 对本任务的启发 | 本方案差异 |
| --- | --- | --- | --- |
| SAM 辅助医学半监督 | SemiSAM、CPC-SAM、CPAC-SAM、SFR | SAM 应作为可交互先验，而不是普通分割头 | 本方案优化 KnowSAM 内部 prompt 与 distillation 的闭环，而不是复制 prompt consistency |
| 不确定性与教师冲突抑制 | E-BayesSAM、AD-MT | 教师质量必须被校准 | 本方案不新增贝叶斯分支或多教师 EMA，而是从 KnowSAM 已有三分支与 SAM 输出中估计可靠性 |
| 困难区域和伪标签质量 | PH-Net、DiffRect | 伪标签不能等权训练 | 本方案将样本质量绑定到 A3 解剖合理性与 prompt/teacher 可靠性 |
| 结构边界约束 | SAM-MedUS、ABD | 超声细结构边界需要被显式建模 | 本方案把边界约束作为外侧裂结构控制器，而不是孤立边界 loss |

差异化定位：这些工作大多改变网络结构、提示策略、教师更新或伪标签修正器。本方案的核心是重构 KnowSAM 的原有信息通路，使 SAM 与 SGDL 在 A3 超声任务中进行可靠性交互。

## 3. 框架创新一：RCP，可靠性校准的共识提示机制

### 3.1 Baseline 痛点

KnowSAM 原始框架将 `fusion_map` 直接送入 SAM prompt encoder：

```text
masks = fusion_map[:, class_id]
```

这个设计的隐含假设是 fusion 分支已经足够可靠。但在 A3 外侧裂半监督数据中，无标签样本多、边界弱、前景面积小，fusion_map 可能存在假阳性、断裂或局部膨胀。错误 prompt 会导致 SAM 输出进一步偏移，随后又作为教师反向蒸馏给 SGDL，形成错误闭环。

### 3.2 方法

RCP 将原始 raw fusion prompt 改为三分支共识 prompt。先由 UNet、VNet 和 fusion head 的概率输出构建共识：

```text
P_cons = Sharpen((P_unet + P_vnet + P_fusion) / 3)
```

再用共识熵和双分支前景分歧估计 prompt 可靠性：

```text
U_prompt(x) = [H(P_cons(x)) + |P_unet_fg(x) - P_vnet_fg(x)|] / 2
R_prompt(x) = clamp(exp(-alpha * U_prompt(x)), r_min, 1)
```

最终 prompt 不再强制相信某一分支，而是在高可靠区域使用共识概率，在低可靠区域退回均匀先验：

```text
P_prompt(x) = R_prompt(x) * P_cons(x) + [1 - R_prompt(x)] * Uniform
MaskPrompt(x) = logit(P_prompt(x))
```

### 3.3 顶会级创新性表达

RCP 不是简单换一个输入 mask，而是改变 KnowSAM 的核心框架关系：SAM 不再被 noisy fusion_map 被动驱动，而是由三分支一致性筛选后的可靠 prompt 驱动。这使得 baseline 从“单分支提示 SAM”变成“多分支共识校准提示 SAM”，属于信息流级别的框架优化。

### 3.4 代码实现

实现位置：

1. `trainer.py` 中新增 `build_a3_consensus_prompt`。
2. 训练和验证阶段均将 SAM prompt encoder 的输入由 `fusion_map` 替换为 `prompt_logits`。
3. 新增参数 `--rcp_alpha`、`--rcp_min_weight`、`--rcp_sharpen`。

## 4. 框架创新二：BRCE，可靠性闭环协同进化

### 4.1 Baseline 痛点

KnowSAM 原始蒸馏是单向的：

```text
SAM output -> equal-weight KL -> UNet / VNet
```

这会造成两个问题：

1. SAM 在超声低对比边界处的错误会等权传播。
2. SGDL 到 SAM 的 prompt 质量和 SAM 到 SGDL 的 teacher 质量没有被统一建模。

### 4.2 方法

BRCE 将 RCP 的 prompt 可靠性和 SAM 输出后的 teacher 可靠性合并，形成闭环协同：

```text
SGDL consensus -> R_prompt -> calibrated SAM prompt
SAM prediction + SGDL disagreement -> R_teacher
R_prompt + R_teacher -> pseudo-label quality
R_teacher weighted KD -> SGDL update
```

teacher 可靠性定义为：

```text
U_teacher(x) =
  [H(P_sam(x)) + H(P_fusion(x)) + |P_unet_fg(x)-P_vnet_fg(x)|] / 3
R_teacher(x) = clamp(exp(-beta * U_teacher(x)), t_min, 1)
```

蒸馏损失从全局等权 KL 改为：

```text
L_BRCE =
  sum_x R_teacher(x) * KL(P_sam(x) || P_student(x))
  / sum_x R_teacher(x)
```

### 4.3 顶会级创新性表达

BRCE 的关键不是“加权 KD”本身，而是将 KnowSAM 的两段交互统一为一个可靠性闭环：SGDL 先以可靠 prompt 影响 SAM，SAM 再以可靠 teacher 反向影响 SGDL。这个闭环将 SAM 从静态 teacher 改造成被任务数据动态校准的协同参与者，更符合 foundation model 与 task-specific model 协同优化的最新趋势。

### 4.4 代码实现

实现位置：

1. `utils/losses.py` 中新增 `WeightedKDLoss`。
2. `trainer.py` 中新增 `build_reliability_map`。
3. 原始 `KDLoss(pred_UNet, pred_sam)` 和 `KDLoss(pred_VNet, pred_sam)` 已替换为可靠性加权蒸馏。
4. 训练日志新增 `prompt_weight_mean`、`uckd_weight_mean`，便于汇报闭环可靠性变化。

## 5. 框架创新三：AQC，解剖质量控制的伪标签课程

### 5.1 Baseline 痛点

KnowSAM 的无标签训练依赖伪标签和 MixUp，但并未区分伪标签质量。A3 外侧裂的结构特点非常明确：目标细长、面积占比小、边界弱。若伪标签为空、明显膨胀或与分支预测冲突，仍然等权参与训练，会降低半监督收益。

### 5.2 方法

AQC 不再把伪标签质量视为后处理，而是作为框架中的训练调度器。样本质量同时由三部分决定：

1. prompt 阶段可靠性 `R_prompt`：该样本是否能给 SAM 提供可信提示。
2. teacher 阶段可靠性 `R_teacher`：SAM 输出是否适合蒸馏。
3. A3 解剖面积合理性：前景面积是否落在宽松外侧裂范围。

```text
Q_i =
  clamp(
    0.5 * mean(R_prompt_i) + 0.5 * mean(R_teacher_i),
    q_min,
    1
  )
  * AnatomyScore_i
```

其中：

```text
AnatomyScore_i = exp(-25 * area_violation_i)
```

在 MixUp 的无标签区域，使用样本质量加权 CE+Dice：

```text
L_AQC =
  sum_i Q_i * [CE_i + Dice_i] / sum_i Q_i
```

同时，AQC 使用外侧裂结构控制器约束边界和面积：

```text
L_boundary = |Sobel(P_fg) - Sobel(Y_fg)|
L_area = ReLU(lower - area(P_fg))^2 + ReLU(area(P_fg) - upper)^2
```

### 5.3 顶会级创新性表达

AQC 不是简单的 pseudo-label filtering。它把 A3 外侧裂的任务知识显式引入半监督训练调度，让“是否相信无标签样本”由 prompt 可靠性、teacher 可靠性和解剖合理性共同决定。相较 PH-Net 的困难 patch、DiffRect 的扩散校正、AD-MT 的教师冲突抑制，本方案更轻量，并且与 KnowSAM 的 SAM-SGDL 信息流强耦合。

### 5.4 代码实现

实现位置：

1. `trainer.py` 中新增 `compute_a3_quality_weight` 和 `weighted_segmentation_loss`。
2. `mix_up` 增加 `unlabeled_quality_weight`。
3. `utils/losses.py` 中新增 `BoundaryLoss` 和 `soft_area_prior_loss`。
4. 新增参数 `--qapl_min_weight`、`--sap_boundary_weight`、`--sap_shape_weight`、`--sap_area_lower`、`--sap_area_upper`。

## 6. 最终框架流程

```text
A3 ultrasound image
  -> SGDL produces UNet / VNet / fusion predictions
  -> RCP builds reliability-calibrated consensus prompt
  -> SAM receives calibrated prompt and predicts masks
  -> BRCE estimates SAM teacher reliability
  -> AQC estimates sample-level pseudo-label quality
  -> reliability-weighted distillation + quality-weighted MixUp + anatomy control
  -> updated SAM adapters and SGDL branches
```

该流程的本质变化是：

1. prompt 输入被重构。
2. teacher 蒸馏被重构。
3. 无标签训练调度被重构。
4. 三者共享可靠性变量，而不是互相独立的模块。

## 7. 论文式贡献写法

建议在汇报或论文中使用如下表达：

1. We propose an anatomy-aware reliability-calibrated prompt co-evolution framework for semi-supervised A3 ultrasound lateral fissure segmentation, which upgrades KnowSAM from raw fusion prompting to consensus-calibrated SAM interaction.
2. We design a bidirectional reliability loop between the SGDL branches and SAM, where SGDL provides uncertainty-aware prompts to SAM and SAM returns reliability-weighted knowledge to the task-specific branches.
3. We introduce an A3 anatomical quality controller that couples prompt reliability, teacher reliability, and lateral-fissure scale priors to schedule pseudo-label learning under scarce annotations.

中文表述：

1. 提出面向 A3 超声外侧裂半监督分割的解剖感知可靠性校准提示协同框架，将 KnowSAM 从原始 fusion-map 提示升级为多分支共识校准提示。
2. 构建 SGDL 与 SAM 的双向可靠性闭环，使 SGDL 以可靠提示驱动 SAM，SAM 再以可靠教师信号反向优化 SGDL。
3. 设计 A3 解剖质量控制器，联合 prompt 可靠性、teacher 可靠性和外侧裂尺度先验，对无标签伪监督进行课程化调度。

## 8. 消融实验设计

| 编号 | 设置 | 验证目的 |
| --- | --- | --- |
| A0 | 原始 KnowSAM | baseline |
| A1 | A0 + RCP | 验证 prompt 信息流优化是否有效 |
| A2 | A0 + BRCE | 验证可靠性闭环蒸馏是否优于等权 KD |
| A3 | A0 + AQC | 验证解剖质量控制是否提升伪标签学习 |
| A4 | A0 + RCP + BRCE | 验证 prompt 可靠性和 teacher 可靠性的闭环协同 |
| A5 | 完整 A3-RCP-KnowSAM | 验证完整框架收益 |

建议指标：

1. Dice / IoU：整体分割质量。
2. HD95 / ASSD：边界偏差。
3. Precision / Recall：误检和漏检。
4. Foreground area ratio：是否减少空预测与异常膨胀。
5. `prompt_weight_mean`、`uckd_weight_mean`、`qapl_quality_mean`：展示框架中的可靠性变量如何随训练变化。

## 9. 当前代码修改记录

| 文件 | 框架级修改 |
| --- | --- |
| `trainer.py` | 新增 RCP 共识提示；训练和验证均改用 `prompt_logits` 输入 SAM；新增 BRCE 可靠性蒸馏；新增 AQC 样本质量调度 |
| `utils/losses.py` | 新增可靠性加权 KD、边界控制器、软面积先验 |
| `train_semi_SAM.py` | 新增 RCP/BRCE/AQC 参数，并将损失文件纳入实验快照 |

## 10. 汇报建议

PPT 中不建议按“模块一、模块二、模块三”展示，而应按 baseline 框架缺陷到新框架闭环展示：

1. 第一页：A3 外侧裂任务特性，说明仅处理 A3 切面。
2. 第二页：KnowSAM baseline 信息流图，突出 raw fusion prompt 和 equal KD。
3. 第三页：指出 baseline 的三处结构瓶颈。
4. 第四页：展示 A3-RCP-KnowSAM 总框架图。
5. 第五页：RCP 如何替代 raw fusion prompt。
6. 第六页：BRCE 如何形成 SGDL-SAM 可靠性闭环。
7. 第七页：AQC 如何调度伪标签并注入解剖先验。
8. 第八页：实验设置、A3-only 数据划分和日志结果。
9. 第九页：消融实验和预期指标。
10. 第十页：贡献总结与后续对比实验计划。
