# KnowSAM 解析文档

## 1. 项目定位

KnowSAM 是一个面向半监督医学图像分割的框架。它并不是简单地将 Segment Anything Model, SAM, 直接用于医学图像分割，而是将 SAM 与两个经典分割网络 UNet 和 VNet 结合起来，形成一个双学生加一个可学习提示分支的协同训练框架。

从当前仓库实现来看，KnowSAM 的核心目标是：

- 在只有少量标注数据的情况下，用无标签数据提升分割性能。
- 通过 learnable prompt 让 SAM 更适配医学图像场景。
- 通过 SAM 反向蒸馏 UNet 和 VNet，使传统分割网络获得更强的结构先验。
- 通过双分支一致性、熵约束和后期混合训练，提高无标签数据利用效率。

当前实现的核心训练逻辑主要位于：

- `Model/model.py`
- `Model/prompt.py`
- `Model/discriminator.py`
- `Model/sam/build_sam.py`
- `trainer.py`
- `utils/losses.py`

---

## 2. 整体框架概览

KnowSAM 可以理解为三个模块共同工作：

1. `SGDL` 主干分割模块
2. `SAM` 提示分割模块
3. `Learnable Prompt` 模块

训练时输入一张图像后，会经过两条主要路径：

- 路径 A：输入图像进入 `UNet + VNet`，得到两个分割结果和一个融合结果。
- 路径 B：同一张图像进入 SAM 的图像编码器，同时使用 learnable prompt 和融合分割图作为提示，生成 SAM 的分割结果。

随后：

- `UNet/VNet` 的输出通过融合头得到最终融合图 `fusion_map`
- `fusion_map` 又作为 mask prompt 反向输入 SAM
- `SAM` 的预测结果再蒸馏给 `UNet/VNet`

因此整个框架形成了一个闭环：

- 传统分割网络为 SAM 提供先验提示
- SAM 为传统分割网络提供蒸馏监督

这也是 KnowSAM 与“仅用 SAM 做零样本分割”最大的不同点。

---

## 3. 网络架构设计

## 3.1 SGDL 主干：UNet + VNet + 融合头

在 `Model/model.py` 中，`KnowSAM` 模型内部包含三个核心子模块：

- `UNet`
- `VNet`
- `Discriminator`

其中：

- `UNet` 是标准 2D 编码器解码器结构，偏向局部纹理建模。
- `VNet` 虽然最初常用于体数据，但这里使用的是 2D 卷积版本，结构更偏深层级特征与残差式传播。
- `Discriminator` 并不是 GAN 意义上的判别器，更准确地说，它是一个融合网络，用来综合两个分支的预测信息。

### UNet 分支

UNet 分支的特点：

- 五层编码器解码器结构
- 编码端通道逐层为 `16, 32, 64, 128, 256`
- 解码端通过跳跃连接恢复空间信息
- 输出 `pred_UNet`

它更偏向捕捉局部结构与边缘细节。

### VNet 分支

VNet 分支的特点：

- 同样使用逐层下采样和上采样
- 使用单独的卷积块与反卷积恢复空间分辨率
- 输出 `pred_VNet`

与 UNet 并行存在的意义不是简单集成，而是制造结构差异，让两个网络在无标签数据上形成互补预测。

### 融合头 Discriminator

`Discriminator` 的输入不是简单的 `pred_UNet` 与 `pred_VNet` 拼接，而是三类信息：

- `ambiguous_area`：两个网络二值预测不一致的区域
- `uncertainty_area`：两个网络各自预测熵图组成的不确定性信息
- `pred_logits`：两个网络原始 logits

然后通过卷积网络进行融合，输出最终分割图 `fusion_map`。

它的意义在于显式学习：

- 哪些位置两个网络冲突
- 哪些位置它们都不确定
- 哪些信息更值得信任

因此 `fusion_map` 比简单平均更有表达力。

---

## 3.2 SAM 分支

KnowSAM 并没有直接使用原始 SAM 推理，而是把 SAM 放进训练图中，作为一个可微训练分支参与训练。

SAM 部分由以下组件组成：

- `image_encoder`
- `prompt_encoder`
- `mask_decoder`
- `super_prompt`

关键点在于：

- SAM 的主干参数基本冻结
- 只训练 `Adapter` 参数和 `super_prompt` 参数

这是一种参数高效微调策略。

### 冻结与可训练部分

在 `trainer.py` 中，代码只允许以下参数更新：

- 参数名包含 `Adapter`
- 参数名包含 `super_prompt`

其余 SAM 参数被冻结。

这意味着 KnowSAM 并不是重训一个完整 SAM，而是：

- 保留 SAM 的通用视觉先验
- 用少量可学习参数让它适应医学图像分割任务

---

## 3.3 Learnable Prompt 设计

在 `Model/prompt.py` 中定义了 `Super_Prompt`。

它的核心作用是：

- 不使用人工手工框或点
- 直接从特征中学习出每个类别对应的 box embedding

具体做法是：

- 对每个类别都建立一个单独的 `box_decoder_embedding`
- 通过卷积、下采样、全局池化和前馈网络，输出对应的 box embedding
- 这些 embedding 送入 SAM 的 `prompt_encoder`

所以这里的 prompt 不是人工指定的，而是模型从数据中自动学出来的。

这就是论文标题中 “Learnable Prompting” 的落地实现。

---

## 3.4 SAM 的提示来源

KnowSAM 中，SAM 的提示来自两个来源：

### 1. Learnable box prompt

由 `Super_Prompt` 产生的 `boxes_embedding`

### 2. Dense mask prompt

由 `fusion_map` 插值到低分辨率后，作为稠密 mask prompt 输入 SAM

也就是说，SAM 在训练中接收到的并不是人工提示，而是：

- 来自主干特征学习得到的框提示
- 来自融合分割结果的掩码提示

这使得 SAM 能在半监督任务中与分割主干形成深度耦合。

---

## 4. 训练流程解析

训练主流程位于 `trainer.py` 的 `train()` 函数中。

一次前向传播大致分为以下步骤：

1. 输入图像进入 SAM 的 `image_encoder`
2. 输入图像同时进入 `SGDL`，得到：
   - `pred_UNet`
   - `pred_VNet`
   - `pred_UNet_soft`
   - `pred_VNet_soft`
   - `fusion_map`
3. `fusion_map` 经 softmax 得到 `fusion_map_soft`
4. `image_embeddings` 输入 `super_prompt`，得到每个类别的 `boxes_embedding`
5. `fusion_map` 的每一类作为稠密掩码提示输入 SAM 的 `prompt_encoder`
6. SAM 解码得到 `pred_sam`
7. 对 `pred_sam`, `pred_UNet`, `pred_VNet`, `fusion_map` 计算各类损失
8. 分别更新：
   - SAM 可训练部分
   - SGDL 分支

训练中同时维护两套优化器：

- `optimizer_sam`
- `optimizer_SGDL`

这说明作者明确把 “SAM 提示分支” 和 “传统分割主干分支” 看作两个可耦合但相对独立的优化对象。

---

## 5. 损失函数设计

KnowSAM 的损失可以拆成五大类。

## 5.1 有标签监督损失

对于有标签样本，模型对以下输出计算监督：

- `fusion_map`
- `pred_UNet`
- `pred_VNet`
- `pred_sam`

监督形式是：

- Cross Entropy Loss
- Dice Loss

具体代码形式为：

- `ce_loss(...) + dice_loss(...)`

这部分只对 batch 中前 `labeled_bs` 个样本使用。

因此，有标签监督是整个框架的基础锚点。

---

## 5.2 双分支一致性损失

KnowSAM 用 `UNet` 和 `VNet` 构成双学生结构。

在无标签样本上，这两个学生不能各自随意漂移，因此作者加入了一致性约束：

- `UNet_cons_loss`
- `VNet_cons_loss`

这两个损失由 `loss_diff1` 和 `loss_diff2` 实现，本质上是两个分支之间的 BCE 一致性约束。

它的目标是：

- 让两个结构不同的网络在相同无标签图像上产生相近结果
- 从而减少无标签学习中的不稳定性

### 注意

当前仓库实现中，`loss_diff1/2` 的 `for` 循环写法会覆盖前一类的结果，更像是只保留了最后一个类别通道的 BCE。这一点从实现角度看是可疑的，正式复现实验时建议再次确认是否与论文原始代码一致。

---

## 5.3 熵最小化损失

KnowSAM 对 `UNet` 与 `VNet` 的输出还分别施加了熵最小化约束：

- `UNet_enp_loss`
- `VNet_enp_loss`

含义是：

- 在无标签图像上鼓励预测分布更尖锐
- 降低不确定性
- 使伪监督更稳定

这类损失在半监督学习中很常见，本质上是让模型在无标签样本上尽量“做出明确判断”。

---

## 5.4 SAM 蒸馏损失

蒸馏损失是 KnowSAM 的核心之一。

当前实现中：

- `pred_sam` 被作为 teacher 信号
- `pred_UNet` 和 `pred_VNet` 作为 student

通过 `KDLoss` 进行 KL 蒸馏：

- `UNet_kd_loss`
- `VNet_kd_loss`

蒸馏的意义是：

- 利用 SAM 更强的通用视觉先验指导传统分割网络
- 让 UNet 和 VNet 的输出分布向 SAM 靠拢

因此论文标题中的 “SAM-induced Knowledge Distillation” 在代码里就是通过这两个 KL 蒸馏项实现的。

---

## 5.5 后期 uncertainty-guided mixup

当训练迭代数超过 `mixed_iterations` 后，会启用 `mix_up()`。

这部分机制比较有特点：

1. 用当前预测的熵图找出高不确定区域
2. 在 patch 级别选择这些区域
3. 用 labeled 图像和 unlabeled 图像局部拼接
4. 对拼接后的图像和标签再训练

这里会构造：

- `volume_batch_mix`
- `label_batch_mix`

标签来源是：

- labeled 区域用真实标签
- unlabeled 区域用伪标签

这相当于做了一次 uncertainty-aware 的 patch mixing，使得模型在难区域上获得更丰富的监督信号。

---

## 6. 有标签与无标签数据设计

## 6.1 数据目录结构

当前 2D 图像任务中，代码要求的数据结构为：

```text
dataset_root/
  labeled/
    image/
    mask/
  unlabeled/
    image/
    mask/
  val/
    image/
    mask/
```

其中：

- `labeled` 用于有监督训练
- `unlabeled` 用于无监督或半监督部分
- `val` 用于模型验证和保存最优权重

---

## 6.2 一个 batch 中如何混合有标签和无标签

训练阶段使用 `TwoStreamBatchSampler`。

逻辑是：

- `labeled_idxs` 负责有标签样本
- `unlabeled_idxs` 负责无标签样本
- 每个 batch 中固定有一部分 labeled，一部分 unlabeled

例如：

- `batch_size = 24`
- `labeled_bs = 12`

则一个 batch 中：

- 前 12 个是有标签样本
- 后 12 个是无标签样本

这样模型在每一次更新中都能同时看到有标签和无标签图像。

---

## 6.3 有标签与无标签增强策略

在 `dataloader/transforms.py` 中：

- 有标签数据使用 `train_weak`
- 无标签数据使用 `train_strong`

这是一种典型半监督设计：

- 有标签数据保留较稳定的弱增强，保证监督可靠
- 无标签数据施加强增强，训练模型对扰动保持一致和鲁棒

---

## 6.4 为什么当前实现中无标签目录也保留 mask

从算法角度，无标签数据不应该需要真实掩码。

但在当前仓库实现里，`dataset.py` 会统一读取：

```python
label_path = case.replace("image", "mask")
```

因此即便某个样本逻辑上是 “unlabeled”，代码仍然会尝试读取对应 mask 文件。

需要明确区分：

- 从算法设计看，无标签 GT 不应该参与监督
- 从当前工程实现看，无标签目录下仍然需要存在 mask 文件，以保证 loader 能跑通

真正用于有监督损失的只有 batch 里前 `labeled_bs` 个样本。

---

## 7. 无标签数据是如何被利用的

这是理解 KnowSAM 的关键。

## 7.1 无标签数据并不直接参与 Dice/IoU 计算

无标签样本没有可靠人工标注时，不能直接计算：

- Dice
- IoU
- HD95

因为这些指标必须依赖真实 GT。

所以无标签数据在 KnowSAM 中主要用于训练约束，而不是正式评估。

---

## 7.2 无标签数据在训练中如何产生学习信号

无标签数据主要通过以下四种方式参与：

### 1. 双分支一致性

UNet 与 VNet 对同一张无标签图像应该输出相近结果。

### 2. 熵最小化

要求网络在无标签图像上的预测更自信。

### 3. SAM 蒸馏

SAM 的输出作为 teacher，引导 UNet 和 VNet 学习。

### 4. 后期伪标签混合

超过 `mixed_iterations` 后：

- 利用 `pred_sam_soft` 为无标签样本生成伪标签
- 通过 uncertainty-guided mixing 构造混合样本
- 对混合样本继续训练

因此，无标签数据不是 “没有监督”，而是通过：

- 结构一致性
- 信息熵正则
- teacher 蒸馏
- 伪标签混合

被间接监督。

---

## 7.3 无标签样本的“结果”是如何计算出来的

如果这里的“结果”指训练时给无标签样本的目标信号，那么它来自两类来源：

### 来源 A：SAM 预测

在主训练阶段：

- `pred_sam` 作为 teacher
- 用 KL 蒸馏给 `UNet` 和 `VNet`

### 来源 B：伪标签

在 mixup 阶段：

- 对无标签样本，用 `pred_sam_soft` 或其 argmax 得到伪标签
- 伪标签参与混合监督

因此无标签样本的“监督结果”并不是人工标签，而是模型内部构造出来的 teacher signal 与 pseudo label。

---

## 8. 验证与测试指标设计

## 8.1 当前代码中使用的评估指标

在 2D 图像任务路径中，当前实现使用：

- Dice
- IoU
- Hausdorff 相关指标

在 `utils/utils.py` 里的 `eval()` 中返回：

- `dc`
- `jc`
- `hausdorff_distance * 0.95`

这里需要特别注意：

- 代码中的第三项命名上常被当作 `hd95`
- 但实现不是严格的标准 `HD95`
- 它只是将普通 Hausdorff distance 乘以 `0.95`

因此如果用于正式论文复现，建议自行核对或修正这部分实现。

---

## 8.2 训练阶段的验证逻辑

训练阶段的 `val()` 只使用 `val` 集，也就是有标签验证集。

然后根据验证结果保存最优模型：

- `sam_best_model.pth`
- `SGDL_best_model.pth`

模型保存主要依据是平均 Dice。

所以训练阶段模型选择完全依赖有标签验证集。

---

## 8.3 测试阶段是否也必须用有标签数据

是的。

如果要计算以下正式分割指标：

- Dice
- IoU
- HD95

那么测试集必须有真实 mask。

否则只能得到预测图，无法得到这些定量结果。

因此对于问题：

“这些指标是否均通过有标签的数据进行验证？”

答案是：

- 是的，正式精度指标一定是在有标签数据上算的。
- 无标签数据不能直接用于标准分割指标评估。

---

## 9. 实验设计应如何理解

从当前代码实现可读出的实验设计特点包括：

- 训练集被拆成有标签和无标签两部分
- 验证集是单独的有标签数据
- 使用双学生结构做半监督学习
- SAM 作为外部先验提供蒸馏和提示
- 在训练后期加入 uncertainty-guided mixup

此外，`patients_to_slices()` 中还定义了不同数据集在不同标注量设置下的映射。

这表明论文中的 “1-label, 10-label, 20-label” 等实验设定，不是纯比例，而更接近固定数量的样本切片配置。

例如 tumor 数据集里：

- `1 -> 15`
- `10 -> 145`
- `20 -> 290`
- `30 -> 435`

这说明实验里“标注量”是预先定义好的。

---

## 10. KnowSAM 的主要创新点总结

结合当前代码实现，KnowSAM 的创新点可以总结为以下几点。

### 创新点 1：Learnable Prompting

不是手工点提示或框提示，而是通过 `Super_Prompt` 从特征中学习 prompt。

意义：

- 避免人工构造提示
- 让 prompt 自适应医学图像任务

### 创新点 2：SAM-induced Knowledge Distillation

SAM 不只是辅助推理，而是直接参与训练，向传统分割网络提供蒸馏监督。

意义：

- 让传统医学分割网络继承 SAM 的通用视觉先验

### 创新点 3：双学生互补结构

UNet 与 VNet 是两种结构不同的学生网络，再通过融合头整合冲突与不确定性信息。

意义：

- 提升无标签学习稳定性
- 提供更可靠的融合预测

### 创新点 4：不确定性感知的混合训练

训练后期依据预测熵选择不确定 patch，做 labeled/unlabeled mixing。

意义：

- 更有效利用无标签区域
- 强化困难区域学习

---

## 11. 当前实现中的注意事项

以下内容并不是论文思路的问题，而是当前仓库实现中需要留意的工程细节。

### 1. `loss_diff1/2` 实现可能存在覆盖问题

`for` 循环中每个类别的 BCE 没有累加，最终更像只保留了最后一个类别的损失。

这可能影响一致性损失的实际效果。

### 2. 2D 路径中的 “hd95” 不是标准实现

若做正式结果报告，建议替换为真正的 `hd95` 实现。

### 3. 无标签目录仍然需要 mask 文件

这是当前数据加载实现决定的，不是算法本身要求。

### 4. `prediction.py` 默认测试集并不适配所有任务

原仓库版本偏向固定数据集命名方式，例如 `test_CVC-300` 等。如果换成自定义数据集，通常需要根据自己的目录结构调整测试入口。

---

## 12. 对“无标签数据如何计算结果”的直接回答

这个问题可以分成两个层面。

### 层面 1：训练时无标签数据如何得到监督信号

答案是通过：

- 双分支一致性损失
- 熵最小化损失
- SAM 蒸馏损失
- 后期伪标签混合损失

也就是说，无标签数据并不是没有目标，而是目标来自模型内部构造出的 teacher signal 与 pseudo label。

### 层面 2：无标签数据能否计算 Dice、IoU、HD95

答案是不行，除非你手里实际上有它们的 GT。

因为这些指标本质上都是：

- 预测结果 vs 真实标签

没有真实标签，就不能做严格的量化精度评估。

所以：

- 训练时无标签样本能参与优化
- 但正式验证和测试指标仍然必须依赖有标签数据

---

## 13. 最后的整体理解

如果用一句话概括 KnowSAM：

它是一个利用 learnable prompt 将 SAM 引入半监督医学图像分割训练闭环，并通过双学生一致性、蒸馏和不确定性混合策略提升小样本分割性能的框架。

其训练逻辑的本质是：

- 有标签数据负责提供稳定锚点
- 无标签数据负责提供额外约束和伪监督
- SAM 负责提供强先验与蒸馏目标
- 双学生结构负责提升泛化和鲁棒性

因此 KnowSAM 的重点不是 “SAM 单独分割得多准”，而是 “如何让 SAM 的知识有效服务于半监督训练”。
