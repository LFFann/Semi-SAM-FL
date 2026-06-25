# A3-PASS：面向婴儿脑部超声外侧裂的三核心创新半监督分割架构

更新时间：2026-05-07

本文档是在原 A3-PASS 方案基础上的收缩版。目标是避免方法复杂化，将顶会级创新收敛为三个核心创新点，同时保持任务边界清晰：

1. 任务仍然是半监督图像分割；
2. 最终输出仍然是像素级 mask；
3. 主评价指标仍然是 Dice、IoU、95HD；
4. 不以“Dice 与临床目标脱钩”为论文主线；
5. 创新重点是重定义半监督分割中的学习对象与网络信息流。

推荐方法名：

**A3-PASS: Acoustic-Anatomical Posterior State-Space Semi-supervised Segmentation**

中文名称：

**A3 声学-解剖后验状态空间半监督分割框架**

---

## 1. 核心结论

A3-PASS 的主张不是在 KnowSAM 上继续叠加注意力、边界损失、伪标签筛选或 SAM prompt 策略，而是重新审视婴儿脑部超声外侧裂分割中的半监督学习对象。

传统半监督分割默认：

```text
unlabeled image -> pseudo mask -> pixel supervision
```

该假设在外侧裂超声中很脆弱。原因是超声图像并不是稳定的外观图像，而是解剖结构经过声束传播、散斑、阴影、伪影、探头角度和扫查平面共同作用后的观测结果。局部像素纹理不稳定，但外侧裂作为细长解剖结构，其位置、方向、尺度、连续性和距离场等低维结构状态相对更稳定。

因此，A3-PASS 将半监督分割重新定义为：

> 在少量标注 mask 和大量无标注超声图像中，先学习外侧裂的声学-解剖状态后验，再由该后验条件化生成最终像素 mask。

最终仍然用 Dice、IoU、95HD 评价 mask。变化不在评价体系，而在训练时无标注数据提供的监督信号：

```text
pseudo mask supervision
  -> acoustic-anatomical posterior supervision
```

---

## 2. 核心动机：从超声图像的物理与数学本质出发

### 2.1 物理本质

婴儿脑部超声图像中的外侧裂不是自然图像中的普通物体。它的可见性和边界受以下因素控制：

1. 声束入射角改变局部回声强度；
2. 散斑噪声破坏局部纹理稳定性；
3. 阴影和伪影会产生类似裂隙的假边界；
4. 外侧裂是细长、小面积、弱边界结构；
5. 扫查平面变化会改变可见形态。

这意味着像素强度并不是外侧裂类别的稳定观测。更合理的理解是：

```text
ultrasound image = acoustic observation of latent anatomy
```

因此，直接从无标注图像生成 hard pseudo mask，等价于把不稳定声学观测过早离散化为像素真值。

### 2.2 数学本质

传统半监督分割优化的是：

\[
\min_\theta L_{sup}(f_\theta(x_l), y_l)
+ \lambda L_{unsup}(f_\theta(x_u), \hat{y}_u)
\]

其中 \(\hat{y}_u\) 是伪标签 mask。

隐含假设是：

\[
\hat{y}_u \approx y_u
\]

但在外侧裂超声中，更合理的概率图是：

\[
z \rightarrow y,\quad z \rightarrow x
\]

其中：

1. \(z\)：潜在声学-解剖状态，例如位置、方向、尺度、连续性、距离场；
2. \(y\)：像素 mask；
3. \(x\)：超声图像观测。

模型真正需要估计的是：

\[
p(y|x)=\int p(y|x,z)q(z|x)dz
\]

传统方法直接学习 \(p(y|x)\)，并在无标注数据上用当前 \(p(y|x)\) 生成伪标签。A3-PASS 则显式学习 \(q(z|x)\)，再通过 \(p(y|x,z)\) 生成 mask。

---

## 3. 问题定义

### 3.1 传统定义

给定少量标注数据：

\[
D_l=\{(x_i,y_i)\}_{i=1}^{N_l}
\]

和大量无标注数据：

\[
D_u=\{x_j\}_{j=1}^{N_u}
\]

学习分割模型：

\[
f_\theta(x)\rightarrow y
\]

评价指标为 Dice、IoU、95HD。

### 3.2 A3-PASS 定义

A3-PASS 不改变最终任务，而是改变中间学习定义：

\[
x \rightarrow q_\theta(z|x) \rightarrow p_\theta(y|x,z)
\]

其中 \(z\) 是由 mask 自动派生、并可在无标注图像中自监督学习的声学-解剖状态。

最终预测仍为：

\[
\hat{y}=argmax\ p_\theta(y|x,z)
\]

最终评价仍为：

```text
Dice / IoU / 95HD
```

### 3.3 状态变量必须足够克制

为避免方法复杂化，A3-PASS 只保留三类状态，不引入过多临床变量：

| 状态 | 含义 | 来源 |
|---|---|---|
| Geometry state | 质心、面积比例、主轴方向、长宽比 | 从标注 mask 自动计算 |
| Dense state | 低分辨率 signed distance field 或 skeleton heatmap | 从标注 mask 自动生成 |
| Latent state token | 网络学习的结构 token | 由 state head 学习 |

这些状态不是新标注，也不是新评价指标，而是半监督训练中的中间变量。

---

## 4. 现有架构的不足

### 4.1 U-Net / CNN 分割范式

隐含假设：

```text
local texture is sufficient for pixel classification
```

外侧裂超声中该假设不成立。散斑和伪影会让相似纹理对应不同类别，也会让同一结构在不同图像中呈现不同纹理。

具体错误：

1. 把阴影边缘误分为外侧裂；
2. 把真实外侧裂低对比区域漏分；
3. 小目标被背景吞掉；
4. 预测 mask 断裂或膨胀。

这不是简单加 attention 能根治的问题，因为问题来自像素外观与结构类别之间的非稳定对应关系。

### 4.2 SAM / foundation model 辅助分割范式

隐含假设：

```text
foundation model prior can regularize medical segmentation
```

该假设部分成立，但不充分。SAM 的先验主要来自自然图像对象，它擅长可见边界和对象分组，而外侧裂是弱边界、细长、非闭合、受声学成像影响的结构。

具体错误：

1. SAM 对伪影边界产生过强响应；
2. prompt 偏移后 mask 被系统性带偏；
3. SAM 输出被当作 teacher 后反向污染 student；
4. SAM 产生的 mask 不一定符合外侧裂细长结构。

### 4.3 KnowSAM

KnowSAM 的核心闭环可概括为：

```text
SGDL fusion map -> SAM prompt -> SAM mask -> KD back to SGDL
```

它的问题不是“没有可靠性估计”这么简单，而是整个闭环的核心变量仍然是 mask。若初始 fusion map 偏，SAM prompt 也偏；若 SAM mask 偏，KD 又把偏差蒸馏回 SGDL。

### 4.4 A3-RCP-KnowSAM

A3-RCP 已经进行了重要修正：

1. consensus prompt 替代 raw fusion prompt；
2. reliability-weighted KD 替代等权 KD；
3. quality-aware pseudo label 替代等权伪标签；
4. 边界和面积先验约束小目标分割。

但 A3-RCP 仍然属于：

```text
reliability-calibrated pseudo-mask framework
```

也就是说，它提高了伪标签和 teacher 的使用质量，但没有改变无标注样本的核心学习对象。

A3-PASS 的区别是：

```text
A3-RCP: how to use pseudo masks more safely
A3-PASS: how to avoid using pseudo masks as the first-order unlabeled target
```

---

## 5. Ours 架构

## 5.1 总体信息流

```text
Input ultrasound image x
  -> Shared image encoder E(x)
       |
       |-- Pixel segmentation branch
       |     p_s(y|x)
       |
       |-- Acoustic-anatomical state posterior branch
       |     q(z|x)
       |
       |-- State-conditioned mask decoder
             p_d(y|x,z)

Training with labeled data:
  mask CE/Dice supervision
  + automatically derived state supervision
  + state-conditioned mask reconstruction

Training with unlabeled data:
  posterior state consistency
  + posterior reliability estimation
  + reliability-gated pixel pseudo supervision

Inference:
  x -> q(z|x), E(x) -> p_d(y|x,z) -> mask

Evaluation:
  Dice / IoU / 95HD
```

### 5.2 与 SAM / KnowSAM 的关系

A3-PASS 可以保留 SAM 或 A3-RCP 的部分能力，但它们不是核心创新。

合理使用方式：

1. SAM image encoder 可作为辅助 feature prior；
2. A3-RCP consensus prompt 可作为 SAM proposal 输入；
3. SAM mask 可作为弱 proposal，与 pixel branch 一起参与 state posterior 估计；
4. 最终方法核心必须独立于 SAM，即去掉 SAM 后仍能形成 state-space semi-supervised segmentation。

论文中应强调：

> SAM is used as an auxiliary proposal source rather than the central teacher. The central learning signal is the acoustic-anatomical posterior state.

---

## 6. 三个核心创新点

## 创新点一：Acoustic-Anatomical State Posterior

英文名称：

**AASP: Acoustic-Anatomical State Posterior**

中文解释：

声学-解剖状态后验建模。

### 核心动机

超声图像的像素外观不稳定，但外侧裂的结构状态相对稳定。传统伪标签半监督直接将无标注图像映射为 pseudo mask，容易把声学伪影固化为前景标签。

AASP 的目标是先估计：

\[
q_\theta(z|x)
\]

而不是直接相信：

\[
p_\theta(y|x)
\]

### 具体设计

对标注样本，从 mask 自动生成状态目标：

```text
mask y
  -> geometry vector g(y)
  -> low-resolution SDF d(y)
  -> skeleton / center heatmap h(y)
```

state branch 输出：

```text
z = [geometry token, dense state map, latent token]
```

训练目标：

\[
L_{state}^{labeled}
= L_{geo}(\hat{g},g)
+ L_{dense}(\hat{d},d)
+ L_{token}
\]

### 为什么不是普通多任务学习

普通多任务学习把 skeleton、boundary、SDF 当作辅助 loss，最终仍由 pixel branch 单独分割。AASP 的状态不是辅助输出，而是后续 mask decoder 的条件变量，也是无标注学习的主变量。

### 解决的问题

1. 降低局部声学纹理对伪标签的支配；
2. 提供比 hard mask 更稳定的无标注学习对象；
3. 让模型显式学习外侧裂的细长、小面积、方向性结构。

---

## 创新点二：State-Conditioned Mask Decoding

英文名称：

**SCMD: State-Conditioned Mask Decoder**

中文解释：

状态条件化 mask 解码器。

### 核心动机

外侧裂 mask 不应只由像素分类器决定。对于弱边界和低对比区域，模型需要结合图像证据和结构状态来判断最终 mask。

传统 decoder 学习：

\[
p(y|x)
\]

SCMD 学习：

\[
p(y|x,z)
\]

### 具体设计

推荐简单实现，不做复杂生成模型：

```text
image feature E(x)
  + state token z
  -> cross-attention / FiLM modulation / dynamic convolution
  -> segmentation logits
```

可选三种实现，按复杂度排序：

1. FiLM：用 state vector 生成 feature scale 和 bias；
2. Cross-attention：state token 查询图像 token；
3. Dynamic convolution：state token 生成轻量卷积核。

最小可行版本建议使用 FiLM 或 cross-attention，避免过度复杂。

### 训练目标

标注样本：

\[
L_{decode}^{labeled}
= CE(p_d(y|x,z),y)
+ Dice(p_d(y|x,z),y)
\]

最终推理：

\[
\hat{y}=argmax\ p_d(y|x,\hat{z})
\]

### 为什么不是“加一个外侧裂 decoder”

普通 decoder 只是增加参数。SCMD 的核心是改变 mask 生成方式：mask 是由 image evidence 和 state posterior 共同决定的，而不是纯 pixel classifier 输出。

### 解决的问题

1. 避免 pixel branch 被局部伪影直接带偏；
2. 让预测 mask 保持细长结构和全局方向一致；
3. 在边界模糊时用状态后验补充图像证据。

---

## 创新点三：Posterior-Guided Unlabeled Learning

英文名称：

**PGUL: Posterior-Guided Unlabeled Learning**

中文解释：

后验引导的无标注学习。

### 核心动机

无标注样本不是不能用 pseudo mask，而是不能一开始就把 pseudo mask 当成主要监督。A3-PASS 先判断 state posterior 是否稳定，再决定是否使用像素伪标签。

### 具体设计

对无标注样本 \(x_u\)，构造弱增强和强增强：

```text
x_w = weak_aug(x_u)
x_s = strong_aug(x_u)
```

先做状态一致性：

\[
L_{state-cons}
= D(q(z|x_w), q(z|x_s))
\]

再估计 state reliability：

\[
R_z = \sigma(
-H(q(z|x))
-D(p_s(y|x), p_d(y|x,z))
)
\]

其中：

1. state posterior 熵越低，越可靠；
2. pixel branch 与 state-conditioned decoder 越一致，越可靠；
3. reliability 用于控制无标注 pixel supervision。

最终无标注损失：

\[
L_u =
\lambda_1 L_{state-cons}
+ \lambda_2 R_z L_{pseudo}
+ \lambda_3 R_z L_{decode-cons}
\]

其中 \(L_{pseudo}\) 仍可使用 hard pseudo mask 或 soft pseudo mask，但它被 state reliability 门控。

### 为什么不是普通伪标签筛选

普通伪标签筛选依据 pixel confidence。PGUL 的筛选依据是 state posterior reliability。它回答的不是：

```text
this pixel is confident or not
```

而是：

```text
this image induces a stable anatomical state or not
```

### 解决的问题

1. 抑制高置信错误伪标签；
2. 减少 teacher-student 确认偏差；
3. 让无标注样本先贡献结构分布，再贡献像素监督；
4. 保持最终分割评价仍然是 Dice / IoU / 95HD。

---

## 7. 三个创新点之间的关系

A3-PASS 只保留三个核心创新点，信息流如下：

```text
AASP:
  x -> q(z|x)

SCMD:
  E(x), z -> p(y|x,z)

PGUL:
  unlabeled x -> state consistency + reliability-gated pseudo supervision
```

其中：

1. AASP 定义模型要学习的中间状态；
2. SCMD 保证状态会影响最终 mask；
3. PGUL 让无标注数据在状态空间中发挥作用。

如果去掉 AASP，方法退化为普通 segmentation；
如果去掉 SCMD，状态只是不影响预测的辅助任务；
如果去掉 PGUL，无标注数据仍然主要依赖传统伪标签。

---

## 8. 数学形式

### 8.1 总体目标

\[
\min_\theta
L_l + \lambda L_u
\]

其中：

\[
L_l =
L_{seg}(p_s(y|x_l),y_l)
+ \alpha L_{state}(q(z|x_l),T(y_l))
+ \beta L_{decode}(p_d(y|x_l,z_l),y_l)
\]

\[
L_u =
\gamma L_{state-cons}
+ \eta R_z L_{pseudo}
+ \rho R_z L_{decode-cons}
\]

### 8.2 状态目标自动生成

\[
T(y)=\{g(y),d(y),h(y)\}
\]

其中：

1. \(g(y)\)：几何向量；
2. \(d(y)\)：低分辨率 signed distance field；
3. \(h(y)\)：skeleton 或 center heatmap。

这些目标全部从已有 mask 派生，不需要额外人工标注。

### 8.3 状态后验

简化实现：

\[
q(z|x)=Head_z(E(x))
\]

可输出：

```text
state vector + dense state map
```

不建议第一版引入复杂 VAE 或扩散模型，否则方法会过重。

### 8.4 状态条件解码

\[
p_d(y|x,z)=Decoder(E(x),z)
\]

最终可融合 pixel branch：

\[
p(y|x)=R_z p_d(y|x,z)+(1-R_z)p_s(y|x)
\]

第一版也可以直接使用 \(p_d\) 作为最终输出，避免推理复杂。

---

## 9. 论文故事线

### 9.1 一句话故事

Existing semi-supervised segmentation methods exploit unlabeled ultrasound images by propagating pixel-level pseudo masks, which is unreliable for weak-boundary anatomical structures under acoustic artifacts. We propose A3-PASS, a state-space semi-supervised framework that first learns acoustic-anatomical posterior states and then decodes segmentation masks conditioned on these states.

### 9.2 Introduction 逻辑链

第一段：半监督医学图像分割通过少量标注和大量无标注数据降低标注成本，外侧裂超声分割是典型应用。

第二段：现有方法包括 U-Net、teacher-student、CPS、SAM-assisted segmentation、KnowSAM 等，通常使用 pseudo labels、consistency regularization 或 foundation model distillation。

第三段：这些方法默认无标注图像可以被当前模型转换为可靠 pseudo mask。但在超声中，像素纹理由声学传播和伪影决定，外侧裂又是细长弱边界结构，因此高置信 pseudo mask 可能是错误声学证据。

第四段：从物理上看，超声图像是潜在解剖状态的声学观测；从数学上看，分割应建模 \(p(y|x)=\int p(y|x,z)q(z|x)dz\)，而不是直接用 \(p(y|x)\) 自训练。

第五段：提出 A3-PASS，包括 acoustic-anatomical state posterior、state-conditioned mask decoding 和 posterior-guided unlabeled learning。它仍然输出标准 mask，并用 Dice / IoU / 95HD 评价，但无标注学习对象从 pseudo mask 转为 state posterior。

### 9.3 Contributions

1. We reformulate semi-supervised lateral sulcus segmentation as acoustic-anatomical posterior learning, where unlabeled ultrasound images first regularize latent structural states rather than directly supervising pixel pseudo masks.
2. We introduce a state-conditioned mask decoder that generates segmentation masks from both image evidence and anatomical posterior states, enabling structure-aware segmentation under weak boundaries and acoustic artifacts.
3. We propose posterior-guided unlabeled learning, which uses state consistency and state reliability to control pseudo supervision, reducing confirmation bias while preserving standard Dice / IoU / 95HD evaluation.

---

## 10. 实验设计

### 10.1 主实验

主表只使用标准分割指标：

```text
Dice ↑
IoU ↑
95HD ↓
```

Baseline：

1. U-Net；
2. VNet 或 DeepLabV3+；
3. Mean Teacher；
4. CPS；
5. UA-MT 或 uncertainty-aware SSL；
6. SAM / MedSAM-style baseline；
7. KnowSAM；
8. A3-RCP-KnowSAM；
9. A3-PASS。

标注比例：

```text
5%, 10%, 20%, full labeled
```

如果数据规模有限，至少保留：

```text
10%, 20%, full labeled
```

### 10.2 三核心消融

必须围绕三个创新点做消融，避免实验散乱。

| 设置 | 目的 |
---|---|
| Full A3-PASS | 完整方法 |
| w/o AASP | 去掉状态后验，只保留 pixel branch |
| w/o SCMD | 状态只作辅助监督，不参与 mask 解码 |
| w/o PGUL | 无标注数据回到普通 pseudo mask |
| PGUL with entropy reliability | 用 pixel entropy 替代 state reliability |
| pixel consistency instead of state consistency | 验证 state consistency 是否必要 |
| w/o SAM branch | 证明核心方法不依赖 SAM |
| A3-RCP + state auxiliary loss | 证明不是简单加辅助任务 |

### 10.3 机制分析

主评价仍然是 Dice / IoU / 95HD，但需要机制分析支撑论文主张。

建议分析：

1. state reliability 与 pseudo mask Dice 的相关性；
2. pixel entropy 与 pseudo mask Dice 的相关性；
3. state consistency loss 收敛曲线；
4. AASP 预测的 SDF / skeleton 可视化；
5. SCMD 与普通 decoder 的预测差异；
6. 失败伪标签被 PGUL 降权的案例。

### 10.4 低标注实验

A3-PASS 的价值应主要体现在少标注场景。若 full labeled 下提升较小但 5% / 10% 下提升明显，这是合理结果。

论文中应重点展示：

```text
label scarcity -> pseudo-mask confirmation bias increases
state posterior learning -> stronger low-label robustness
```

### 10.5 泛化实验

如果数据允许，加入：

1. 跨 fold；
2. 跨图像质量；
3. 跨操作者；
4. 跨设备；
5. 跨扫查平面。

仍然使用 Dice / IoU / 95HD。泛化实验不是为了引入新指标，而是证明 state posterior 比 pixel pseudo mask 更稳。

### 10.6 失败案例

必须展示失败案例：

1. 外侧裂几乎不可见；
2. 强阴影遮挡；
3. 伪影形状与外侧裂相似；
4. 极小前景区域；
5. 标注边界本身不稳定。

失败案例中需要指出：A3-PASS 降低错误传播，但不能消除成像不可判定性。

---

## 11. 审稿风险

### 11.1 风险一：被认为只是多任务学习

如果写法变成：

```text
we add SDF head, skeleton head and boundary loss
```

会被认为是普通医学分割增量。

防御：

1. 强调 AASP 是无标注学习的主变量；
2. 证明 w/o SCMD 后性能下降，即状态不是辅助输出；
3. 证明 PGUL 优于普通 pseudo label 和 pixel consistency。

### 11.2 风险二：状态定义过于手工

如果状态全部是人工设计几何量，审稿人会质疑泛化性。

防御：

1. 只使用从 mask 自动派生的通用状态；
2. 加入 latent state token；
3. 展示方法可迁移到其他细结构超声分割任务；
4. 不声称状态完全等价于真实解剖，只称为 segmentation-useful posterior state。

### 11.3 风险三：复杂度高但 Dice 提升有限

防御：

1. 控制核心模块只有三个；
2. 报告参数量和推理速度；
3. 强调低标注比例收益；
4. 用消融证明每个核心模块必要。

### 11.4 风险四：与 A3-RCP 差异不够

防御：

1. A3-RCP 是 reliability-calibrated pseudo-mask framework；
2. A3-PASS 是 state-posterior-guided SSL framework；
3. 消融中加入 “A3-RCP + state auxiliary loss”，证明简单辅助 loss 不等价于 A3-PASS。

### 11.5 风险五：过度声称物理建模

A3-PASS 并没有完整模拟超声传播方程。不要声称 “physics-based ultrasound segmentation”。

更稳妥表述：

```text
physics-motivated acoustic-anatomical posterior modeling
```

即从超声物理不稳定性出发设计状态空间，但不夸大为严格物理仿真。

---

## 12. 与低质量创新的边界

| 常规想法 | 是否能作为主贡献 | 在 A3-PASS 中的位置 |
---|---|---|
| 加边界 loss | 不能 | 可作为 dense state target 的派生监督 |
| 加注意力模块 | 不能 | 可作为 SCMD 的实现方式 |
| 加多尺度融合 | 不能 | 可作为 encoder 细节 |
| 加形态学后处理 | 不能 | 可用于自动生成 skeleton / SDF |
| 加不确定性权重 | 不能单独作为贡献 | 被 PGUL 的 state reliability 替代 |
| 加 SAM prompt | 不能 | SAM 只是 proposal source |
| CNN-Transformer 融合 | 不能 | backbone 选择，不是核心创新 |
| 伪标签筛选 | 不能 | 升级为 posterior-guided learning |
| consistency loss | 不能 | 从 pixel consistency 改为 state consistency |
| 外侧裂专用 decoder | 风险高 | 必须写成 state-conditioned decoder |

---

## 13. 实施路线

### 13.1 最小可行版本

目标：先验证三个核心创新，而不是一次性实现复杂系统。

新增模块：

1. `StateTargetGenerator`：从标注 mask 生成 geometry vector、SDF、skeleton；
2. `AnatomicalStateHead`：预测 state vector 和 dense state；
3. `StateConditionedMaskDecoder`：用 state token 条件化 mask 解码；
4. `PosteriorGuidedUnlabeledLoss`：实现 state consistency 和 state reliability。

不建议第一版加入：

1. VAE；
2. diffusion；
3. 大型 prototype memory；
4. 复杂 EM 训练；
5. 额外临床标签。

### 13.2 建议代码结构

```text
variants/A3_PASS_KnowSAM/
  trainer_a3_pass.py
  prediction_a3_pass.py
  train_a3_pass.py

utils/
  state_targets.py
  losses_a3_pass.py

Model/
  state_head.py
  state_decoder.py
```

### 13.3 与当前 A3-RCP 代码复用

可复用：

1. 数据加载与 transforms；
2. Dice / IoU / 95HD evaluation；
3. SGDL backbone；
4. SAM image encoder；
5. A3-RCP 的 consensus prompt 作为可选 SAM proposal；
6. prediction 脚本的保存与 monitor 逻辑。

需要新增：

1. state target 生成；
2. state branch；
3. state-conditioned decoder；
4. unlabeled state consistency。

---

## 14. 推荐论文标题

1. **A3-PASS: Acoustic-Anatomical Posterior State-Space Learning for Semi-supervised Ultrasound Segmentation**
2. **Beyond Pseudo Masks: Posterior State Learning for Semi-supervised Lateral Sulcus Segmentation**
3. **State-Conditioned Semi-supervised Segmentation of Infant Brain Ultrasound**
4. **Learning Acoustic-Anatomical Posteriors for Weak-Boundary Ultrasound Segmentation**
5. **From Pixel Pseudo-labels to Anatomical Posteriors in Semi-supervised Medical Segmentation**

---

## 15. 核心结论回收

A3-PASS 的顶会潜力来自一个明确、可检验的问题重定义：

> 对于强噪声、弱边界、细结构的超声分割，无标注数据不应首先被压缩为像素伪标签，而应首先用于学习声学-解剖状态后验；最终 mask 由图像证据和状态后验共同生成。

三个核心创新点足够支撑完整论文：

1. **AASP**：把无标注学习对象从 pixel pseudo mask 改为 acoustic-anatomical state posterior；
2. **SCMD**：把 mask 生成从 \(p(y|x)\) 改为 \(p(y|x,z)\)；
3. **PGUL**：用 state posterior consistency 和 state reliability 控制无标注监督。

最终任务仍然是半监督分割，最终指标仍然是 Dice、IoU、95HD。这样既不会偏离当前 KnowSAM 项目的实验体系，也能避免方法被审稿人看成简单的 loss 或模块堆叠。

最关键的验证问题是：

> 在相同标注比例和相同 Dice / IoU / 95HD 评价下，state-posterior-guided semi-supervised learning 是否比 pseudo-mask-guided semi-supervised learning 更稳定、更有效？

如果答案为是，A3-PASS 才有资格作为主线论文；如果答案为否，则应回退到 A3-RCP 作为工程增强路线。

