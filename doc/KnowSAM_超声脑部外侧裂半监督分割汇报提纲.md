# 超声脑部外侧裂半监督分割汇报提纲

## 1. 当前 A3 数据与结果

- 仅使用 A3 切面，不涉及其他切面和视频
- A3 总样本数：167
- 数据划分：labeled 40 / unlabeled 109 / val 9 / test 9
- 已标注 A3：68
- 纯未标注 A3：99
- best val Dice：0.7024
- test Dice：0.5874
- test IoU：0.4286
- test HD95：3.6204

## 2. 文献与 GitHub
- **KnowSAM (TMI 2025)**：可学习提示 + SAM 蒸馏 + 双分支协同，是当前 A3 半监督分割基线。
  - Paper: https://ieeexplore.ieee.org/document/10843257
  - GitHub: https://github.com/taozh2017/KnowSAM
- **PH-Net (CVPR 2024)**：困难补丁挖掘与对比学习，适合超声噪声强、边界模糊的 A3 图像。
  - Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_PH-Net_Semi-Supervised_Breast_Lesion_Segmentation_via_Patch-wise_Hardness_CVPR_2024_paper.html
  - GitHub: https://github.com/jjjsyyy/PH-Net
- **CPC-SAM (MICCAI 2024)**：交叉提示一致性，适合提升双分支在困难边界区域的互补监督。
  - Paper: https://arxiv.org/abs/2407.05416
  - GitHub: https://github.com/JuzhengMiao/CPC-SAM
- **SemiSAM+ (MedIA 2025)**：基础模型时代的半监督重构，强调极少标注条件下的稳定泛化。
  - Paper: https://arxiv.org/abs/2502.20749
  - GitHub: https://github.com/YichiZhang98/SemiSAM
- **SAM-MedUS (JMI 2025)**：面向超声模态的基础模型与边界损失设计，可为 A3 任务提供超声先验。
  - Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11865838/
  - GitHub: https://github.com/tf2bb/SAM-MedUS
- **E-BayesSAM (MICCAI 2025)**：贝叶斯不确定性估计与 Token 剪枝，适合控制超声低置信区域蒸馏。
  - Paper: https://arxiv.org/abs/2508.17408
  - GitHub: https://github.com/mp31192/E-BayesSAM


## 3. 三大创新点

### 创新点一：U-CKD

- 用不确定性图动态控制蒸馏强度。
- 在低置信区域抑制错误伪标签传播。
- 面向 A3 模糊边界与局部噪声问题。

### 创新点二：QAPL

- 对未标注 A3 做质量感知伪标签筛选与迭代训练。
- 让 109 个未标注 A3 的价值被更充分释放。
- 突出新数据划分带来的方法升级。

### 创新点三：SAP-BR

- 引入外侧裂形态先验、边界损失和困难补丁强化。
- 约束预测结果的细长率、连通性和边界位置。
- 提升结果的解剖合理性与可解释性。

## 4. 汇报建议主线

1. 任务背景与 A3 约束
2. 新数据划分与训练配置
3. 新结果与日志分析
4. 当前瓶颈
5. 三大创新点
6. 下一步实验计划
