# 独立目标任务与选项扰动评估

在 SST-2、CB、COPA 和 HellaSwag 的 11,070 道验证题上，原版 UniMC 的任务宏准确率为 **78.6733%**。使用题内选项中心化和跨选项注意力的评分头（centered full）的三种子均值为 **77.8103%**，差值 **−0.8630 个百分点**。配对 bootstrap 的 95% 区间为 **[−2.2188, +0.0519]** 个百分点。三个种子的点估计均低于原版，区间覆盖零。源开发集上观察到的 +0.1172 个百分点未在四个目标任务的宏准确率上复现。

## 原题结果

每项任务等权，每个头使用预先固定的 42、43、44 三个种子。跨种子标准差与题目配对置信区间衡量不同来源的波动。

| 模型 | 宏准确率（%） | 种子标准差（百分点） | 宏 NLL |
|---|---:|---:|---:|
| 原版 UniMC | 78.6733 | — | 0.546063 |
| 未中心化 full | 78.6741 | 0.0014 | 0.546164 |
| 中心化 self | 77.7932 | 0.0101 | 0.550429 |
| 中心化 MLP | 77.7912 | 0.0223 | 0.549770 |
| 中心化 full | 77.8103 | 0.0278 | 0.549766 |

在相同中心化表示下，full 比 self 高 0.0171 个百分点，比 MLP 高 0.0191 个百分点。两项配对区间均覆盖零，额外的跨选项注意力未显示出明确收益。

| 任务 | 原题数 | 原版准确率（%） | 中心化 full 均值（%） | 差值（百分点） |
|---|---:|---:|---:|---:|
| SST-2 | 872 | 93.0046 | 93.0810 | +0.0765 |
| CB | 56 | 67.8571 | 64.2857 | −3.5714 |
| COPA | 100 | 97.0000 | 97.0000 | 0.0000 |
| HellaSwag | 10,042 | 56.8313 | 56.8745 | +0.0432 |

CB 上三个种子都把原版答对的同两道题（ID 13、40）改错，正确数从 38/56 降至 36/56。两题原版前两名的 logit 差分别为 0.0221 和 0.0464。CB 占宏平均的四分之一，仅这两题便贡献了 −0.8929 个百分点，解释了大部分宏平均差异。

## 选项扰动与缺答案识别

每道原题做一次预先固定的非零循环移位。4 个任务的 13 组预测在选项回映后均 **100% 保留原选择**。此结果只覆盖本次轮换操作。

CB、HellaSwag 的每道题各生成两种删项版本，一版删干扰项并保留正确答案，另一版删去标注正确项。拒识指标在这两组之间计算，两组题数和剩余选项数相同。以缺标注正确项为正类，以 1 减最大 softmax 概率为拒识分数。

| 任务 | 删干扰项准确率（%，原版 / full） | 缺答案 AUROC（原版 / full） | 缺答案 AP（原版 / full） |
|---|---:|---:|---:|
| CB | 78.57 / 73.21 | 0.4914 / 0.4881 | 0.5013 / 0.5065 |
| HellaSwag | 64.50 / 64.52 | 0.5677 / 0.5681 | 0.5489 / 0.5489 |

删除一个干扰项后，两模型在两项任务上的准确率均高于各自原题得分。对于删掉正确项的题，原始置信度在 CB 上的区分能力接近随机，在 HellaSwag 上也有限。此处衡量的是拒识分数的排序能力，尚未评估固定阈值下的拒答表现。

## 复核记录

- 回传包为 `UniMC_target_results.tar.gz`，SHA-256 为 `2e0cbed58661222c6931e0d27f1ce4b00382c81627426efff93f3ba1015fbacb`。四份完成记录中的输入指纹、基座签名和 12 个评分头的 SHA-256 与本地固定版本匹配，52 份预测均通过文件校验。每项任务首批两条输入的官方预测对照通过，四项任务均记录零次选项截断。
- 按任务和 ID 对齐全部 550,368 条预测（42,336 条输入 × 13 组模型），核对选项顺序、标签及 softmax 概率。用独立实现从 logits 复算各任务各模型的原题准确率和 NLL，均与回传指标一致。使用项目记录的 NumPy 1.26.4 复算全部条件及 20,000 次配对 bootstrap 后，结果表与逐题变化清单均逐字节一致，全量指标最大绝对差为 2.23 × 10⁻¹⁶ 以内。NumPy 2.5.4 下，full 对 MLP 的区间端点最多漂移 0.00081 个百分点，解释相同。
- [全量指标](metrics.json)和[逐题变化](changed.csv)取自回传包。数据与扰动定义见[评测协议](../README.md)。本结果覆盖两源训练的评分头和原始分数；四源 v0.3 决策层及其温度校准有[单独结果](../../calibration/results/report.md)。

## 英文报告用段落

Across four equally weighted held-out tasks, the centered full-attention head averaged 77.81% accuracy over three seeds, compared with 78.67% for the frozen UniMC baseline (difference −0.86 percentage points, paired 95% bootstrap interval [−2.22, +0.05]). All three seeds changed the same two correct baseline answers on CB into errors, reducing CB accuracy from 38/56 to 36/56. SST-2 and HellaSwag showed small mean gains, and COPA accuracy was unchanged. Every evaluated model preserved its selected answer under the tested cyclic option rotation. Using one minus the maximum softmax probability to identify questions with the annotated answer removed yielded mean AUROCs of 0.488 on CB and 0.568 on HellaSwag for the centered full-attention head. The source-development gain therefore did not carry over to mean target accuracy, and raw confidence offered limited discrimination of missing-answer cases.
