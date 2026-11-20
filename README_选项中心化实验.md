> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

# 选项表示中心化：基于真实诊断的单项对照

原始诊断包：`UniMC_pilot_diagnostics.zip`。固定原实验 55518、调参 55537、紧凑头 55555。

本地逐样本复核发现：参数有实际更新，MLM 表示与原模型输出一致；被检查的低学习率/小头检查点在 1,517 条源验证题上均未改变原版预测。MLM full 小头的选项残差分差中位数为 PIQA 0.000638、CQA 0.004982，而原版 top-2 分差中位数为 0.7811、1.8373。原大 full 头改变了 59 题，纠正 27、破坏 29，另有 3 题错到错，未获得净收益。

同题 MLM 选项向量的余弦相似度中位数为 PIQA 0.9919、CQA 0.9425。共同成分可能妨碍小头学习相对差异，这是本轮待验证的假设。

本轮唯一改变是，对每题真实 K 个选项的 MLM 向量进行中心化：

`centered_h[i] = h[i] - mean(h[0:K])`

然后照常进入已有的 LayerNorm、MLP/self/full 和残差相加。中心化输入向量会改变非线性头的学习；仅从最终 logits 中减均值不会改变 softmax 或预测，不能代替此操作。

固定使用 4,000 条源训练数据、1,517 条源验证数据、MLM 特征、d=64、FFN=256、dropout=0.1、lr=1e-4、seed=42、最多 10 轮、patience=2。仅新增三个头，与已完成的三个未中心化 MLM 小头对照。不使用校准或目标数据，不改变模型/训练核心文件，不重新运行编码器或 MLM 变换。

在固定 128 条训练样本的确定性诊断中，full 打分层初始梯度范数从 0.0572 增至 0.4921。这个结果只证明局部优化信号量级改变，不证明准确率收益，需由本轮源验证实验检验。

**对照含义**：中心化本身读取整个候选集合。中心化后的 self/MLP 已使用集合信息；此时 self 的含义是没有跨选项注意力，而不是端到端独立选项评分。self/full 仍使用相同输入、参数和训练设置，仅注意力掩码不同。

## 运行与产物

解压到服务器 `unimc_model_optimization`，运行：

```bash
sbatch slurm/run_centered_heads.sbatch
```

1 GPU，4 CPU，16G，最长 1 小时。日志为项目根目录 `unimc_center_JOBID.out/.err`。中心化缓存写入 `cache/centered_mlm_JOBID`，结果写入 `runs/centered_mlm_JOBID`，包含检查点、训练曲线、每题验证预测、comparison.csv/json。

完成标记：`CENTERED HEAD EXPERIMENT COMPLETE`。所有结果与 baseline 同时报告。

新的缓存签名记录中心化和父特征签名，检查点可继续通过已有 `predict_head.py` 使用；推理必须提供同样中心化后的缓存。把新检查点直接用于未中心化缓存会被签名检查拒绝。后续校准/目标缓存也应先经过同一中心化脚本；改变实际候选集仍需重新运行原始编码器，不能只删缓存行来代表新输入。

检查已覆盖均值为零、共同平移不变性、候选换序、标签无关、原始 logits/指标保持不变、通用训练/导出接口及错误输入签名拒绝。未在本机训练真实全量中心化模型。
