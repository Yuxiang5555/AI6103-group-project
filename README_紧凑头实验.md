> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

# UniMC：紧凑评分头与冻结 MLM 表示的小实验

前两轮已完成：原版源验证集 Macro Accuracy 为约 76.42%；原始 256 维头在 lr=1e-4 时略低，lr=1e-5/3e-5 时与原版持平。这些是当前两项源任务、seed=42 的结果。

本轮检验两个设计假设：减少评分头容量是否有帮助；复用官方已经训练的 MLM 降维变换，再接小头，是否比从原始隐藏向量学习更有效。这是待验证的设计，尚无真实数据收益结论。

固定使用 `cache/pilot_55518` 中的 4,000 条训练缓存和 1,517 条源验证缓存。沿用原来的 lr=1e-4、seed=42、dropout=0.1、AdamW、batch=128、最多 10 轮、patience=2 和源验证集选模规则。不读取校准集或目标任务数据。

比较如下：

| 表示与头 | 实验 | 目的 |
|---|---|---|
| 原始表示 + d=256 | 读取原三组结果 | 已有对照 |
| 原始表示 + d=64，FFN=256 | 各训 mlp/self/full | 考察较小头 |
| 冻结官方 MLM 变换后的表示 + d=64，FFN=256 | 各训 mlp/self/full | 考察复用输出变换及压缩表示的整体方案 |

两个新设置共训练 6 个头，self/full 在各设置内使用相同结构及参数数量，只有注意力掩码不同。若新表示有效，后续可加入同维度随机冻结变换，进一步区分已训练变换与单纯降维的作用。

`project_mlm_features.py` 加载原本地 checkpoint，复用官方 `AlbertMLMHead` 的 dense、activation、LayerNorm，保持冻结。它只对已有选项向量做变换，不执行 ALBERT 编码器，也不重新编码原始文本。

转换前核对 checkpoint/config SHA256，运行官方 MLM decoder 输入及 yes 输出的一致性检查。原始隐藏向量已经存为 FP16，因此另行记录重建 yes 分数与原始分数的误差；训练和 baseline 始终使用保存下来的原始 base_logits。新表示及其父签名会写入缓存，避免混用原始头或其他 checkpoint。

## 运行

将本补丁解压到服务器的 `unimc_model_optimization`，在该目录运行：

```bash
sbatch slurm/run_compact_heads.sbatch
```

脚本加载 `unimc` 环境，使用 1 GPU、4 CPU、16G 内存，最长运行 2 小时。日志写入项目根目录 `unimc_compact_JOBID.out/.err`。

结果目录为 `runs/compact_pilot_55518_JOBID`，里面包含六次训练的检查点/曲线和 comparison.csv/json。投影缓存写入独立的 `cache/mlm_pilot_55518_JOBID`。若失败，保留日志与已生成文件用于诊断。

完成标记：`COMPACT HEAD EXPERIMENT COMPLETE`。汇总同时展示原版、原来的三个大头和本轮六个小头；若原版仍最好，结果会明确标记，没有将零修正 baseline 冒充训练收益。

新转换路径通过了人工模块/特征的回归检查；真实 checkpoint 的一致性检查在服务器作业中执行。
