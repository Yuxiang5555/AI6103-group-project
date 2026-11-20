> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

# 中心化：三个随机种子的配对确认

seed 42 的源验证实验 55752 首次得到小幅正收益：baseline Macro Accuracy 约 76.42%；中心化 MLP/self/full 分别约 76.71%/76.53%/76.69%。提升主要来自 PIQA，CQA 持平或轻微下降。按准确率目前 MLP 最高，按 NLL full 最低；这些结果还不能证明跨选项注意力优于 MLP。

本轮固定现有配置，补做随机种子 43、44。每个新种子同时训练中心化/未中心化两种输入的 mlp/self/full，共新增 12 次训练。seed 42 的六个现有检查点直接复用。最终共有 18 组训练结果。

保持 MLM 表示、d=64、FFN=256、heads=4、dropout=0.1、lr=1e-4、weight_decay=0.01、batch=128、最多 10 轮、patience=2、balanced sampling 和源验证选模规则。固定源训练 4,000、源验证 1,517，复用所有特征，不进行新的数据编码。不使用校准或目标任务。

脚本检查两种输入的样本、标签及原版分数一致，中心化向量确实由原向量减去同题均值得到；同一模式/种子的中心化与未中心化模型初始权重必须完全一致。核心代码、原参考配置和缓存 SHA256 均需保持一致。

## 产物与解释

`seed_results.csv`：18 组的逐种子结果，包括最佳轮次、PIQA/CQA、Macro Accuracy/NLL 以及相对 baseline 纠正/破坏的题目数量。

`comparison.json`：每种输入/模式的三种子均值和样本标准差；同模式同种子的中心化减未中心化配对差值；中心化 full 减 self/MLP 的配对差值。不会挑选表现最好的单个种子来代表整个方法。

标准差描述随机初始化、采样与训练造成的波动，不是泛化置信区间。随机种子不是新的验证样本，不能把同一批题目的三个预测当作独立样本来增加样本量。最终跨任务收益仍需配置固定后的独立目标任务评估。

中心化本身使用候选集合信息。此处 self 的对照含义仍是没有跨选项注意力；不能将它表述为完全独立选项评分。

运行：在服务器 `unimc_model_optimization` 目录执行 `sbatch slurm/confirm_centering.sbatch`。脚本使用 1 GPU、4 CPU、16G，最长 2 小时。

日志：项目根目录 `center_confirm_JOBID.out/.err`。

模型/结果：`runs/centering_confirm_JOBID`。

完成标记：`CENTERING CONFIRMATION COMPLETE`。完成后自动生成项目根目录 `UniMC_centering_confirmation.zip`，包含汇总、训练记录和逐题验证预测，便于直接拷回分析，不含骨干权重、特征缓存、校准或目标数据。
