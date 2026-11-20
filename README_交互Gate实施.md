> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

# 交互 Gate 补充实验：实施方法

本轮采用已讨论的最小补充路线：在**当前中心化 MLM → 64 维头部**上新增交互 Gate，复用已有 centered MLP/self/full 三种子对照，只新增 seed42/43/44 三次训练。它不是原始 4096→256 版本的 Gate 实验；原始版本作为之前的探索记录保留。

## 1. 先明确这次改了什么

同一题的选项表示为输入，使用同一个 `MultiheadAttention` 模块计算两个结果：

```text
self_message = Attention(只看本选项)
full_message = Attention(允许看所有有效选项)
g = sigmoid(alpha)
message = self_message + g * (full_message - self_message)
```

其后继续使用原来的残差连接、FFN 和评分层，最终 `logits = base_logits + delta`。评分层零初始化，初始保持 baseline。

`g` 是一个全局可学习标量，初始固定为 0.5，只增加 1 个参数，full/self 原来 58,689 个参数，新增模型共 58,690 个参数。它不随题目变化，不是置信度、温度或拒识概率。Gate 控制的是两种注意力结果之间的混合，不乘在整个最终修正分数上。

固定 g=0/1 的实现检查分别对应 self/full。训练模型则始终学习 sigmoid(alpha)。中心化本身仍包含集合信息，g=0 仅关闭新增的跨选项注意力。

实现文件全部新增：

| 文件 | 作用 |
|---|---|
| `interaction_gate_head.py` | 新模型及严格的加载接口，继承原评分头，保留旧代码 |
| `train_interaction_gate.py` | 与原训练协议一致的新增训练，记录 Gate 值、梯度和初始化核对 |
| `predict_interaction_gate.py` | 同时支持新 Gate 与旧头部，导出同格式未校准 logits |
| `run_interaction_gate.py` | 核对旧实验、训练三个种子、复用对照、统一导出与汇总 |
| `tests/test_interaction_gate.py` | 端点、梯度、padding、换序、保存加载和人工数据流程检查 |
| `slurm/run_interaction_gate.sbatch` | 正常运行 |
| `slurm/recover_interaction_gate.sbatch` | 三次训练均完成、后续导出失败时，仅恢复导出汇总 |

原 `decision_head.py`、`train_head.py`、`predict_head.py`、投影和中心化代码均不改写。不能用旧 `train_head.py --gate` 或旧 `predict_head.py` 代替本轮新入口。

## 2. 公平比较如何保持

固定现有 source_train 共 4,000 道、source_val 共 1,517 道，复用 `cache/centered_mlm_55752` 四个 train/val 缓存。训练和选模型不读取 source_cal 或目标数据。

设置与参考相同：d=64、heads=4、FFN=256、dropout=0.1、lr=1e-4、weight_decay=0.01、batch=128、最多10轮、patience=2、任务均衡采样、梯度裁剪1.0，按源验证 Macro Accuracy 选模型，同分按较低 Macro NLL。

每个种子从头初始化，不从已有最佳权重继续微调。训练前逐个参数核对：除新增 alpha 外，初始权重必须与该种子旧 full 的 `initial.pt` 完全相同；缓存 SHA、数据角色与划分也必须相同。若核对不通过，脚本停止。

两个注意力结果共用一套权重，但需要计算两次注意力。它不是两层 Transformer，也不是零额外计算。两次注意力的 dropout 随机抽样路径与旧单分支不同，因此这里保持相同初始化、采样协议和训练设置，不声称逐步使用完全相同的 dropout 掩码。新旧头部独立训练后再比较。

## 3. 上传补充包

在**本地 Windows PowerShell**执行：

```powershell
cd "WORKSPACE\output"
scp .\UniMC_interaction_gate_starter.zip USER@CLUSTER_HOST:/path/to/unimc_project/
```

切换到 SSH 服务器窗口执行：

```bash
module load anaconda
eval "$(conda shell.bash hook)"
conda activate unimc
cd /path/to/unimc_project/unimc_model_optimization

python -m zipfile -e ../UniMC_interaction_gate_starter.zip .
sbatch slurm/run_interaction_gate.sbatch
```

作业请求 MGPU-TC2/normal、1 GPU、4 CPU、16G、最长2小时。先在计算节点运行10项代码检查，再核对真实缓存与端点，然后新增三次训练。无需重新下载模型或重新编码，不要在登录节点直接运行训练脚本。

已有参考必须保留：

```text
cache/centered_mlm_55752/{piqa,cqa}_{train,val}.pt
runs/pilot_55518/experiment_manifest.json
runs/centering_confirm_55757/comparison.json
```

比较表中九个 centered 参考的 `run_dir` 也必须存在，包括 seed42 的 `runs/centered_mlm_55752/` 和 seed43/44 的确认实验目录。full 三个参考还需保留 `initial.pt`。脚本会提前检查。

## 4. 怎样查看进度

`sbatch` 会返回本轮作业号。下面用 12345 举例，实际替换为返回的数字：

```bash
squeue -j 12345
tail -n 40 ../interaction_gate_12345.out
tail -n 30 ../interaction_gate_12345.err
```

正常日志依次包含：

```text
NONZERO-SCORER GATE ENDPOINT CHECK PASSED
TRAINING interaction_gate seed=42 gate_init=0.5
TRAINING interaction_gate seed=43 gate_init=0.5
TRAINING interaction_gate seed=44 gate_init=0.5
GROUP ...
PAIRED gated minus full ...
GATE seed=42 g=...
INTERACTION GATE EXPERIMENT COMPLETE
```

训练期间每轮记录 train CE、源验证指标、Gate 训练前后数值，以及 Gate 梯度绝对值。初始评分器为零，第一步 Gate 梯度为零是允许的；后续是否有非零梯度会写入 summary，不能只凭 g 是否接近1判断有效。

## 5. 会生成哪些结果

服务器代码目录中的 `runs/interaction_gate_本轮作业号/` 包含：

- `gated_seed42/43/44/`：初始及最佳检查点、config、history、summary。
- `endpoint_check.json`：在真实 PIQA/CQA 输入上，用非零评分器核对固定 Gate 两端。
- `predictions/`：baseline一份，MLP/self/full/gated各三个种子的未校准逐题预测，共13份。
- `comparison.json`、`seed_results.csv`、`report.md`：指标、Gate 数值、纠正/破坏数和配对比较。

成功后自动生成项目父目录：

```text
/path/to/unimc_project/UniMC_interaction_gate_results.zip
```

压缩包包含上述报告、当前完整预测、12个头部最佳检查点、训练记录与所需代码，不包含基座和缓存。正常原版预测由同一缓存的原始分数生成。旧模型目录和之前的交付 ZIP 保持原样。

完成后在**本地 PowerShell**拷回：

```powershell
cd "WORKSPACE\output"
scp USER@CLUSTER_HOST:/path/to/unimc_project/UniMC_interaction_gate_results.zip .
```

## 6. 怎样解释结果

首要看同种子 `gated − full`；另外比较 `gated − self`、`gated − MLP`，而不是仅挑 Gate 最好的种子与原版比较。每种头部报告三个种子的均值和样本标准差，记录各源任务准确率、Macro Accuracy/NLL、纠正数和破坏数。

脚本给出20,000次逐题配对 bootstrap：按任务分层，同题的三个种子整体携带，模型固定，不把三个预测当作三道独立题。源开发集经过多轮探索，这些区间只作探索分析，不能替代独立任务验证。

如果 Gate 几乎不更新、没有纠正更多题或比 full 更差，都保留结果。不要看到结果后反复更换 Gate 初值来追逐这批源验证集的小幅收益。无论结果如何，本轮交付的是可复现的 Gate 扩展证据；独立任务、校准与拒识仍按团队统一协议评估。

## 7. 仅恢复导出

只有当三个 `gated_seed*/summary.json` 都已产生、日志显示后续导出/汇总失败时，使用：

```bash
sbatch slurm/recover_interaction_gate.sbatch runs/interaction_gate_原训练作业号
```

该模式不会调用训练入口，要求新旧代码、数据和三次训练都完整一致；导出到新的 `predictions_resume_恢复作业号` 目录，保留旧输出。如果训练尚未完成，不能用这个命令冒充完成。

当前已在本地 CPU 完成代码检查与人工数据训练流程；真实 A40 训练和性能结论须以上述服务器作业结果为准。
