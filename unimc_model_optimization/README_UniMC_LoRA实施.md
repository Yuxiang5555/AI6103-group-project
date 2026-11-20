> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

# 第②项：UniMC 编码器 LoRA 的实施方法

本轮只改变编码器的可训练范围，保留原 UniMC 编码、特殊注意力、位置编号、MLM 层和 yes-logit 评分。不会接入之前的中心化评分头或 Gate。预训练基座参数冻结，在官方 ALBERT 的物理共享 query/value 线性层加入 LoRA。

## 1. 方法

每个目标线性层改为：

```text
y = W_base x + (alpha/r) B A dropout(x)
```

固定 W_base，只训练 A/B。A 随机初始化，B 为零，因此初始预测严格保持原模型。按实际共享模块注入一次，不能把 ALBERT 的逻辑层数当成独立参数层数。若该模型为一个共享组、一个内层，query/value 两处 r=8 合计131,072个可训练参数；以真实检查日志打印的数量为准。

默认 r=8、alpha=16、LoRA dropout=0.05。实现使用一个小型 PyTorch 包装，不安装 PEFT，不升级已工作的 transformers/Fengshen 环境。[LoRA 原论文](https://arxiv.org/abs/2106.09685)

原有隐藏向量缓存不能作为这轮训练输入。训练时每批次执行原始编码器的前向和反向，LoRA 改变后的表示也会影响原评分。保存的 adapter 只含A/B，加载时仍需要原始基座和同一版本官方代码。

## 2. 固定数据和训练设置

继续用 `data/pilot_seed42_v2`：PIQA/CQA 各2,000条源训练，源验证919/598条。校准集、独立目标任务均不用于训练或选模型。首轮不增加源数据，不加候选扰动，以免同时改变多个因素。

| 设置 | 本轮值 |
|---|---|
| 训练目标 | 原官方 loss：OP分类损失＋MLM损失；后者包含选项yes/no的O-MLM与官方随机掩码的篇章MLM |
| 训练标签填充 | 将官方 collator 的 padding 标签改为−100，填充位置不计入MLM损失 |
| 优化器 | AdamW，lr=1e-4，weight_decay=0.01，梯度裁剪1.0 |
| batch | micro-batch=1，梯度累积16，等效16条/更新 |
| 训练轮数 | 最多3轮，patience=1 |
| 采样 | 与之前一样按源任务均衡抽样 |
| 精度 | 默认BF16训练，FP32验证；所有比较与原始导出保持FP32评分 |
| 显存 | 对实际共享层组使用non-reentrant梯度检查点；检查作业按512长度填充做峰值测试 |
| 选模型 | source Macro Accuracy最高，同分选Macro NLL较低者；不将差于基线的已训练模型替换成原版冒充提升 |
| 种子 | 先42确认运行，再固定配置补43/44并报告全部 |

本轮还原了原生训练目标和在线掩码训练。它与此前只在缓存上训练OP评分头的训练设置有差异；不能声称只改变了一个网络参数而其他训练细节完全相同。它检验的是“用原生目标进行编码器参数高效适应”这条路线。

## 3. 新文件

- `unimc_lora.py`：低秩层、注入与共享层检查点、适配器保存加载。
- `unimc_online.py`：复用官方逐题编码和collator，修正MLM标签填充，在线FP32预测。
- `run_unimc_lora.py`：真实模型可行性检查和单种子训练。
- `predict_unimc_lora.py`：从原始canonical JSONL输出新模型及原版分数，供目标评估/校准使用。
- `collect_unimc_lora.py`：核对并汇总固定配置的三个种子。
- `tests/test_unimc_lora.py`：轻量实现检查；有依赖时还执行一个微型官方ALBERT的原生loss反向测试。
- `slurm/check_unimc_lora.sbatch`、`slurm/train_unimc_lora.sbatch`：两个阶段分别提交。

旧评分头、缓存、Gate、旧实验和之前的交付包不改写。不要运行旧的 `train_head.py --gate` 来代替本轮训练。

## 4. 先上传，再提交真实检查

本地 Windows PowerShell：

```powershell
cd "WORKSPACE\output"
scp .\UniMC_lora_starter.zip USER@CLUSTER_HOST:/path/to/unimc_project/
```

SSH服务器窗口：

```bash
module load anaconda
eval "$(conda shell.bash hook)"
conda activate unimc
cd /path/to/unimc_project/unimc_model_optimization

python -m zipfile -e ../UniMC_lora_starter.zip .
sbatch slurm/check_unimc_lora.sbatch
```

检查请求MGPU-TC2/normal、1 GPU、4 CPU、24G、30分钟，在计算节点运行。不要在登录节点执行真实模型训练。

保留以下已存在的文件：

```text
../models/unimc-albert-235m/
../Fengshenbang-LM-main/
data/pilot_seed42_v2/{piqa,cqa}/{train,val}.jsonl
cache/pilot_55518/piqa_val.pt.metadata.json
runs/centering_confirm_55757/predictions_resume_55764/baseline_val.jsonl
```

检查会核对原基座、分词器和官方编码哈希，对8道真实验证题核对原版分数和零适配器预测；以512长度完成两步原生loss训练，检查梯度、A/B更新、冻结权重哈希不变、关闭适配器后恢复原分数，同时报告峰值显存。这仅是可行性测试，不能用它的loss或预测评价最终性能。

查看日志：

```bash
squeue -u "$USER"
tail -n 40 ../lora_check_*.out
tail -n 40 ../lora_check_*.err
```

成功标记：`UNIMC LORA CHECK COMPLETE`。报告为 `runs/lora_check_作业号/check_report.json`，并自动生成父目录 `UniMC_lora_check.zip`。第一次请先确认检查结果，再提交正式训练。若BF16检查失败或出现显存问题，先根据具体报错修正，不盲目提交三个长作业。

本地尚未运行真实235M模型。轻量检查不能替代上述A40可行性测试。

## 5. 检查通过后，训练第一个种子

在服务器代码目录执行：

```bash
sbatch slurm/train_unimc_lora.sbatch 42
```

脚本自动寻找最新且代码、基座、数据、rank、精度、batch均匹配的已通过检查报告。没有匹配报告就停止；不会加载检查作业更新后的权重，而是从原基座及新的零B初始化开始。

作业上限5小时50分钟，4 CPU、1 GPU、24G。实际耗时由在线编码器训练决定，不承诺与缓存头部实验相同。每100个micro-batch打印进度，每轮完整FP32验证。训练与验证都会明显慢于之前的小头部训练。

```bash
squeue -u "$USER"
tail -n 40 ../unimc_lora_*.out
tail -n 40 ../unimc_lora_*.err
```

日志包括 `MATCHED_FEASIBILITY_CHECK`、`LORA TRAIN`、`LORA VALIDATING`、`LORA EPOCH`，最后是 `UNIMC LORA TRAINING COMPLETE`。

结果目录 `runs/lora_seed42_作业号` 包括config/history/summary、初始/最佳/最后adapter、最佳验证预测及元数据。保存的是适配器，不是完整基座。正常结束后父目录生成 `UniMC_lora_seed42_results.zip`。

若作业因时限或错误中断，现有epoch检查点可以保留，但本版本不支持优化器/RNG状态的断点续训，不能把部分训练冒充完整三轮结果。

## 6. 补43/44和三种子汇总

确认代码与训练流程正常后，保持配置不变补做：

```bash
sbatch slurm/train_unimc_lora.sbatch 43
sbatch slurm/train_unimc_lora.sbatch 44
```

集群每用户只有1 GPU额度，作业按资源排队。三个种子的配置和训练预算一致，不能只报告表现最好的种子。

完成后，把下面三个路径替换成实际结果目录，运行轻量汇总：

```bash
python collect_unimc_lora.py \
  --runs runs/lora_seed42_作业号A runs/lora_seed43_作业号B runs/lora_seed44_作业号C \
  --output runs/lora_three_seed_summary
```

该命令只读取结果、不训练。它核对三次数据、基座、代码、参数和逐题ID一致，复算指标，并生成父目录 `UniMC_lora_three_seed_results.zip`。

任一结果包拷回本地PowerShell，例如第一个种子：

```powershell
cd "WORKSPACE\output"
scp USER@CLUSTER_HOST:/path/to/unimc_project/UniMC_lora_seed42_results.zip .
```

## 7. 对接独立评估与校准

新LoRA模型的分数必须重新在线计算。之前full评分头的calibration文件不能作为LoRA模型的校准输入。对固定后的LoRA模型，在GPU Slurm作业内运行：

```bash
python predict_unimc_lora.py \
  --input data/任务.jsonl --role target_eval \
  --checkpoint runs/lora_seed42_实际作业号/best_adapter.pt \
  --output predictions/任务_lora_seed42.jsonl --device cuda:0
```

源校准文件改成 `--role source_cal`，逐个任务及种子导出。脚本先关闭适配器获取原版分数，再开启适配器，输出同题 `logits`、`base_logits`、`prediction`、`probabilities`、候选及标签。分数仍未校准，source_cal/目标数据不用于本轮训练或选模型。

源验证的提升不保证独立目标任务提升。最终仍需固定配置后的跨任务评估；LoRA只是下一条待检验的方法，不能预先承诺增益。
