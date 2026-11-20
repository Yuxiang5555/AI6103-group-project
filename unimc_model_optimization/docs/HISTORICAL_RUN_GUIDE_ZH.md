# Historical Run Guide

Consolidated historical notes; recorded run IDs and external artifacts retain their original meaning. Current instructions are in OPERATIONS.md.


## LoRA损失诊断

> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

## UniMC LoRA：首轮损失与梯度诊断

### 当前结论

seed42、作业55840的最佳已训练检查点：源验证Macro Accuracy为76.1112%，原版76.4164%，下降0.3052个百分点。检查包与训练配置匹配；两步检查中基座参数哈希保持不变，A/B参数正常获得梯度并更新，512长度测试峰值已分配显存1.5545 GiB。这证实了两步技术检查通过，尚未解释准确率下降。

本补丁只新增诊断脚本、测试、Slurm脚本和本文档。保留原LoRA代码、签名、检查点和结果，当前LoRA也不替换先前的centered full候选。

### 诊断内容

从PIQA、CommonsenseQA的`source_train`中各固定抽16题，采样种子20261118。一次生成输入掩码，随后复用同一组输入，检查初始、第1轮最佳、第2轮最后三个adapter。不同检查点使用相同的逐题Dropout随机种子。

正式诊断的整个前向和损失计算统一为FP32，关闭autocast。它是对已有检查点的FP32事后分析，并非复现原BF16训练步骤。开始时额外对第一题做原训练AMP模式和FP32两次无梯度前向，保留官方loss、同logits重算的FP32 loss、实际dtype和差值，写入`precision_probe.json`并打印`PRECISION PROBE`。

记录三个目标：

1. OP：正确选项位置的分类交叉熵。
2. 选项O-MLM：选项标记处yes/no词的交叉熵。
3. 篇章MLM：其余被监督token处的交叉熵。

官方联合目标为`OP + MLM`，MLM在所有有效目标token上取平均。因此两个MLM部分必须按各自目标token数量加权，不能直接把它们各自的平均loss相加。正式FP32诊断逐题严格核对加权MLM与同logits的完整FP32交叉熵、官方FP32 loss一致，原容差2e-5/1e-5保留，并记录原始值、加权值、目标数量。

在同一次FP32前向上，分别计算三个部分对LoRA A/B的梯度，记录范数、MLM/OP范数比、各梯度之间的余弦及合成梯度。没有篇章目标的题，其篇章梯度记零，相关余弦记null。汇总同时保留均值和范数比中位数，完整逐题记录也会导出。

没有优化器，优化步数为0；每个检查点诊断后校验adapter未变，全部结束后校验冻结参数哈希未变。仅加载源训练样本；不加载验证、校准或目标任务样本。

这32题是事后探索，不代表训练全程。梯度余弦为负不能直接证明验证准确率下降的原因，也不能证明移除MLM会改善效果。诊断用于决定是否值得做后续受控对照。

### 上传并提交

在本机 **Windows PowerShell** 执行：

```powershell
cd "WORKSPACE\output"
scp .\UniMC_lora_diagnostic_fix.zip USER@CLUSTER_HOST:/path/to/unimc_project/
```

在已经SSH登录的 **Linux终端** 执行：

```bash
module load anaconda
eval "$(conda shell.bash hook)"
conda activate unimc
cd /path/to/unimc_project/unimc_model_optimization
python -m zipfile -e ../UniMC_lora_diagnostic_fix.zip .
sbatch slurm/diagnose_unimc_lora.sbatch
```

默认读取`runs/lora_seed42_55840`，需要保留其中`config.json`及`initial_adapter.pt`、`best_adapter.pt`、`last_adapter.pt`，还需要此前的LoRA实现、模型目录、Fengshen代码和原源训练文件。训练文件及实现签名不匹配会停止。目录不同可传入实际目录：

```bash
sbatch slurm/diagnose_unimc_lora.sbatch runs/lora_seed42_实际作业号
```

脚本申请1张GPU、4核CPU、24G内存、30分钟；运行前会先执行五个诊断测试。无需重新安装依赖。此前失败的诊断目录和日志保留，新作业号会创建新的诊断目录；只重跑诊断，不重跑训练。

### 查看结果

```bash
squeue -u "$USER"
grep -E 'PRECISION PROBE|LOSS DIAGNOSTIC|COMPONENT|COMPLETE|REPORT_ARCHIVE' ../lora_loss_diag_*.out | tail -n 25
tail -n 40 ../lora_loss_diag_*.err
```

以`UNIMC LORA LOSS DIAGNOSTIC COMPLETE`为成功标志。队列消失只表示作业已退出，需结合日志判断是否成功。

诊断目录为`runs/lora_loss_diag_作业号`，包含`precision_probe.json`、`diagnostic.json`、`per_example.csv`、`report.md`。打包文件为`/path/to/unimc_project/UniMC_lora_diagnostics.zip`，包含上述结果及实际运行的诊断脚本。

成功后在本机 **Windows PowerShell** 拷回：

```powershell
cd "WORKSPACE\output"
scp USER@CLUSTER_HOST:/path/to/unimc_project/UniMC_lora_diagnostics.zip .
```

本地五个测试覆盖加权MLM损失/梯度还原、无篇章目标的分支、共享层梯度检查点上多次求梯度后参数保持不变、低精度scalar与FP32结果的差异记录，以及错误拆分仍被严格拦截。真实A40上的修复版诊断尚需运行，当前不报告任何未得到的诊断结论。

### 修复记录

首版服务器任务在MLM核对处停止：重算1.123233676，官方返回1.123046875，差约0.0001868（此前说明曾误写为0.00187）。首版本地三个测试只覆盖FP32，未发现该比较风险。仅凭截图无法确认服务器差异的全部来源，修复版保留dtype/数值证据，并把正式诊断统一为FP32；没有放宽正式核对容差。原训练实现和检查点不修改，失败任务没有优化器或参数更新。

2026-11-18返回结果已复核：96条FP32正式诊断中官方loss与完整FP32重算差值均为0，加权拆分最大误差约2.38e-7。三个adapter及配置哈希与作业55840一致，32题抽样ID已在本地重现。完整审计见工作区output下`UniMC_lora_diagnostic_analysis_55840/LoRA损失诊断_复核结论.md`。本地未重跑完整编码器。

## 中心化三种子确认

> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

## 中心化：三个随机种子的配对确认

seed 42 的源验证实验 55752 首次得到小幅正收益：baseline Macro Accuracy 约 76.42%；中心化 MLP/self/full 分别约 76.71%/76.53%/76.69%。提升主要来自 PIQA，CQA 持平或轻微下降。按准确率目前 MLP 最高，按 NLL full 最低；这些结果还不能证明跨选项注意力优于 MLP。

本轮固定现有配置，补做随机种子 43、44。每个新种子同时训练中心化/未中心化两种输入的 mlp/self/full，共新增 12 次训练。seed 42 的六个现有检查点直接复用。最终共有 18 组训练结果。

保持 MLM 表示、d=64、FFN=256、heads=4、dropout=0.1、lr=1e-4、weight_decay=0.01、batch=128、最多 10 轮、patience=2、balanced sampling 和源验证选模规则。固定源训练 4,000、源验证 1,517，复用所有特征，不进行新的数据编码。不使用校准或目标任务。

脚本检查两种输入的样本、标签及原版分数一致，中心化向量确实由原向量减去同题均值得到；同一模式/种子的中心化与未中心化模型初始权重必须完全一致。核心代码、原参考配置和缓存 SHA256 均需保持一致。

### 产物与解释

`seed_results.csv`：18 组的逐种子结果，包括最佳轮次、PIQA/CQA、Macro Accuracy/NLL 以及相对 baseline 纠正/破坏的题目数量。

`comparison.json`：每种输入/模式的三种子均值和样本标准差；同模式同种子的中心化减未中心化配对差值；中心化 full 减 self/MLP 的配对差值。不会挑选表现最好的单个种子来代表整个方法。

标准差描述随机初始化、采样与训练造成的波动，不是泛化置信区间。随机种子不是新的验证样本，不能把同一批题目的三个预测当作独立样本来增加样本量。最终跨任务收益仍需配置固定后的独立目标任务评估。

中心化本身使用候选集合信息。此处 self 的对照含义仍是没有跨选项注意力；不能将它表述为完全独立选项评分。

运行：在服务器 `unimc_model_optimization` 目录执行 `sbatch slurm/confirm_centering.sbatch`。脚本使用 1 GPU、4 CPU、16G，最长 2 小时。

日志：项目根目录 `center_confirm_JOBID.out/.err`。

模型/结果：`runs/centering_confirm_JOBID`。

完成标记：`CENTERING CONFIRMATION COMPLETE`。完成后自动生成项目根目录 `UniMC_centering_confirmation.zip`，包含汇总、训练记录和逐题验证预测，便于直接拷回分析，不含骨干权重、特征缓存、校准或目标数据。

## 恢复导出

> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

## 三种子确认：恢复预测导出与统计

作业 55757 已完成 12 次新训练，并在导出第一个 baseline 预测的子进程启动阶段失败。错误为 `MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1`，与 Intel/GNU OpenMP 运行时冲突有关。

修复只修改确认流程，不修改 decision_head.py、train_head.py、data_io.py 或 predict_head.py：Linux 预测子进程显式使用 `MKL_THREADING_LAYER=GNU`；新增 `--report-only` 恢复模式。

在项目的 `unimc_model_optimization` 解压补丁后运行：

```bash
sbatch slurm/recover_centering.sbatch
```

该作业固定读取 `runs/centering_confirm_55757`，只核验和导出已有检查点，不调用训练器。它会先检查所有 18 组的配置、源数据 SHA256、initial.pt、best.pt、history.json 和 summary.json；缺少任何必要结果时直接报错，绝不自动重训。

为保留先前的部分输出，新预测写到原结果目录下独立的 `predictions_resume_恢复JOBID`。原确认 manifest 保留，另写恢复版本记录。ZIP 只收集本次完整预测、当前汇总和训练记录，不混入旧的部分预测。

日志：项目根目录 `center_recover_JOBID.out/.err`。1 GPU、4 CPU、16G，最长 30 分钟。GPU 用于已有小头的验证推理。

完成标记仍为 `CENTERING CONFIRMATION COMPLETE`，自动生成项目根目录 `UniMC_centering_confirmation.zip`。

常规新作业的 slurm/confirm_centering.sbatch 也已修正线程环境；本次恢复请使用 recover_centering.sbatch，避免创建新的训练实验。

本地检查覆盖了 Linux 子进程线程环境修复、父进程环境保留、Windows 兼容、恢复不调用训练器、缺失结果拒绝、保留部分预测文件、配对种子与统计计算。未在本机复现服务器的 Linux MKL 组合；服务器恢复作业会实际验证预测导出。

参考：Intel mkl-service 对线程层与已加载 OpenMP 库的检查实现：
https://github.com/IntelPython/mkl-service/blob/master/mkl/_mklinitmodule.c

## 执行步骤

> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

## UniMC 模型优化：执行指南与起始代码

本目录实现你们已确定的模型优化路线：冻结英文 UniMC，缓存选项表示，训练一个轻量残差决策层。主要研究问题是：相同决策层允许选项之间交互后，是否能比禁止交互的版本取得更好的跨任务分类结果。

本目录不包含置信度校准模块。所有预测导出原始 logits，校准组可以独立接入。

### 0. 代码已经验证到哪里

- 在本机 Python 3.10、PyTorch 2.14.1 CPU 环境通过 13 项单元测试，包括实际 4096 维输入的前向和反向计算。
- 用人工构造的特征跑通三个版本的训练、源验证集选模型、保存/加载检查点、目标 logits 导出及原版分数导出。
- 尚未在本机下载真实 UniMC 权重、运行 Fengshen 特征提取或测试 A40。`extract_features.py` 是基于官方类接口编写的适配器，需要在你们已经跑通官方 UniMC 的环境中确认。
- 人工特征的检查结果只证明工程流程可运行，不能当作 UniMC 性能提升结果。
- 数据模板为固定的项目模板，不声称完全复现论文逐任务最佳模板。首批官方接口一致性检查，验证的是相同输入模板下两种读取方法一致。

### 1. 先明确模型优化组的边界

输入：固定数据划分、官方 UniMC、统一预处理。

输出：三个新增评分器、训练代码、最佳源验证检查点、预测接口和小实验结果。

主要比较：

| 名称 | 实现 | 作用 |
|---|---|---|
| baseline | 原始 UniMC 的 yes logits | 保留原始参照 |
| mlp | 每个选项独立经过共享 MLP | 检查重新评分的作用 |
| self | Transformer 只允许选项关注自身 | 与 full 保持相同参数结构 |
| full | 相同 Transformer 允许读取其他有效选项 | 检查显式选项交互的作用 |

`self` 与 `full` 使用同一个类，只有注意力掩码不同。相同种子下，初始参数也相同。两者都训练之后再比较，不能直接在一个训练好的 full 模型上关闭交互，代替独立训练的 self 对照。

Gate 为可选扩展，不影响先完成上述四组。

### 2. 文件分工

```text
prepare_data.py        本地数据转换、源验证/校准划分
prepare_pilot_data.py  自动下载 PIQA/CQA，排除训练池与 holdout 重叠后固定小实验划分
extract_features.py    复用官方编码和模型，提取选项表示与分数
decision_head.py       MLP / self / full 共用的残差评分框架
data_io.py             缓存契约、校验、batch 和源验证评测
train_head.py          新决策层训练、早停、检查点与配置记录
predict_head.py        导出未校准 logits 及分类指标
tests/test_heads.py    掩码、初始化、换序、padding、梯度及标签检查
smoke_test.py          不需要模型权重的人工特征端到端检查
requirements.txt      小模块所需依赖
```

模型优化 A：负责准备/核对输入，跑通提取，缓存，MLP 和训练入口。

模型优化 B：负责 self/full 两组、掩码检查、超参数小实验和可选 Gate。

两人一起核对共享设置及交付文件，后续汇总实验组统一跑正式大实验。

### 3. 环境：先验证小模块，再接真实模型

以下服务器命令按 Linux bash 编写，工作目录为本目录。Windows 上的单元测试、训练及预测命令也可用 Python 执行。

已有可用 PyTorch 环境时先运行：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m unittest discover -s tests -v
python smoke_test.py
```

需要新建小模块环境时：

```bash
python3 -m venv .venv-head
.venv-head/bin/python -m pip install -r requirements.txt
.venv-head/bin/python -m unittest discover -s tests -v
.venv-head/bin/python smoke_test.py
```

下文用 `python` 表示小模块环境的解释器；如果采用上述虚拟环境，可换成 `.venv-head/bin/python`。

特征提取需要另一个前提：你们的官方 UniMC 已经能在服务器正常预测。沿用该环境，设置两个服务器路径：

```bash
UNIMC_PYTHON=/实际路径/官方环境/bin/python
FENGSHEN_REPO=/实际路径/Fengshenbang-LM
```

`extract_features.py` 直接使用官方 `UniMCDataset`、`UniMCDataModel`、`UniMCModel` 和英文 ALBERT 类。官方代码的历史依赖需由已有可运行环境保证；本目录没有把历史 Fengshen 全套依赖验证成新的安装锁文件。

官方入口：
- https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/unimc
- https://huggingface.co/IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English

### 4. 先做 PIQA + CommonsenseQA 小实验

它们分别主要是 2 选 1 和 5 选 1，适合先验证可变候选数量。正式实验再加入 ARC-Challenge、OpenBookQA。

可以先用标准库脚本自动准备小实验：

```bash
python -u prepare_pilot_data.py --output data/pilot_seed42_v2
```

脚本复用 `raw/pilot_seed42` 中已下载的原始文件，剔除候选文本重复的歧义题；保留源 holdout 池，从源训练池排除与 holdout 的 `content_hash` 完全相同的题目，再按种子 42 抽取最多 2,000 条训练数据。holdout 等分为 val/cal，清理数量和被排除题目的 ID 写入 `provenance.json`。新输出目录必须不存在；直接运行下方通用转换器时，训练池与 holdout 重叠仍会报错。

准备下面的本地文件：

```text
raw/piqa/train.jsonl
raw/piqa/train-labels.lst
raw/piqa/dev.jsonl
raw/piqa/dev-labels.lst
raw/cqa/train.jsonl
raw/cqa/dev.jsonl
```

PIQA 原始 train/dev 压缩包的地址见其公开数据加载脚本：
https://huggingface.co/datasets/ybisk/piqa/raw/main/piqa.py

其中引用的公开包为：
https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip

CommonsenseQA 可以由数据组从公开数据集导出 train / validation 为 JSONL：
https://huggingface.co/datasets/tau/commonsense_qa

例如，在安装了 `datasets` 的数据环境中：

```python
from pathlib import Path
from datasets import load_dataset

Path("raw/cqa").mkdir(parents=True, exist_ok=True)
ds = load_dataset("tau/commonsense_qa")
ds["train"].to_json("raw/cqa/train.jsonl")
ds["validation"].to_json("raw/cqa/dev.jsonl")
```

不要使用标签为 -1 或没有 answerKey 的隐藏测试划分。

转换命令：

```bash
python prepare_data.py --task piqa --role source --input raw/piqa/train.jsonl --labels raw/piqa/train-labels.lst --holdout raw/piqa/dev.jsonl --holdout-labels raw/piqa/dev-labels.lst --output data/piqa --seed 42 --limit-train 2000
python prepare_data.py --task commonsenseqa --role source --input raw/cqa/train.jsonl --holdout raw/cqa/dev.jsonl --output data/cqa --seed 42 --limit-train 2000
```

每个目录生成 train.jsonl、val.jsonl、cal.jsonl 和 manifest.json。

- train：训练新头。
- val：调超参数、早停、选择检查点。
- cal：只交给校准组拟合温度，不参与本组模型选择。
- 这里把独立源 holdout 池等分为 val/cal；应先核实原 UniMC 是否用过该池。程序只保证新头训练与 holdout 不重叠，不能证明旧权重没有见过该数据。
- 正式全量数据请另建目录，去掉 `--limit-train`。新输出目录必须不存在，避免混淆不同划分。
- 两个任务抽样都固定种子；先把 manifest 发给校准组和大实验组。

规范化后每行格式为：

```json
{"id":"train:example-1","dataset":"piqa","role":"source_train","texta":"A question or context","textb":"","question":"","choices":["option one","option two"],"gold_label":1,"content_hash":"..."}
```

`gold_label` 是 0 开始的选项序号，不是文本 token 位置。ARC/CQA/OBQA 的 answerKey 通过实际候选键映射，不能一律假设键是 A/B/C/D。

### 5. 先提取 64 条，验证官方输出一致

```bash
"$UNIMC_PYTHON" extract_features.py --fengshen-root "$FENGSHEN_REPO" --input data/piqa/train.jsonl --output cache/pilot_piqa.pt --limit 64 --batch-size 4 --verify-official --device cuda
```

第一次可能从 Hugging Face 下载 checkpoint；也可以用 `--model /本地完整模型目录`。建议固定 Hub `--revision` 为实际 commit，并保留下载后的目录。缓存还会记录权重 SHA256、tokenizer、代码与配置摘要。

核对生成的 `cache/pilot_piqa.pt.metadata.json`：

- `official_prediction_check.passed` 为 true，首批概率和预测与官方接口一致。
- `feature_signature.hidden_dim` 是实际骨干隐藏维度，英文 xxlarge 为预期的 4096。
- `role` 为 source_train，`n` 为 64。
- 默认 FP32 前向，缓存的选项隐藏表示为 FP16，原始 logits 为 FP32。

该一致性检查需要 CUDA，因为官方预测类内部直接使用 `.cuda()`。本脚本从 ALBERT 主干输出提取最后一层隐藏表示，不额外保留每层、每个 token 的全部特征。

如果失败，先修复模型加载、官方预处理或数值一致性，再训练新头。先保持 `--amp none`；后续若要加速，所有缓存采用相同精度设置并重新核对。

官方预处理可能对过长选项直接截短。本脚本默认遇到这类样本就报出 ID，供人工检查；只有明确接受后才加 `--allow-option-truncation`，并在所有组保持相同设置。

### 6. 提取固定缓存

先提取小实验所需的 train/val，cal 可以一并准备：

```bash
for task in piqa cqa; do
  for split in train val cal; do
    "$UNIMC_PYTHON" extract_features.py --fengshen-root "$FENGSHEN_REPO" --input "data/$task/$split.jsonl" --output "cache/${task}_${split}.pt" --batch-size 4 --verify-official --device cuda
  done
done
```

耗时主要在这里。先测 64 条的用时，再决定是否调整提取 batch size。后续 mlp/self/full 直接共用缓存，不重跑骨干。

缓存只保留选项标记处特征：hidden `[K,H]`、base_logits `[K]`、真实标签、候选顺序和 ID。

不要直接删掉缓存中的一个选项，声称等同于重新编码了新的候选集合。候选变化可能通过共享上下文影响骨干表示；这种扩展实验要重新提取特征。

### 7. 新模块结构

对每个选项：

```text
hidden [4096]
    -> LayerNorm
    -> Linear(4096,256)
    -> MLP / self-attention block
    -> LayerNorm
    -> Linear(256,1)
    -> delta
final_logit = base_logit + delta
```

Transformer 用一层、4 个头、FFN 1024、GELU、dropout 0.1，不添加选项序号位置嵌入。最后 Linear 权重和偏置为零，所以训练第 0 步精确还原基线。

普通候选 K=3 时，允许关注的位置示意如下（1=允许）：

```text
self                   full
1 0 0                  1 1 1
0 1 0                  1 1 1
0 0 1                  1 1 1
```

实现已经处理变长候选与 padding。补齐选项不会被有效候选读取；padded query 保留可读取的有效 key，避免产生整行全部被屏蔽的 softmax。

注意 PyTorch MultiheadAttention 的布尔 mask 中 True 表示禁止关注。

### 8. 训练三组，先固定其它设置

```bash
for mode in mlp self full; do
  python train_head.py --train cache/piqa_train.pt cache/cqa_train.pt --val cache/piqa_val.pt cache/cqa_val.pt --mode "$mode" --output "runs/${mode}_seed42_lr1e4" --lr 0.0001 --epochs 10 --patience 2 --batch-size 128 --seed 42 --device auto
done
```

优化器 AdamW，weight decay 0.01，梯度裁剪 1.0。仅新头参数进入优化器。

损失直接对最终 logits 使用交叉熵，不先 softmax，不重算 MLM/O-MLM。

默认 balanced sampling：按数据集大小的倒数给样本权重，有放回抽样，使各任务获得大致相同的训练机会。所有组采用相同规则。

选模型规则提前固定为：源验证各任务 Accuracy 的算术平均，若相同再比较源验证平均 NLL；不查看目标任务结果选配置。

每个 run 包含：

```text
config.json     配置、缓存摘要、参数量和环境
initial.pt      修正为零的初始化检查点
best.pt         最佳“已训练”检查点
history.json    每轮训练 CE、源验证结果和耗时
summary.json    与原版基线的源验证对比
```

程序不把原版基线悄悄当成训练后的 best.pt。若新头不如原版，summary 会保留两者结果。

### 9. 小实验如何判断下一步

先看：

1. 有限损失和梯度，训练程序能保存并加载检查点。
2. MLP 与原版：重新评分是否有帮助。
3. full 与 self：参数结构相同的情况下，选项交互是否有额外作用。
4. PIQA 与 CQA 是否出现不同趋势，是否一个任务主导整体结果。

第一次只比较学习率 1e-4 / 3e-4，其他参数固定。源验证选择好后，把主要版本用种子 13、42、2026 复跑。正式种子可以由全组另行统一，但所有模型必须相同。

检查尺度：新 delta 是否快速大到覆盖原分数，是否训练准确率上升而源验证持续恶化。先用早停和源验证比较控制；不据目标结果逐步挑选结构。

可以在 full 基础上加 `--gate`：`final_logit = base_logit + sigmoid(gate_logit) * delta`。这是修正幅度门控，不等同于只控制跨选项注意力。Gate 属于额外实验；主比较仍是 mlp/self/full。

### 10. 扩展四个源任务

ARC-Challenge 使用 `--task arc`，OBQA 使用 `--task openbookqa`，支持官方 question/stem/choices 对象和常见 HF 导出字段。

按相同规则生成 train/val/cal 后提取缓存；训练命令的 `--train`、`--val` 后面追加两个任务的对应文件。必须保证：

- 同一官方 checkpoint、同一编码设置、同一缓存精度。
- 相同训练数据、相同训练/验证划分、相同抽样、相同超参数搜索预算。
- self/full 不改维度或额外加层来制造差异。

正式目标评估与大实验由汇总组运行。目标结果不回流到你们的训练和模型选择。

### 11. 给汇总组、校准组导出 logits

例如，目标 HellaSwag 的公开有标签验证集由数据组导出到本地，再转换：

```bash
python prepare_data.py --task hellaswag --role target --input raw/hellaswag_validation.jsonl --output data/hellaswag
"$UNIMC_PYTHON" extract_features.py --fengshen-root "$FENGSHEN_REPO" --input data/hellaswag/eval.jsonl --output cache/hellaswag_eval.pt --batch-size 4 --verify-official --device cuda
python predict_head.py --cache cache/hellaswag_eval.pt --baseline --output predictions/base_hellaswag.jsonl
python predict_head.py --cache cache/hellaswag_eval.pt --checkpoint runs/full_seed42_lr1e4/best.pt --output predictions/full_seed42_hellaswag.jsonl
```

SST-2、CB、COPA 分别使用 `--task sst2/cb/copa`。这些转换固定了本项目模板；原版与新头始终共用同一个目标缓存。SST-2 的负面 label=0 会映射到第二个选项（terrible），不是第一个。

校准组得到每个版本在同一批源 cal 样本上的 logits，以及对应目标输出。例如：

```bash
python predict_head.py --cache cache/piqa_cal.pt cache/cqa_cal.pt --checkpoint runs/full_seed42_lr1e4/best.pt --output predictions/full_seed42_sourcecal.jsonl
python predict_head.py --cache cache/piqa_cal.pt cache/cqa_cal.pt --baseline --output predictions/base_sourcecal.jsonl
```

每个模型自己拟合温度，源 cal 行按相同 ID 对齐。目标 gold_label 只供最终指标计算，不能用来拟合温度。

导出记录示意：

```json
{"id":"eval:example-1","dataset":"hellaswag","gold_label":2,"choices":["...","...","...","..."],"num_options":4,"logits":[1.2,0.7,2.0,-0.1],"base_logits":[1.1,0.8,1.7,0.0],"prediction":2}
```

这里只展示主要字段；完整输出还包含正确计算的 softmax probabilities。校准应读取 logits，不把 probabilities 再当 logits。

旁边的 `.metrics.json` 记录 Accuracy、各数据集结果与 NLL；`.metadata.json` 记录缓存、权重摘要和角色。预测保持输入顺序，校准只需要有效选项，不包含 padding。

### 12. 模型优化组最终交付

- 本目录源码，以及实际运行环境的版本记录。
- 固定数据 manifest、缓存 metadata 和提取精度。
- 原版、MLP、self、full 的小实验结果；Gate 若完成则附上。
- 每个模型的 config.json、best.pt、history.json、summary.json。
- 一条可以复现训练的命令，一条可以导出 logits 的命令。
- 方法说明：冻结骨干、残差评分、选项集合交互、对照掩码和选模型规则。

缓存上的耗时不是完整部署耗时。大实验组测延迟/显存时，需要把真实骨干前向加上新增决策层，并固定 batch、输入长度与数值精度。

### 13. 故障定位

| 现象 | 首先检查 |
|---|---|
| 官方库 ImportError | 是否使用原来能跑通 UniMC 的解释器、仓库与依赖 |
| 提取分数与官方不一致 | 模型 eval、yes token、position_ids、3D attention mask、AMP 设置和候选索引 |
| 标签越界/错误 | 使用选项序号而非 token 位置；保存原始 gold_label；answerKey 按实际键映射 |
| loss/梯度 NaN | 原始特征是否有限、候选 mask、是否出现整行被屏蔽的 attention |
| 候选换序改变输出 | 新模块是否引入了序号嵌入、mask 是否随候选同步、是否忘记 eval |
| 新头训练不更新 | 首步 scorer 梯度是否存在；零输出初始化时上游梯度首步为零是正常的，随后应有更新 |
| source_cal/target_eval 被拒绝训练 | 这是角色隔离；使用 source_train 与 source_val，不要改角色来绕过 |
| 目标提升不稳定 | 汇总组比较相同种子的 self/full，查看各任务及答对数量；不按单次目标结果选结构 |

## 紧凑头实验

> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

## UniMC：紧凑评分头与冻结 MLM 表示的小实验

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

### 运行

将本补丁解压到服务器的 `unimc_model_optimization`，在该目录运行：

```bash
sbatch slurm/run_compact_heads.sbatch
```

脚本加载 `unimc` 环境，使用 1 GPU、4 CPU、16G 内存，最长运行 2 小时。日志写入项目根目录 `unimc_compact_JOBID.out/.err`。

结果目录为 `runs/compact_pilot_55518_JOBID`，里面包含六次训练的检查点/曲线和 comparison.csv/json。投影缓存写入独立的 `cache/mlm_pilot_55518_JOBID`。若失败，保留日志与已生成文件用于诊断。

完成标记：`COMPACT HEAD EXPERIMENT COMPLETE`。汇总同时展示原版、原来的三个大头和本轮六个小头；若原版仍最好，结果会明确标记，没有将零修正 baseline 冒充训练收益。

新转换路径通过了人工模块/特征的回归检查；真实 checkpoint 的一致性检查在服务器作业中执行。

## 选项中心化实验

> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

## 选项表示中心化：基于真实诊断的单项对照

原始诊断包：`UniMC_pilot_diagnostics.zip`。固定原实验 55518、调参 55537、紧凑头 55555。

本地逐样本复核发现：参数有实际更新，MLM 表示与原模型输出一致；被检查的低学习率/小头检查点在 1,517 条源验证题上均未改变原版预测。MLM full 小头的选项残差分差中位数为 PIQA 0.000638、CQA 0.004982，而原版 top-2 分差中位数为 0.7811、1.8373。原大 full 头改变了 59 题，纠正 27、破坏 29，另有 3 题错到错，未获得净收益。

同题 MLM 选项向量的余弦相似度中位数为 PIQA 0.9919、CQA 0.9425。共同成分可能妨碍小头学习相对差异，这是本轮待验证的假设。

本轮唯一改变是，对每题真实 K 个选项的 MLM 向量进行中心化：

`centered_h[i] = h[i] - mean(h[0:K])`

然后照常进入已有的 LayerNorm、MLP/self/full 和残差相加。中心化输入向量会改变非线性头的学习；仅从最终 logits 中减均值不会改变 softmax 或预测，不能代替此操作。

固定使用 4,000 条源训练数据、1,517 条源验证数据、MLM 特征、d=64、FFN=256、dropout=0.1、lr=1e-4、seed=42、最多 10 轮、patience=2。仅新增三个头，与已完成的三个未中心化 MLM 小头对照。不使用校准或目标数据，不改变模型/训练核心文件，不重新运行编码器或 MLM 变换。

在固定 128 条训练样本的确定性诊断中，full 打分层初始梯度范数从 0.0572 增至 0.4921。这个结果只证明局部优化信号量级改变，不证明准确率收益，需由本轮源验证实验检验。

**对照含义**：中心化本身读取整个候选集合。中心化后的 self/MLP 已使用集合信息；此时 self 的含义是没有跨选项注意力，而不是端到端独立选项评分。self/full 仍使用相同输入、参数和训练设置，仅注意力掩码不同。

### 运行与产物

解压到服务器 `unimc_model_optimization`，运行：

```bash
sbatch slurm/run_centered_heads.sbatch
```

1 GPU，4 CPU，16G，最长 1 小时。日志为项目根目录 `unimc_center_JOBID.out/.err`。中心化缓存写入 `cache/centered_mlm_JOBID`，结果写入 `runs/centered_mlm_JOBID`，包含检查点、训练曲线、每题验证预测、comparison.csv/json。

完成标记：`CENTERED HEAD EXPERIMENT COMPLETE`。所有结果与 baseline 同时报告。

新的缓存签名记录中心化和父特征签名，检查点可继续通过已有 `predict_head.py` 使用；推理必须提供同样中心化后的缓存。把新检查点直接用于未中心化缓存会被签名检查拒绝。后续校准/目标缓存也应先经过同一中心化脚本；改变实际候选集仍需重新运行原始编码器，不能只删缓存行来代表新输入。

检查已覆盖均值为零、共同平移不变性、候选换序、标签无关、原始 logits/指标保持不变、通用训练/导出接口及错误输入签名拒绝。未在本机训练真实全量中心化模型。
