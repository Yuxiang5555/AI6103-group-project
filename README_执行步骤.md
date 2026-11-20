> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

# UniMC 模型优化：执行指南与起始代码

本目录实现你们已确定的模型优化路线：冻结英文 UniMC，缓存选项表示，训练一个轻量残差决策层。主要研究问题是：相同决策层允许选项之间交互后，是否能比禁止交互的版本取得更好的跨任务分类结果。

本目录不包含置信度校准模块。所有预测导出原始 logits，校准组可以独立接入。

## 0. 代码已经验证到哪里

- 在本机 Python 3.10、PyTorch 2.14.1 CPU 环境通过 13 项单元测试，包括实际 4096 维输入的前向和反向计算。
- 用人工构造的特征跑通三个版本的训练、源验证集选模型、保存/加载检查点、目标 logits 导出及原版分数导出。
- 尚未在本机下载真实 UniMC 权重、运行 Fengshen 特征提取或测试 A40。`extract_features.py` 是基于官方类接口编写的适配器，需要在你们已经跑通官方 UniMC 的环境中确认。
- 人工特征的检查结果只证明工程流程可运行，不能当作 UniMC 性能提升结果。
- 数据模板为固定的项目模板，不声称完全复现论文逐任务最佳模板。首批官方接口一致性检查，验证的是相同输入模板下两种读取方法一致。

## 1. 先明确模型优化组的边界

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

## 2. 文件分工

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

## 3. 环境：先验证小模块，再接真实模型

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

## 4. 先做 PIQA + CommonsenseQA 小实验

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

## 5. 先提取 64 条，验证官方输出一致

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

## 6. 提取固定缓存

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

## 7. 新模块结构

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

## 8. 训练三组，先固定其它设置

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

## 9. 小实验如何判断下一步

先看：

1. 有限损失和梯度，训练程序能保存并加载检查点。
2. MLP 与原版：重新评分是否有帮助。
3. full 与 self：参数结构相同的情况下，选项交互是否有额外作用。
4. PIQA 与 CQA 是否出现不同趋势，是否一个任务主导整体结果。

第一次只比较学习率 1e-4 / 3e-4，其他参数固定。源验证选择好后，把主要版本用种子 13、42、2026 复跑。正式种子可以由全组另行统一，但所有模型必须相同。

检查尺度：新 delta 是否快速大到覆盖原分数，是否训练准确率上升而源验证持续恶化。先用早停和源验证比较控制；不据目标结果逐步挑选结构。

可以在 full 基础上加 `--gate`：`final_logit = base_logit + sigmoid(gate_logit) * delta`。这是修正幅度门控，不等同于只控制跨选项注意力。Gate 属于额外实验；主比较仍是 mlp/self/full。

## 10. 扩展四个源任务

ARC-Challenge 使用 `--task arc`，OBQA 使用 `--task openbookqa`，支持官方 question/stem/choices 对象和常见 HF 导出字段。

按相同规则生成 train/val/cal 后提取缓存；训练命令的 `--train`、`--val` 后面追加两个任务的对应文件。必须保证：

- 同一官方 checkpoint、同一编码设置、同一缓存精度。
- 相同训练数据、相同训练/验证划分、相同抽样、相同超参数搜索预算。
- self/full 不改维度或额外加层来制造差异。

正式目标评估与大实验由汇总组运行。目标结果不回流到你们的训练和模型选择。

## 11. 给汇总组、校准组导出 logits

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

## 12. 模型优化组最终交付

- 本目录源码，以及实际运行环境的版本记录。
- 固定数据 manifest、缓存 metadata 和提取精度。
- 原版、MLP、self、full 的小实验结果；Gate 若完成则附上。
- 每个模型的 config.json、best.pt、history.json、summary.json。
- 一条可以复现训练的命令，一条可以导出 logits 的命令。
- 方法说明：冻结骨干、残差评分、选项集合交互、对照掩码和选模型规则。

缓存上的耗时不是完整部署耗时。大实验组测延迟/显存时，需要把真实骨干前向加上新增决策层，并固定 batch、输入长度与数值精度。

## 13. 故障定位

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
