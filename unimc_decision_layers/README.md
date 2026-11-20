# UniMC 模型优化：决策层代码包（交接文档）

负责人：Sun Peihan　　版本：v0.3（2026-11-18，划分改为 train/val/calib 三切 + 官方 validation 作域内测试）

对应讨论稿里的"改进方向 1：让选项在最后一步互相比较"。
整条流水线：**官方 UniMC（冻结）→ 缓存每个选项的 [O-MASK] 表示和原始分数 → 训练四个新决策层 → 导出逐样本 logits 和结果表**。

## 1. 文件

| 文件 | 作用 |
|---|---|
| `unimc_core.py` | 官方 UniMC 的推理路径（预处理逻辑逐行照搬官方 `UniMCDataset.encode`），输出每个选项的 4096 维表示 + 原始 "yes" logit |
| `prepare_data.py` | 下载 8 个数据集，统一成 UniMC 输入格式，固定划分 |
| `extract_features.py` | 冻结 UniMC 跑一遍，写特征缓存 `cache/`，同时给出官方基线准确率 |
| `decision_layers.py` | 四个新决策层 + 推理接口 `DecisionScorer` |
| `train_decision.py` | 训练、按源域验证集选模、多种子、导出逐样本 logits |
| `evaluate.py` | 汇总成 Accuracy / Macro-F1 / 参数量 / 延迟 / 显存 表 |
| `pipeline.py` | 端到端：输入文本 → 原模型答案 + 改进模型答案（给 Gradio 用） |
| `smoke_test.py` | 不需要 GPU 和权重的全流程自测（假数据） |
| `run_all.sh` | A40 上从头到尾的命令 |

## 2. 最小运行命令

```bash
pip install -r requirements.txt
python smoke_test.py                                   # 1 分钟，CPU 即可
python prepare_data.py --out data                      # 下数据
python extract_features.py --data data --out cache --dtype bf16 --batch 32   # 提特征（A40 约 0.5–1 h）
python train_decision.py --kind all --seeds 0 1 2 --cache cache --out runs   # 4 模型 × 3 种子
python evaluate.py --runs runs --cache cache --out results
```

## 3. 数据与划分

| 数据集 | 用途 | 划分 |
|---|---|---|
| ARC-Challenge, CommonsenseQA, OpenBookQA, PIQA | **源任务**：训练新决策层 | 官方 train 切三份：80% → `train`（训练）、10% → `val`（选模）、10% → `calib`（留给置信度校准，决策层训练不碰）；官方 validation 原封不动 → `test`（域内测试，训练和选模都没碰过）；ARC-C、OBQA 带标签的官方 test 另存为 `test_official` |
| SST-2, CB, COPA, HellaSwag | **目标任务**：只评估 | 官方带标签的评估划分 → `test`；标签不参与训练、调参、拟合温度 |

划分用 seed 0 固定，`data/splits.json` 记录了哪些样本进了 val / calib（按讨论稿『训练 / 验证 / 校准三部分不重叠』）。
提示模板（question / choice 的措辞）写在 `prepare_data.py` 的 `TEMPLATES` 里，按论文 Table 9 / Figure 3 设置（CB："Base on the paragraph." + "We can infer that … / We can not infer that … / It is difficult for us to infer …"；SST-2："What is sentiment of reviews?" + "It's great. / It's terrible."；HellaSwag 不加 question），COPA 论文没给，用的是 cause / effect 两句问话。所有模型共用同一套。

注意：论文 Table 8 列出的 14 个 MC 训练集里**包含 ARC、CommonsenseQA、OpenBookQA、PIQA**，也就是说官方权重已经见过我们的 4 个源任务；SST-2、CB、COPA、HellaSwag 则是论文的零样本评估集。我们的划分和论文的"MC 训练 → 零样本"协议是一致的，但源任务上的基线会偏高、新决策层在源任务 val 上的提升空间可能不大，重点还是看跨任务结果。

## 4. 统一交接格式（缓存文件）

`cache/<数据集>/<划分>.pt`，`torch.load` 后是一个 dict：

| 键 | 形状 / 类型 | 说明 |
|---|---|---|
| `feats` | float16 `(N, M, 4096)` | 第 i 个样本第 j 个选项的 [O-MASK] 表示（ALBERT 最后一层 hidden state）；padding 位置为 0 |
| `raw` | float32 `(N, M)` | 官方原始分数 = 该位置词表里 "yes" 的 logit；padding 位置为 −1e4 |
| `mask` | bool `(N, M)` | True = 真实选项，False = padding |
| `label` | int64 `(N,)` | 正确选项下标 |
| `n_opt` | int64 `(N,)` | 每个样本的真实选项数 |
| `ids` | list[str] | 样本 ID，如 `csqa-train-17`，对应 `data/csqa/train.jsonl` 第 17 行（文本、问题、选项原文都在那里） |
| `meta` | dict | 模型名、精度、max_length、版本号 |

M = 该数据集最大选项数（PIQA/SST-2/COPA 2，CB 3，ARC/OBQA/HellaSwag 4，CSQA 5）。
官方预测 = 对 `raw` 在 `mask` 为 True 的位置做 softmax 后取 argmax；`cache/<数据集>/<划分>.baseline.jsonl` 里已逐样本写好。

## 5. 四个决策层（`decision_layers.py`）

输入 `feats, raw, mask`，输出 `(N, M)` 的新 logits（padding 为 −inf）。四个模型共用同一个输入投影（LayerNorm → 4096→256）和同一个打分头（256→1），**只有中间一块不同**：

| `--kind` | 中间块 | 回答的问题 |
|---|---|---|
| `mlp` | 每个选项各自过一个残差 MLP，选项之间不交互 | 单纯多一个评分层有没有用 |
| `tf_noint` | 1 层选项 Transformer，attention mask 是单位阵：每个选项只能看自己；**参数量和 `tf_int` 完全相同** | 单纯加深度有没有用 |
| `tf_int` | 同一层，但选项可以互相 attend | 选项互相比较本身有没有用 |
| `tf_gate` | `tf_int` + 每个选项一个门 g∈(0,1)，`h ← h + g·Attn(h)` | 控制交互强度是否更好 |

新分数 = **原始分数 + 修正分数**（`--no_residual` 可关掉做消融）。打分头最后一层初始化为 0，所以训练第 0 步四个模型的输出都恰好等于官方 UniMC，训练只会在数据支持时才偏离基线。原始分数也作为特征喂回去（样本内标准化后），`--no_raw_feat` 可关掉。
padding 选项不参加 attention（key padding mask）、不参加损失（logit −inf）、不参加归一化。

训练：交叉熵，AdamW（lr 2e-4，wd 0.01），cosine 到 0，20 epoch，batch 256，按**源域 val 准确率**选 epoch。参数量约 1.2M（d=256）。

## 6. 训练输出（给大实验 / 校准用）

```
runs/<kind>/seed<k>/
  best.pt            权重 + 配置（DecisionScorer.load 直接读）
  log.csv            每个 epoch 的 lr / train_loss / train_acc / val_loss / val_acc
  results.json       该模型在所有数据集所有划分上的 acc_raw / acc_new
  preds/<数据集>_<划分>.jsonl   逐样本：id, label, raw_logits, new_logits, pred_raw, pred_new
runs/train_table.json
results/table_acc.md  table_f1.md  cost.md  per_run.csv    （evaluate.py 生成）
```

校准（Temperature Scaling）请用 `preds/*_calib.jsonl` 里的 `raw_logits` / `new_logits` 拟合温度，再用在 `*_test.jsonl`（和 `*_test_official.jsonl`）上。

## 7. 推理接口

```python
from decision_layers import DecisionScorer
sc = DecisionScorer.load("runs/tf_int/seed0/best.pt", device="cuda")
logits = sc.score(feats, raw, mask)     # (N, M) 或单样本 (M,)
probs  = logits.softmax(-1)

# 从文本开始（demo）：
from pipeline import UniMCPlus
p = UniMCPlus("IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English", "runs/tf_int/seed0/best.pt")
p.predict([{"texta": "it 's just incredibly dull .", "textb": "",
            "question": "What is sentiment of follow review?", "choice": ["it's terrible", "it's great"]}])
```

## 8. 已做的检查 / 未解决的问题

已做：
- 预处理与官方代码逐行一致（`[MASK]` 放在每个选项前；选项之间 attention 块对角、互不可见；每个选项位置编码从 question_len 重新数；"yes" 的 token id 用 `tokenizer.encode('yes')[1]`）。
- `smoke_test.py` 在假数据上跑通四个模型、两个种子、导出、汇总、接口。

未解决（需要在 A40 上确认）：
- 【待验证】`extract_features.py` 的基线准确率应接近论文零样本数字（Table 3 / Table 12：CB 75.7、COPA 95.2、HellaSwag 62.5、SST-2 91.6）；如果差很多，先怀疑提示模板，再怀疑 attention mask 的实现。`run_all.sh` 里先跑 32 个样本的 fp32 小测试。
- 【待验证】bf16 与 fp32 的预测一致率（预计 >99%）。
- 官方代码在"选项很长 + 文本接近 512"时 position id 可能超过 511 而报错，这里改成截断到 511（`unimc_core.py` 标了【补充】）。
- PIQA 在 HuggingFace 上是脚本型数据集，`prepare_data.py` 里做了三种加载方式的回退；如果都失败，手动下载 jsonl 放到 `data/piqa/`。
