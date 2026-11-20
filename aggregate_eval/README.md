# 汇总实验组 · 独立目标评估

已准备 11,070 道带标签的公开验证题，构造 42,336 条原题及扰动输入。逐条检查通过，覆盖原题 ID、选项回映、正确标签、删除条件和数据哈希。本实验所需的 12 个评分头已发布在 `unimc_model_optimization/models/heads/`。这套评分头的目标准确率、扰动后的性能及拒识结果仍待实测。四源 v0.3 决策层（种子 0/1/2）已有[独立评估和校准结果](../calibration/results/report.md)，其温度只适用于对应模型。

| 公开验证集 | 原题 | 轮换顺序 | 删除一个干扰项 | 删除标注正确项 |
|---|---:|---:|---:|---:|
| SST-2 | 872 | 872 | — | — |
| CB | 56 | 56 | 56 | 56 |
| COPA | 100 | 100 | — | — |
| HellaSwag | 10,042 | 10,042 | 10,042 | 10,042 |
| 总计 | 11,070 | 11,070 | 10,098 | 10,098 |

## 给代跑推理的同学

从 PR 分支运行时，先在仓库根目录用装有 PyArrow 的 Python 准备公开数据。若计算节点无法联网，可在另一台机器运行此命令，再将生成的 `data/aggregate_targets/` 复制到集群仓库的相同位置。

```bash
python aggregate_eval/prepare_targets.py --output data/aggregate_targets
```

如果收到的是含数据的 `UniMC_target_eval.zip`，解压后直接进入 `UniMC_target_eval` 目录即可。随后在分配到的 GPU 节点上激活训练时的 UniMC 环境，从项目根目录运行以下命令。两个参数依次是含 `config.json` 的原版基座目录和含 `fengshen/` 的源码目录。默认从仓库读取评分头。旧交接目录也可作为额外的第一个参数。把示意路径替换成服务器上的实际路径。

```bash
python aggregate_eval/run_and_pack.py /path/to/unimc-albert-235m /path/to/Fengshenbang-LM
```

脚本使用已有权重推理。成功后请回传 `output/UniMC_target_results.tar.gz`。包中含 52 份逐题预测、4 份任务完成记录、数据清单和汇总报告。若作业中断，在同一目录执行原命令会复用已完成且校验通过的任务。若报错，请保留作业日志以便定位。

## 固定协议

- 使用项目既有的分类与选择题模板、公开 validation 标签、仓库固定的 UniMC 基座及 FP32 编码。四个任务在评分头训练及选模后评估。每个任务权重相同。
- 比较原版 UniMC 和四种评分头。主候选为 centered full，对照为 uncentered full、centered self、centered MLP。每个评分头使用优化组已选定的 42、43、44 三个种子，一共 13 组预测。Gate、LoRA 和四源任务重训属于其他实验。
- 对每题用固定哈希选择一次非零循环移位，按选项身份比较答案。三、四选一题各构造一对删项输入，分别删除一个确定的干扰项和数据集标注的正确项。删项保留其余选项顺序。二选一只做轮换，以满足模型至少接收两个选项的要求。HellaSwag 使用固定版本内的行号为题目编号，另保留可重复的上游编号。
- 所有变体从完整输入重新运行官方编码器，再投影和中心化。删正确项样本的标签为 null。“无答案”依赖原数据集每题唯一正确选项的标注假设。公开题目的自然语言歧义尚未逐题人工复标。
- 逐任务报告原题和保留正确项的扰动题准确率、NLL、15 个等宽置信度箱的 ECE、多类 Brier 分数，以及预测是否仍指向原来的选项。原题先按任务等权平均，再报告三个种子的均值及样本标准差。改变答案的题目同时保留纠正、破坏和错到错的变化，附原版前两名分差及选项数。
- 主候选相对每个对照的准确率差采用任务内逐题配对 bootstrap，20,000 次，随机种子 20261120。同一道题的所有训练种子一起重采样。区间针对这些固定的模型和四个任务。
- 在 CB 和 HellaSwag 上，删正确项与删干扰项一一配对，正负样本各半，两组剩余选项数相同。以缺标注正确项为正类计算 AUROC、平均精确率（AP）。原始分数采用 1 减最大 softmax 概率作为拒识分数。校准组交付模型后用其冻结的概率和拒识分数同场评估，阈值和超参数只用源校准数据确定。

## 数据与推理

从 Git 仓库根目录运行。准备数据需要 Python 和 PyArrow，汇总统计需要 NumPy。已生成的输入在本机的 `data/aggregate_targets/`。数据不随 Git 提交，文件版本、行数、排除记录和 SHA-256 保存在 [target_manifest.json](target_manifest.json)。重新准备会校验全部哈希。

```bash
python aggregate_eval/prepare_targets.py --output data/aggregate_targets
python -m unittest discover -s aggregate_eval -v
```

在有官方 UniMC 环境及 CUDA 的计算节点运行以下命令。BASE 和 FENGSHEN 分别指向历史实验所用的基座与编码器。每个任务分别检查权重哈希、原版预测一致性及缓存签名，成功后才发布该任务的逐题输出。重复执行会验证并复用已完成任务。

```bash
python aggregate_eval/run_targets.py --inputs data/aggregate_targets \
  --model "$BASE" --fengshen-root "$FENGSHEN" \
  --out predictions/aggregate_targets
python aggregate_eval/summarize.py --inputs data/aggregate_targets \
  --predictions predictions/aggregate_targets --output output/aggregate_targets
```

如果队友已完成推理，可直接汇总其逐题文件。目录结构为 `predictions/aggregate_targets/<任务>/<模型>.jsonl`，原版用 baseline，其他模型名形如 centered_full_seed42。输入须覆盖同一版本的全部原题和变体，遵循项目 predict_head.py 的输出格式。仅有每个任务的平均准确率无法恢复逐题配对统计。

`report.md` 是主结果表，`metrics.json` 包含所有逐任务、逐种子、逐条件指标，`changed.csv` 给出逐题变化。推理任务记录中的耗时覆盖共享编码、投影及全部模型推理，单模型在线延迟需另外计时。

## 接入校准组结果

给 summarize.py 增加 `--calibrated <目录>` 并使用新的输出路径。目录沿用相同任务和模型名。每行应包含 id、dataset、与预测对齐的 choices、按选项排列且总和为 1 的 probabilities，以及越大越倾向拒答的 rejection_score。校准组需随结果给出所用模型、源校准划分和拟合记录。此路径用于评估已冻结的校准方法。
