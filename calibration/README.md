# UniMC 最终版置信度校准

这部分对应小组分工 D，使用最终 v0.3 交付包中的原始 UniMC 和四种新增决策层。四种决策层是 `mlp`、`tf_noint`、`tf_int`、`tf_gate`，各有 seed 0、1、2。正式结果在 `results/report.md`，包含可供英文报告整合的方法、设置、结果表和跨任务分析。

每个模型和种子在四个源任务的校准题上拟合一个全局温度，然后把这个温度应用到所有测试任务。源域校准题共 3,193 道，来自 ARC-C 112 道、CommonsenseQA 974 道、OpenBookQA 496 道、PIQA 1,611 道。拟合目标是按题目平均的 NLL，每道校准题权重相同。原模型的分数在十二套输出中逐题核对一致，最终只拟合和统计一次。

本次固定使用 15 个等宽分箱计算 ECE，置信度是预测选项的最大 softmax 概率。NLL 使用自然对数，Brier 使用各有效选项平方误差之和的样本平均，范围为 [0, 2]。目标任务的总表先对 SST-2、CB、COPA、HellaSwag 四个任务等权平均，再计算三个训练种子的均值和样本标准差。可靠性图展示 seed 0，完整表保留全部种子。

本次由执行者确定的设置是 15 个 ECE 分箱、按校准题目等权拟合、按目标任务等权汇总，以及种子标准差 `ddof=1`。这些设置在读取目标任务结果前固定。

| 文件 | 用途 |
|---|---|
| `temperature_scaling.py` | 温度拟合、概率转换、ECE/NLL/Brier 计算 |
| `run_calibration.py` | 检查交接数据、运行全量校准、生成表格和报告 |
| `plot_results.py` | 生成五张可靠性图和一张目标指标对比图 |
| `input_manifest.json` | 最终包中 168 份预测的 SHA-256、12 份权重的 SHA-256、版本信息 |
| `results/temperatures.json` | 原模型一个温度、四种决策层各三个温度，以及对应版本身份 |
| `results/target_macro_summary.csv` | 四个目标任务的宏平均结果，包含种子标准差 |
| `results/metrics.csv` | 每个模型、种子、数据集、划分在校准前后的完整结果 |
| `results/report.md` | 英文方法和结果分析，可交给报告整合同学 |
| `results/reliability_bins.csv` | 所有置信度箱的数量、平均置信度和实际准确率 |
| `results/audit.json`、`results/verification.json` | 输入一致性、独立数值复核、冻结温度重放检查 |

本次实际输入已经放在本地 `unimc_decision_layers/runs/<模型>/seed<种子>/preds/`。它们来自小组原有 Drive 交付包，逐题文件有 `id`、`label`、`raw_logits`、`new_logits`、`pred_raw`、`pred_new`。读取时只保留真实选项，合并不同选项数的数据时以负无穷 padding，padding 概率为零。

仓库的忽略规则会排除逐题 JSONL。其他组员复现时，把同一最终包 `unimc_opt/runs/` 中四种主模型、三个种子的 `preds/` 文件放到上面的对应位置，保留仓库中的 `unimc_decision_layers/data/splits.json`。脚本会核对文件 SHA-256、校准题号和模型间对齐，结果可追溯到原始交付文件。[下载最终版预测交付包](https://drive.google.com/file/d/1znKY0iB9I0rVcg2hetnm6m12dWa0z_BT/view)

将下载的 `unimc_opt_results.zip` 放到仓库根目录，可以运行下面的命令恢复本次所用的 168 份预测。

```bash
python - <<'PY'
import json
from pathlib import Path
from zipfile import ZipFile

manifest = json.loads(Path("calibration/input_manifest.json").read_text())
with ZipFile("unimc_opt_results.zip") as archive:
    for entry in manifest["members"]:
        if entry["purpose"] == "prediction":
            destination = Path(entry["local_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(entry["archive_member"]))
PY
```

从小组仓库根目录运行以下命令。

```bash
python -m pip install -r calibration/requirements.txt
python -m unittest discover -s calibration/tests -v
python -m calibration.run_calibration
```

默认结果写入 `calibration/results/`，也可以用 `--output` 指定其他目录。`audit.json` 是本轮程序检查完成的记录。随交付保存的 `verification.json` 对应发布时的独立数值复核和重放检查，每次重新生成结果时会移除旧核验记录，避免混用不同运行的证据。

使用已经拟合好的温度重放评估时，运行以下命令。

```bash
python -m calibration.run_calibration \
  --temperatures-file calibration/results/temperatures.json \
  --output /tmp/unimc-calibration-replay
```

给演示程序接温度时，按模型和种子选取参数。例如，`tf_int/seed0` 对应有交互 Transformer 的 seed 0。

```python
import json
import numpy as np
from calibration.temperature_scaling import probabilities

with open("calibration/results/temperatures.json", encoding="utf-8") as handle:
    fitted = json.load(handle)
temperature = fitted["fits"]["tf_int/seed0"]["temperature"]

# final_logits 是这一模型在单道题上输出的各有效选项分数。
calibrated = probabilities(np.asarray([final_logits]), temperature)[0]
```

温度绑定对应的模型权重和输入版本。源校准题留出于小组新增决策层的训练和选模，官方 UniMC 上游训练包含这些源任务。目标任务标签仅用于计算最终指标。CB 只有 56 题、COPA 只有 100 题，图中标明样本量，分箱数量明细保存在 CSV 中。

本次结果显示，四种决策层在四目标任务上的平均 ECE、NLL、Brier 都有一定下降。原始 UniMC 的源域最优温度为 0.917180，应用到目标任务后，平均 ECE、NLL、Brier 略升。各任务的迁移效果不同，报告保留了改善和退化的全部结果。

温度缩放和 ECE 的方法参考 Guo 等人的 ICML 2017 论文。[On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)
