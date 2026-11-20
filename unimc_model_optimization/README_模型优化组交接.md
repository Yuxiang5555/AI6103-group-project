> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

# UniMC 模型优化组交接

本包固定一个待独立评估的候选：**冻结 UniMC + 官方 MLM 中间表示 + 题内选项中心化 + full 残差评分头**。保留三个随机种子及全部对照。源开发集上 full 的三种子 Macro Accuracy 为 **76.5336% ± 0.1349 个百分点**，原版为 76.4164%，均值差 +0.1172 个百分点。探索性的逐题配对 bootstrap 区间为 [−0.2734, +0.4994] 个百分点，包含零。当前证据支持将它交给汇总组检验，尚不能宣称准确率显著提升。

## 1. 包内文件与职责

- `models/`：中心化/未中心化 × mlp/self/full × seed42/43/44，共 18 个 `best.pt` 的原样副本。主候选命名为 `centered_full_seed42.pt` 等。
- `provenance/`：每个检查点的配置、训练历史、最佳轮次与源验证结果。
- `data/pilot_seed42_v2/`：本次残差头的源 train/val/cal 及划分说明，供复现和校准使用。
- `calibration/`：baseline 和主候选三个种子对同一批 source_cal 样本的**未校准** logits、元信息与自动生成的指标。模型选择已在导出前固定；这里的指标不能用于反选模型或调头部参数。
- `source_results/`：三种子汇总、逐题配对分析、全部验证预测、变化样本与图表。区分种子标准差和逐题重采样区间。
- `model_manifest.json`：基座 revision、特征签名、18 个检查点的 SHA256、运行环境、导出审计。
- 根目录 Python 脚本：提特征、变换、训练及预测。大体积基座权重、Fengshen 源码与特征缓存未包含在交接包内。

模型优化组负责交付这套候选、对照和源任务小实验。校准组在固定模型上处理 source_cal，并训练自己的校准/拒识模块。汇总实验组统一完成独立任务和候选变化实验，报告组依据统一结果撰写方法、结果与分析。不要继续围绕这批开发集的零点几分收益做大范围搜索。

## 2. 方法及固定参数

每道题有 K≥2 个实际选项。使用官方 UniMC 编码及 ALBERT，保留每个选项标记处的原版 yes logit `b_i` 和 4096 维隐藏向量。骨干和官方 MLM 层始终冻结。

1. `project_mlm_features.py` 使用官方 MLM 的 dense → activation → LayerNorm，将缓存向量转为 decoder 前的 128 维表示 `u_i`；保留原版 `b_i`，不重新用近似向量生成基线分数。
2. 中心化：`v_i = u_i − (1/K) Σ_j u_j`，只对本题实际选项求均值，结果保存为 FP32，不使用标签。
3. 残差头：输入 LayerNorm → 投影到 64 维 → 注意力/前馈模块 → 标量残差 `δ_i`，最终 `z_i = b_i + δ_i`。
4. 初始标量输出层为零，初始模型严格复现 baseline。full 使用跨选项注意力；self 只允许每个选项注意自身；mlp 使用逐选项前馈。所有模式均无选项位置嵌入，支持变长选项和 padding。

固定设置：model_dim=64，heads=4，FFN=256，dropout=0.1，gate=false；AdamW，lr=1e-4，weight_decay=0.01，batch_size=128，最多 10 轮、patience=2、梯度裁剪 1.0；按源任务均衡采样。seed=42/43/44，每次按源验证 Macro Accuracy 选最佳轮次，同分按更低 Macro NLL。full/self 头各 58,689 个可训练参数，mlp 为 46,209 个。

中心化本身已使用候选集合信息。因此 centered self/MLP 也不是完全独立选项模型，self 只消融跨选项注意力。注意力的额外收益应比较 centered full 与 centered self/MLP。

## 3. 数据与已有结果的适用范围

| 源任务 | source_train | source_val | source_cal |
|---|---:|---:|---:|
| PIQA | 2,000 | 919 | 919 |
| CommonsenseQA | 2,000 | 598 | 599 |

划分 seed=42，校准集与模型训练/选轮次分开。已按标准化内容检查重复和跨划分重合，PIQA 官方 train/dev 重复内容从训练池移除，细节见数据 provenance。Macro Accuracy 为两个任务准确率的算术平均，不是合并 1,517 题后的总体准确率。

原论文附录 A.1 将 PIQA、CommonsenseQA 列为 MC 训练任务，附录 A.2 将 SST-2、CB、COPA、HellaSwag 列为评估任务。故本包结果是**新增残差头的源开发集结果**，不能写成目标任务零样本提升。本次 train/val/cal 隔离只说明新增模块的数据划分，不证明发布基座未见过其中内容，基座具体训练划分暴露仍未核实。[UniMC 原论文](https://aclanthology.org/2022.emnlp-main.474.pdf)

三个 centered full 检查点分别净多答对 5、1、1 题，PIQA 收益多于 CQA。它们相对 baseline 的 Macro 差值为 +0.2720、+0.0544、+0.0252 个百分点。所有种子都纠正的题目仅 3 个，同时都破坏的题目 2 个。详细表格见 `source_results/source_results.md`。

统计说明：± 为三个训练种子的样本标准差。bootstrap 按任务分层、按题目配对重采样 20,000 次，同一道题的三个种子预测整体携带，模型固定。它不覆盖训练样本重抽或整个选模过程；这批源开发集被用于多轮探索，因此区间只作探索性分析，不是独立测试的确认性证据。full 相对 centered MLP/self 的探索性区间也包含零。

在最终交接包根目录可复核逐题结果与重采样：

```bash
python source_results/analyze_centering_confirmation.py \
  --predictions source_results/validation_predictions \
  --comparison source_results/three_seed_confirmation.json \
  --source-data data/pilot_seed42_v2 --output analysis_reproduced
```

## 4. 接口给校准与拒识组

每行 JSONL 包含 `id`、`dataset`、`gold_label`、`choices`、`num_options`、`logits`、`base_logits`、`prediction`、`probabilities`。标签和预测均为 **0 起始选项索引**；`logits` 是要交给校准器的模型原始分数，`base_logits` 是原版 UniMC 分数，`probabilities` 只是原始 softmax，尚未校准。按 `(dataset, id)` 联结，比较前核对候选顺序，不按文件行号猜测对应关系。

`calibration/baseline.jsonl` 与 `centered_full_seed42/43/44.jsonl` 各有 1,518 道完全相同的 source_cal 样本。先固定各模型和种子，再在 source_cal 上拟合温度或拒识模块；若还要选择拒识超参数，应在该池内预先划分拟合/验证或使用交叉验证。报告三个种子时，各自拟合、各自评估，不挑选表现最好的种子。目标任务标签只供最后评估，不能用于阈值、温度或模型选择。

无正确选项的数据使用 `gold_label: null`，在独立标注文件提供 `is_answerable`、扰动类型和原始题 ID。当前分类器仍返回一个 argmax，是否拒识由另一个模块决定；不能把 `null` 默认为索引 0。当前特征接口要求至少 2 个候选。

候选增加、删除、改写或打乱时，应从该版本的完整原始输入重新执行编码与中心化。官方编码的隐藏向量依赖整组候选，不能直接删缓存行或仅对旧向量重排来声称测到了端到端鲁棒性。干扰项或缺答案样本要核对语义，避免意外引入第二个正确答案。

## 5. 独立任务推理：四步

以下 Bash 命令在解压后的交接包根目录运行，并放入已分配 GPU 的 Slurm 作业内。`BASE` 指向原始本地基座模型目录，`FENGSHEN` 指向当前已验证的 Fengshen checkout。输入 `data/task.jsonl` 使用既有 canonical schema（包含 role=target_eval、id、dataset、texta、textb、question、choices、gold_label、content_hash）。分类任务可先用 `prepare_data.py` 转换；模板在评估前统一固定。

```bash
export MKL_THREADING_LAYER=GNU
BASE=/path/to/unimc_project/models/unimc-albert-235m
FENGSHEN=/path/to/unimc_project/Fengshenbang-LM-main

python extract_features.py --input data/task.jsonl --output cache/task_raw.pt \
  --model "$BASE" --fengshen-root "$FENGSHEN" --batch-size 2 \
  --max-length 512 --amp none --verify-official --allow-option-truncation --device cuda:0

python project_mlm_features.py --cache cache/task_raw.pt --output-dir cache/task_mlm \
  --model "$BASE" --fengshen-root "$FENGSHEN" --device cuda:0

python center_option_features.py --cache cache/task_mlm/task_raw.pt --output-dir cache/task_center

python predict_head.py --cache cache/task_center/task_raw.pt \
  --checkpoint models/centered_full_seed42.pt --output predictions/task_seed42.jsonl --device cuda:0
```

分别换 seed43/44 得到其余预测。baseline 使用第一步 raw 缓存配合 `--baseline`；未中心化对照使用第二步 MLM 缓存；中心化 MLP/self 使用第三步缓存及对应检查点。所有新输出路径必须不存在。

`model_manifest.json` 记录完整签名；转换脚本内容也属于签名。不要改写换行、另存投影/中心化文件或绕过签名检查。基座固定为 `IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English`，revision `e3aeebc7bd475d559bd0cbd8d17abf4e159d1260`。准确复用官方自定义模型、分词和特殊注意力编码，不用普通 AutoModel 编码代替。已有运行用 Python 3.10.22、torch 2.5.1/CUDA 12.4、transformers 4.20.1、pytorch-lightning 1.9.5、torchmetrics 0.11.4、sentencepiece 0.1.99、protobuf 3.20.3、huggingface-hub 0.25.2、jieba 0.42.1、numpy<2、setuptools<81；完整版本以导出时 manifest 为准。

## 6. 汇总组应固定的实验

先冻结本包配置及三个种子，统一评估 SST-2、CB、COPA、HellaSwag 等预定独立任务。主比较为 baseline 与 centered full；机制比较保留 uncentered full、centered MLP、centered self。对每个模型都用同样的模板、数据版本、候选扰动、校准流程和选模规则。

建议报告：各任务 Accuracy、Macro Accuracy、NLL；校准后的 ECE/Brier；拒识 AUROC/AUPRC、风险–覆盖率或 AURC；候选顺序/数量改变与正确答案移除时的表现。将正常题、无答案题的构造及采样比例写清楚。分类准确率与拒识质量分别给结果，最终比较覆盖率或拒识阈值也应保持公平。

分析重点是哪些题被纠正/破坏、低 margin 题和不同选项数的差异，以及中心化与注意力各贡献多少。性能计时需包含基座编码，当前缓存头部预测时间不能冒充端到端速度。若按原项目计划扩充到四个源任务，应作为独立且预先固定的训练设置重训全部必要对照，不与本两源任务结果混用。

## 7. 在当前服务器生成完整包

模型优化脚本 `build_model_handoff.py` 读取已完成的实验 55757，复用全部 18 个检查点，核对源数据与特征签名，只投影并导出已有 source_cal 缓存。它不启动优化器、不拟合校准器、不读取目标任务。

在项目代码根目录运行 `sbatch slurm/build_model_handoff.sbatch`。日志位于项目父目录的 `model_handoff_JOBID.out/.err`，成功标记为 `MODEL HANDOFF COMPLETE`。产出 `handoff/model_optimization_JOBID/` 及项目父目录的 `UniMC_model_optimization_handoff.zip`。包内不含临时缓存。第一次构建在服务器运行后才能确认真实校准分数的导出成功。
