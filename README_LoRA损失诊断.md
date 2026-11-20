> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

# UniMC LoRA：首轮损失与梯度诊断

## 当前结论

seed42、作业55840的最佳已训练检查点：源验证Macro Accuracy为76.1112%，原版76.4164%，下降0.3052个百分点。检查包与训练配置匹配；两步检查中基座参数哈希保持不变，A/B参数正常获得梯度并更新，512长度测试峰值已分配显存1.5545 GiB。这证实了两步技术检查通过，尚未解释准确率下降。

本补丁只新增诊断脚本、测试、Slurm脚本和本文档。保留原LoRA代码、签名、检查点和结果，当前LoRA也不替换先前的centered full候选。

## 诊断内容

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

## 上传并提交

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

## 查看结果

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

## 修复记录

首版服务器任务在MLM核对处停止：重算1.123233676，官方返回1.123046875，差约0.0001868（此前说明曾误写为0.00187）。首版本地三个测试只覆盖FP32，未发现该比较风险。仅凭截图无法确认服务器差异的全部来源，修复版保留dtype/数值证据，并把正式诊断统一为FP32；没有放宽正式核对容差。原训练实现和检查点不修改，失败任务没有优化器或参数更新。

2026-11-18返回结果已复核：96条FP32正式诊断中官方loss与完整FP32重算差值均为0，加权拆分最大误差约2.38e-7。三个adapter及配置哈希与作业55840一致，32题抽样ID已在本地重现。完整审计见工作区output下`UniMC_lora_diagnostic_analysis_55840/LoRA损失诊断_复核结论.md`。本地未重跑完整编码器。
