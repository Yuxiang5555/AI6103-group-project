> Historical experiment/handoff notes. Package descriptions and run IDs refer to archived artifacts; start with the repository README for the current source-tree layout. Personal machine paths have been replaced.

# 三种子确认：恢复预测导出与统计

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
