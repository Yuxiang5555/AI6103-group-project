# Operations and Reproducibility Guide

Git commands run from the repository root. Python, dependency, test, and Slurm
commands run from `unimc_model_optimization/`, unless stated otherwise.

```bash
cd unimc_model_optimization
```

The default sibling paths now refer to directories inside the Git repository:

```text
AI6103-group-project/
├── unimc_model_optimization/  # component working directory
├── Fengshenbang-LM-main/     # external checkout, ignored
└── models/                  # external backbone, ignored
```

## Reproduction

### Environment and external dependencies

Use a separate Python 3.10 environment. The recorded GPU stack is described in `results/base_model_signature.json`. Install the appropriate PyTorch 2.5.1/CUDA 12.4 build for the machine, then:

```bash
python -m pip install -r requirements.txt -r requirements-unimc-legacy.txt
```

Clone [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM) alongside this component as `../Fengshenbang-LM-main` (excluded from Git), or pass `--fengshen-root` explicitly. Do not install the upstream project's complete requirements into a working legacy environment without reviewing them.

The original experiment did not record an upstream Git commit. Model/encoder/MLM fingerprints provide checks for compatibility, but a current upstream checkout is not automatically guaranteed to match the archived one. Use the original verified checkout for archived signed checkpoints; a changed implementation needs a new compatible extraction/training run. Do not bypass signature checks.

Download the pinned backbone separately:

```bash
python tools/download_backbone.py --output ../models/unimc-albert-235m
```

The helper verifies all four recorded model/tokenizer file hashes. It downloads about 826 MB of PyTorch weights; the weights are not tracked here.

### CPU verification

```bash
python -m unittest discover -s tests -v
python smoke_test.py
```

Set `FENGSHEN_ROOT` if the optional tiny official-ALBERT test should use a nondefault checkout. That test skips if transformers or the checkout is unavailable. Full CPU success does not establish GPU reproducibility or accuracy.

### Generate source data

```bash
python -u prepare_pilot_data.py --raw-dir raw/pilot_seed42_v2 \
  --output data/pilot_seed42_v2 --limit-train 2000 --seed 42
```

The script downloads PIQA and CommonsenseQA, removes within-pool duplicates and train/holdout overlap, and produces canonical train/val/cal JSONL and a provenance file. Compare counts and hashes with the recorded source-data metadata. Later dataset changes may prevent exact regeneration. Use data only under its upstream terms. The component includes only a synthetic example, not the benchmark contents.

### Fresh residual-head workflow

Run expensive operations inside a GPU allocation, not on the cluster login node. From the `unimc_model_optimization/` component directory:

```bash
export MKL_THREADING_LAYER=GNU
mkdir -p cache/raw predictions
for task in piqa cqa; do
  for split in train val cal; do
    python -u extract_features.py \
      --input "data/pilot_seed42_v2/${task}/${split}.jsonl" \
      --output "cache/raw/${task}_${split}.pt" \
      --model ../models/unimc-albert-235m --fengshen-root ../Fengshenbang-LM-main \
      --batch-size 2 --max-length 512 --amp none \
      --verify-official --allow-option-truncation --device cuda:0
  done
done

python project_mlm_features.py --cache cache/raw/*.pt --output-dir cache/mlm \
  --model ../models/unimc-albert-235m --fengshen-root ../Fengshenbang-LM-main --device cuda:0
python center_option_features.py --cache cache/mlm/*.pt --output-dir cache/centered

for variant in mlm centered; do
  for seed in 42 43 44; do
    for mode in mlp self full; do
      python train_head.py \
        --train "cache/${variant}/piqa_train.pt" "cache/${variant}/cqa_train.pt" \
        --val "cache/${variant}/piqa_val.pt" "cache/${variant}/cqa_val.pt" \
        --output "runs/${variant}_${mode}_seed${seed}" --mode "$mode" --seed "$seed" \
        --model-dim 64 --heads 4 --ffn-dim 256 --dropout 0.1 \
        --lr 1e-4 --weight-decay 0.01 --batch-size 128 --epochs 10 --patience 2 --device cuda:0
    done
  done
done
```

Output directories must be new. Feature and model signatures must match. The old job recipes in `slurm/` and default helper arguments contain recorded artifact IDs; they are historical layouts, not a promise to run every stage from an empty clone. Reconfigure their reference paths or restore the required private run artifacts. The fresh loop above establishes new run paths rather than impersonating archived runs.

For Gate and confirmation helpers, inspect `--help` and provide their required cache/reference reports. Tests cover small end-to-end workflows, but the repository does not include the external caches/weights that historical reports reference.

### Fresh LoRA workflow

First create the reference baseline from the same raw validation caches:

```bash
python predict_head.py --cache cache/raw/piqa_val.pt cache/raw/cqa_val.pt \
  --baseline --output predictions/baseline_val.jsonl --device cuda:0

python run_unimc_lora.py --mode check --output runs/lora_check_seed42 \
  --reference-metadata cache/raw/piqa_val.pt.metadata.json \
  --reference-predictions predictions/baseline_val.jsonl --device cuda:0 --amp bf16

python run_unimc_lora.py --mode train --seed 42 --lr 1e-4 --output runs/lora_seed42 \
  --reference-metadata cache/raw/piqa_val.pt.metadata.json \
  --reference-predictions predictions/baseline_val.jsonl --device cuda:0 --amp bf16

python diagnose_unimc_lora.py --run-dir runs/lora_seed42 \
  --output runs/lora_diag_seed42 --device cuda:0
```

The check uses two optimizer steps on temporary adapters. Actual training reinitializes B to zero and requires a compatible passing report. Diagnostics use FP32 and perform no updates. Seed43/44 LoRA are not in the current result set. A lower-learning-rate experiment should use a separate output directory and fixed settings; it has not been run here.

### Slurm and checkpoint handling

Submit from the `unimc_model_optimization/` component directory after creating `logs/`. Templates retain `MGPU-TC2`/`normal`; adapt those directives to your authorized cluster resources. `UNIMC_CONDA_ENV` defaults to `unimc`, and `UNIMC_CODE_DIR` defaults to the Slurm submit directory. Log files use `logs/%x_%j.out` and `.err`.

Source implementations are byte-preserved in Git because checkpoints contain SHA signatures. Avoid automatic Python line-ending conversion or formatting when loading archived models. If code changes intentionally, retrain or regenerate compatible artifacts rather than weakening checks.

Small metric exports are included; raw weights, signed training checkpoints, feature caches, calibration logits, and full handoff archives must be acquired separately. Trained adapters alone are insufficient for inference: the matching backbone and official encoder are also required.

## GitHub publication instructions

远程仓库已命名为 [Yuxiang5555/AI6103-group-project](https://github.com/Yuxiang5555/AI6103-group-project)。本地文件夹仍可叫 `unimc-model-optimization`，不需要因为远程名称变化而移动或重命名目录。建议课程项目协作阶段先使用 Private 仓库，邀请组员，再由团队决定何时公开和采用什么许可证。

仓库根目录只保留统一首页和 Git 配置，完整模型优化代码及实验材料已放进 `unimc_model_optimization/`。23 个原始 Python 文件保持原始字节，原研究目录没有修改。数据、基座权重、缓存和个人交接包不在普通 Git 文件中。

### 1. 创建 GitHub 空仓库

远程空仓库已经创建；无需再次新建。若以后从头操作，可在 [GitHub 新建仓库页面](https://github.com/new) 选择名称 `AI6103-group-project`，不要初始化 README、`.gitignore` 或 LICENSE，因为本地已有对应内容或许可状态说明。

本项目的 HTTPS 地址为 `https://github.com/Yuxiang5555/AI6103-group-project.git`。

### 2. 在本机 Windows PowerShell 初始化

当前工作区已经完成本节的初始化和首次提交，无需重复。若使用ZIP解压到一个新目录，再执行本节。

下面必须在这份独立仓库目录运行，不要在包含其他课程实验的工作区根目录运行 `git add .`：

```powershell
cd "C:\path\to\unimc-model-optimization"
git init -b main
git config core.autocrlf false
git add .
git status --short
git diff --cached --stat
git commit -m "Add UniMC optimization methods, experiments, and diagnostics"
```

如果 Git 首次使用提示身份未配置，只在本仓库设置自己的名字和邮箱，然后再次执行 commit：

```powershell
git config user.name "Your Name"
git config user.email "YOUR_GITHUB_EMAIL"
```

邮箱可用自己 GitHub 账号提供的 noreply 地址；不要填写仓库整理者或虚构身份。

### 3. 连接并推送

如果本地已有指向旧名称的 origin，更新地址：

```powershell
git remote set-url origin https://github.com/Yuxiang5555/AI6103-group-project.git
git remote -v
git push -u origin main
```

如果是在没有 origin 的全新本地副本中，改用 `git remote add origin https://github.com/Yuxiang5555/AI6103-group-project.git`。HTTPS 推送需要 GitHub 认证，按 Git Credential Manager/浏览器提示登录自己的账号即可；不要在脚本里写密码或 token。当前本机没有 GitHub CLI，所以本指南不依赖 `gh`。

### 4. 页面检查与协作

打开仓库，确认英文首页、结果表和图能正常显示。在 Actions 中查看首次 CPU contract tests；本地通过不等于 GitHub 托管工作流已经运行。邀请组员加入协作，分支和 Pull Request 用于后续改动。

后续更新在 Git 仓库根目录运行：

```powershell
git status
git add README.md unimc_model_optimization .github
git diff --cached --stat
git commit -m "Update experiment documentation and results"
git push
```

修改代码时按实际路径添加相应文件。上传前查看 staged diff，避免误把其他课程作业或运行产物混入仓库。

### 5. 大文件与交接包

普通仓库保留源码、表格和图。基座约826 MB，应通过 Hugging Face 下载；缓存和数据通过准备脚本重建。已有训练头、adapter 或交接包如需分发，可由团队另行放到适当的附件/Release 存储，记录版本与SHA，不要直接取消全部忽略规则。

GitHub 普通 Git 文件超过100 MiB会被阻止，网页上传单文件限制25 MiB。参考 [GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。本模板的文件远小于这些限制。

完整流程依据 [GitHub 官方本地项目上传指南](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=windows)。当前工作区已有首次提交和仓库名称更新提交；origin 已改为新仓库地址。本次整理未执行推送。

## Local validation

Checked locally on 20 November 2026:

- 61 CPU tests: 60 passed, 1 skipped because transformers was unavailable for the optional official-ALBERT test.
- Synthetic smoke test passed: training, source selection, save/load, prediction export, and baseline export.
- All 13 Slurm templates passed `bash -n`.
- The 23 original root Python implementations retain their original byte hashes.
- A temporary Git repository confirmed that Python bytes survive staging even with `core.autocrlf=true` because of `.gitattributes`.
- Actual Git ignore checks excluded representative weights, caches, data, predictions, logs, and environment files, while keeping synthetic examples and result exports.
- No private machine identifiers, model-weight files, oversized files, or missing local Markdown link targets were found in the source package.

GitHub-hosted Actions has not yet run, and GPU experiments were not rerun during repository preparation. These checks validate the source packaging and local CPU behavior; the recorded research results retain their original experimental scope.

The component relocation and documentation consolidation preserve the 23 original
Python implementation hashes. CPU and smoke checks are rerun after relocation.

## Third-party resources and license status

This project contains the project's optimization implementations and small result exports. It does not vendor the full Fengshenbang-LM checkout, the pretrained UniMC weights, or benchmark datasets.

- Official code: [IDEA-CCNL/Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM), whose upstream repository specifies [Apache License 2.0](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/LICENSE).
- Base model: [IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English](https://huggingface.co/IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English), revision `e3aeebc7bd475d559bd0cbd8d17abf4e159d1260`. Obtain and use it under its upstream asset terms.
- Source benchmarks: PIQA and CommonsenseQA, downloaded separately by the preparation script. Dataset contents are not included here.
- LoRA method: [Hu et al., LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- UniMC paper: [Zero-Shot Learners for Natural Language Understanding via a Unified Multiple Choice Perspective](https://aclanthology.org/2022.emnlp-main.474/).

No new project-wide license has been assigned by this repository preparation. The project team should decide the license for its own contributions before an open-source release; this file does not change third-party rights or terms.

## Layout consolidation checks

After relocation, the Git repository root has three files: README.md, .gitignore,
and .gitattributes. The original implementation and its working-directory layout
are preserved inside unimc_model_optimization/. Six historical notes are combined
into one guide; the method, result, and LoRA documents are combined into the
technical report; operational, publishing, validation, and attribution material
are combined into this guide. The three package-export notes and the complete
handoff_reference bundle remain at their required component-relative paths.

All 23 original Python implementation hashes are unchanged. The relocated suite
ran 61 tests: 60 passed and the optional official-ALBERT test skipped because
transformers was unavailable locally. The synthetic smoke test and all 13 Slurm
syntax checks passed. A temporary Git repository also verified the new ignore
rules and Python-byte preservation. GitHub-hosted Actions status was not checked
during this restructuring; GPU experiments were not rerun.
