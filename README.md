# AI6103 Group Project — UniMC Model Optimization

The model-optimization component of [AI6103-group-project](https://github.com/Yuxiang5555/AI6103-group-project): an experimental study of residual option scoring, option centering, interaction gating, and LoRA adaptation for the English UniMC ALBERT model.

This repository contains the model-optimization part of the project: implementation, CPU contract tests, Slurm recipes, source-development results, and diagnostic reports. Calibration/rejection fitting and final independent-target evaluation are separate downstream tasks.

## Current evidence

All reported accuracy results use the same PIQA/CommonsenseQA **source validation** split. They are not independent target-task results. Macro accuracy is the equal-weight average of the two task accuracies.

| Method | Seeds | Source macro accuracy | Seed SD (percentage points) | Difference from baseline (pp) |
|---|---|---:|---:|---:|
| Original UniMC | fixed baseline | 76.4164% | — | 0.0000 |
| Centered MLM features + MLP head | 42, 43, 44 | 76.4792% | 0.3394 | +0.0628 |
| Centered MLM features + self-attention control | 42, 43, 44 | 76.3788% | 0.2288 | −0.0376 |
| Centered MLM features + full option attention | 42, 43, 44 | 76.5336% | 0.1349 | +0.1172 |
| Centered features + interaction Gate | 42, 43, 44 | 76.4332% | 0.0674 | +0.0168 |
| Native-objective LoRA, best trained checkpoint | 42 only | 76.1112% | — | −0.3052 |

The centered full head is a candidate for independent evaluation. Its exploratory paired-question interval for improvement over baseline includes zero. Gate did not improve mean accuracy over full attention. The tested LoRA configuration did not improve accuracy. No stable target-task gain is claimed. See [complete results and interpretation](docs/RESULTS.md).

## What is implemented

1. **Frozen-backbone residual heads:** official UniMC option-marker hidden states and baseline yes logits; a small MLP/self/full head adds a score correction. Initial corrections are exactly zero.
2. **Representation experiments:** raw 4096-dimensional states, projection through the official frozen MLM predecoder to 128 dimensions, and within-question option centering. The initial 256-dimensional scoring heads and later compact 64-dimensional heads are separate experiments.
3. **Interaction Gate:** a learned scalar interpolates self-only and cross-option attention. The real Gate study used the centered, compact 64-dimensional configuration.
4. **Shared-layer LoRA:** frozen ALBERT base parameters, low-rank updates to the physical query/value projections, official online encoding, and the native OP + MLM objective. Rank 8 introduces 131,072 parameters in two shared physical modules.
5. **Read-only loss diagnostics:** fixed source-training questions, three saved LoRA checkpoints, separate OP/option-MLM/passage-MLM gradients, and explicit FP32 consistency checks.

Centering itself uses all candidates in a question. Accordingly, the centered self/MLP controls remove cross-option attention, not all candidate-set dependence. See [method details](docs/METHODS.md).

## Repository map

```text
*.py                         Core preparation, extraction, training, prediction, and diagnostics
tests/                       CPU contract tests; optional tiny official-ALBERT check
slurm/                       Batch templates, with relative logs and configurable code location
handoff_reference/           Question-paired analysis helper
tools/                       Pinned base-model download helper
docs/                        Methods, results, reproduction, and GitHub publication guide
reports/                     English LoRA report and figures
results/                     Small JSON/CSV result exports and provenance hashes
examples/                    Synthetic canonical-input example
requirements*.txt            Minimal CPU and legacy UniMC dependencies
```

Original Python implementation files retain their exact research-run bytes, including line endings, because checkpoints and feature caches verify implementation hashes. `.gitattributes` prevents Git from silently normalizing Python files. Only an optional test's private workstation fallback was replaced with `FENGSHEN_ROOT`. Slurm working-directory/log settings and historical notes were made portable. Changes are recorded in [source provenance](results/source_provenance.json).

Weights, feature caches, raw/prepared datasets, predictions, and private handoff archives are excluded from normal Git tracking. The repository is a source-and-evidence release, not a copy of an entire server account or the third-party backbone.

## Quick start: CPU checks

Use Python 3.10. The core synthetic tests require only PyTorch and NumPy:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python smoke_test.py
```

An optional official-ALBERT test skips when its external checkout/dependencies are unavailable. These tests verify contracts and synthetic workflows; they do not substitute for a GPU experiment. GitHub Actions is configured for the CPU checks, but a hosted run is only available after publication.

## Real UniMC experiments

The recorded environment used Python 3.10.22, PyTorch 2.5.1 with CUDA 12.4, transformers 4.20.1, and pytorch-lightning 1.9.5. Keep this legacy environment isolated; the implementation does not require PEFT. Install the matching PyTorch build first, then the extra dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-unimc-legacy.txt
```

The external code and base model are expected alongside the repository:

```text
project/
├── unimc-model-optimization/
├── Fengshenbang-LM-main/
└── models/unimc-albert-235m/
```

The model is `IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English`, pinned to revision `e3aeebc7bd475d559bd0cbd8d17abf4e159d1260`. [Reproduction instructions](docs/REPRODUCIBILITY.md) cover dependency setup, downloading the backbone, generating disjoint source splits, feature extraction, residual-head training, and LoRA checks/training.

For Slurm, submit from the repository root after `mkdir -p logs`. Templates retain the original partition/QoS and historical artifact dependencies; adjust your cluster directives and input paths before submitting. `UNIMC_CONDA_ENV` selects the environment; `UNIMC_CODE_DIR` optionally overrides the code directory. New logs are in `logs/`, rather than a private absolute directory.

## Reports and next experiment

- [English LoRA training and diagnostic report](reports/LORA_REPORT_EN.md)
- [All methods and their scope](docs/METHODS.md)
- [Results, negative findings, and planned work](docs/RESULTS.md)
- [GitHub upload instructions in Chinese](docs/PUBLISH_TO_GITHUB_ZH.md)

The proposed learning-rate comparison (`1e-4` versus `1e-5`, otherwise fixed settings) has **not** been run in the evidence included here. Independent-target, calibration, and no-answer rejection results are also not included.

## Upstream resources and licensing status

The official model and encoding implementation come from [IDEA-CCNL/Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM) and the [UniMC model repository](https://huggingface.co/IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English). They are downloaded separately. This repository has not assigned a project-wide redistribution license; the team should choose one before an open-source release. Third-party terms remain with their respective assets. See [third-party acknowledgements](THIRD_PARTY_NOTICES.md).
