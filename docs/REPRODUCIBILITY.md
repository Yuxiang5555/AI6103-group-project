# Reproduction and artifact requirements

## Environment and external dependencies

Use a separate Python 3.10 environment. The recorded GPU stack is described in `results/base_model_signature.json`. Install the appropriate PyTorch 2.5.1/CUDA 12.4 build for the machine, then:

```bash
python -m pip install -r requirements.txt -r requirements-unimc-legacy.txt
```

Clone [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM) outside this repository, naming the folder `../Fengshenbang-LM-main`, or pass `--fengshen-root` explicitly. Do not install the upstream project's complete requirements into a working legacy environment without reviewing them.

The original experiment did not record an upstream Git commit. Model/encoder/MLM fingerprints provide checks for compatibility, but a current upstream checkout is not automatically guaranteed to match the archived one. Use the original verified checkout for archived signed checkpoints; a changed implementation needs a new compatible extraction/training run. Do not bypass signature checks.

Download the pinned backbone separately:

```bash
python tools/download_backbone.py --output ../models/unimc-albert-235m
```

The helper verifies all four recorded model/tokenizer file hashes. It downloads about 826 MB of PyTorch weights; the weights are not tracked here.

## CPU verification

```bash
python -m unittest discover -s tests -v
python smoke_test.py
```

Set `FENGSHEN_ROOT` if the optional tiny official-ALBERT test should use a nondefault checkout. That test skips if transformers or the checkout is unavailable. Full CPU success does not establish GPU reproducibility or accuracy.

## Generate source data

```bash
python -u prepare_pilot_data.py --raw-dir raw/pilot_seed42_v2 \
  --output data/pilot_seed42_v2 --limit-train 2000 --seed 42
```

The script downloads PIQA and CommonsenseQA, removes within-pool duplicates and train/holdout overlap, and produces canonical train/val/cal JSONL and a provenance file. Compare counts and hashes with the recorded source-data metadata. Later dataset changes may prevent exact regeneration. Use data only under its upstream terms. The repository includes only a synthetic example, not the benchmark contents.

## Fresh residual-head workflow

Run expensive operations inside a GPU allocation, not on the cluster login node. From the repository root:

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

## Fresh LoRA workflow

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

## Slurm and checkpoint handling

Submit from the repository root after creating `logs/`. Templates retain `MGPU-TC2`/`normal`; adapt those directives to your authorized cluster resources. `UNIMC_CONDA_ENV` defaults to `unimc`, and `UNIMC_CODE_DIR` defaults to the Slurm submit directory. Log files use `logs/%x_%j.out` and `.err`.

Source implementations are byte-preserved in Git because checkpoints contain SHA signatures. Avoid automatic Python line-ending conversion or formatting when loading archived models. If code changes intentionally, retrain or regenerate compatible artifacts rather than weakening checks.

Small metric exports are included; raw weights, signed training checkpoints, feature caches, calibration logits, and full handoff archives must be acquired separately. Trained adapters alone are insufficient for inference: the matching backbone and official encoder are also required.
