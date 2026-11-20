#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=pilot_check
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
# Submit from unimc_model_optimization/ after: mkdir -p logs
export MKL_THREADING_LAYER=GNU

module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${UNIMC_CONDA_ENV:-unimc}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "${UNIMC_CODE_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cache_dir="cache/pilot_check_${SLURM_JOB_ID}"
prediction_dir="predictions/pilot_check_${SLURM_JOB_ID}"
mkdir -p "$cache_dir" "$prediction_dir"

for task in piqa cqa; do
    python -u extract_features.py \
        --fengshen-root ../Fengshenbang-LM-main \
        --model ../models/unimc-albert-235m \
        --input "data/pilot_seed42_v2/${task}/train.jsonl" \
        --output "${cache_dir}/${task}_64.pt" \
        --limit 64 \
        --batch-size 2 \
        --amp none \
        --verify-official \
        --allow-option-truncation \
        --device cuda:0

    python -u predict_head.py \
        --cache "${cache_dir}/${task}_64.pt" \
        --baseline \
        --output "${prediction_dir}/${task}_64_baseline.jsonl" \
        --device cuda:0
done

python -u - "$cache_dir" "$prediction_dir" <<'PY'
import json
from pathlib import Path
import sys

cache_dir, prediction_dir = map(Path, sys.argv[1:])
for task in ("piqa", "cqa"):
    metadata = json.loads((cache_dir / f"{task}_64.pt.metadata.json").read_text())
    if metadata["n"] != 64 or metadata["role"] != "source_train":
        raise ValueError(f"Incorrect number of examples or source role: {task}")
    if not metadata["official_prediction_check"]["passed"]:
        raise ValueError(f"Official parity check failed: {task}")
    metrics = json.loads((prediction_dir / f"{task}_64_baseline.jsonl.metrics.json").read_text())
    print(json.dumps({
        "task": task, "n": metadata["n"],
        "hidden_dim": metadata["feature_signature"]["hidden_dim"],
        "official_check_passed": metadata["official_prediction_check"]["passed"],
        "truncated_option_examples": metadata["truncated_option_examples"],
        "baseline_accuracy_on_train_subset": metrics["accuracy"],
    }), flush=True)
print("PILOT FEATURE CHECK COMPLETE", flush=True)
PY
