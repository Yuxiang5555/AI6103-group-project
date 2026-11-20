#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=05:50:00
#SBATCH --job-name=unimc_lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
# Submit from repository root after: mkdir -p logs
export MKL_THREADING_LAYER=GNU
TASK_SEED="${1:-42}"
case "$TASK_SEED" in
    42|43|44) ;;
    *) echo "Use seed 42, 43 or 44" >&2; exit 2 ;;
esac
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${UNIMC_CONDA_ENV:-unimc}"
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "${UNIMC_CODE_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
python -u run_unimc_lora.py --mode train --seed "$TASK_SEED" \
    --output "runs/lora_seed${TASK_SEED}_${SLURM_JOB_ID}" --device cuda:0 --amp bf16
