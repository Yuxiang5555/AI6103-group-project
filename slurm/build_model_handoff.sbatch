#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=model_handoff
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
# Submit from repository root after: mkdir -p logs
export MKL_THREADING_LAYER=GNU
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${UNIMC_CONDA_ENV:-unimc}"
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "${UNIMC_CODE_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
python -u build_model_handoff.py \
    --output "handoff/model_optimization_${SLURM_JOB_ID}" \
    --report runs/centering_confirm_55757/comparison.json \
    --model ../models/unimc-albert-235m \
    --fengshen-root ../Fengshenbang-LM-main --device cuda:0
