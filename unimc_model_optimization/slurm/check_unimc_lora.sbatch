#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=lora_check
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
# Submit from unimc_model_optimization/ after: mkdir -p logs
export MKL_THREADING_LAYER=GNU
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${UNIMC_CONDA_ENV:-unimc}"
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "${UNIMC_CODE_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
python -m unittest discover -s tests -p test_unimc_lora.py -v
python -u run_unimc_lora.py --mode check --output "runs/lora_check_${SLURM_JOB_ID}" --device cuda:0 --amp bf16
