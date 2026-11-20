#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=interaction_gate
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
python -m unittest discover -s tests -p test_interaction_gate.py -v
python -u run_interaction_gate.py \
    --output "runs/interaction_gate_${SLURM_JOB_ID}" \
    --cache cache/centered_mlm_55752 \
    --reference-report runs/centering_confirm_55757/comparison.json \
    --device cuda:0
