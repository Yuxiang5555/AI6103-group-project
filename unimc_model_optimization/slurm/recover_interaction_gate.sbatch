#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=gate_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
# Submit from unimc_model_optimization/ after: mkdir -p logs
export MKL_THREADING_LAYER=GNU
if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch slurm/recover_interaction_gate.sbatch runs/interaction_gate_ORIGINAL_JOB_ID" >&2
    exit 2
fi
TASK_RUN_DIR="$1"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${UNIMC_CONDA_ENV:-unimc}"
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "${UNIMC_CODE_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
python -u run_interaction_gate.py --output "$TASK_RUN_DIR" --report-only --device cuda:0
