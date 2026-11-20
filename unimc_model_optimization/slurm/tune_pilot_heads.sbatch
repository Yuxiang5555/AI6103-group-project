#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=unimc_tune
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
cache_dir="cache/pilot_55518"
reference_dir="runs/pilot_55518"
run_dir="runs/tune_pilot_55518_${SLURM_JOB_ID}"
mkdir "$run_dir"
printf 'REFERENCE_DIR=%s\nCACHE_DIR=%s\nRESULTS_DIR=%s\n' "$reference_dir" "$cache_dir" "$run_dir"

# Compare only the learning rate: preserve the original input files and code.
python -u - "$cache_dir" "$reference_dir" "$run_dir" <<'PY_PREFLIGHT'
import json
from pathlib import Path
import sys

from data_io import FeatureDataset, assert_disjoint, file_sha256

cache_dir, reference_dir, run_dir = map(Path, sys.argv[1:])
reference = json.loads((reference_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
for filename in ("decision_head.py", "train_head.py", "data_io.py"):
    if file_sha256(filename) != reference["scripts_sha256"][filename]:
        raise ValueError(f"Model/training code changed since the reference experiment: {filename}")
expected_settings = {
    "model_dim": 256, "heads": 4, "ffn_dim": 1024, "dropout": 0.1,
    "weight_decay": 0.01, "epochs": 10, "patience": 2,
    "batch_size": 128, "seed": 42, "sampling": "balanced", "gate": False,
}
for mode in ("mlp", "self", "full"):
    config = json.loads((reference_dir / f"{mode}_seed42_lr1e4" / "config.json").read_text())
    if config["arguments"]["lr"] != 0.0001 or config["arguments"]["mode"] != mode:
        raise ValueError(f"Unexpected original mode/learning rate: {mode}")
    for key, expected in expected_settings.items():
        if config["arguments"][key] != expected:
            raise ValueError(f"Unexpected original setting: {mode}/{key}")
    for path, digest in config["data_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError(f"Reference cache changed: {path}")
train = FeatureDataset([cache_dir / f"{task}_train.pt" for task in ("piqa", "cqa")], required_role="source_train")
val = FeatureDataset([cache_dir / f"{task}_val.pt" for task in ("piqa", "cqa")], required_role="source_val")
assert_disjoint(train, val)
if train.signature != val.signature or train.signature != reference["feature_signature"]:
    raise ValueError("Reference/train/validation signatures differ")
if len(train) != 4000 or len(val) != 1517:
    raise ValueError("Unexpected pilot train/validation sizes")
manifest = {
    "reference_dir": str(reference_dir), "cache_dir": str(cache_dir),
    "fixed_settings": expected_settings, "learning_rates": [0.00001, 0.00003],
    "feature_signature": train.signature, "source_cal_used": False, "target_data_used": False,
    "reference_manifest_sha256": file_sha256(reference_dir / "experiment_manifest.json"),
    "selection": "Source validation macro accuracy; tie-break macro NLL. Trained checkpoints and baseline reported separately.",
}
(run_dir / "tuning_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("REFERENCE AND CACHE CHECK PASSED", flush=True)
PY_PREFLIGHT

for lr_tag in 1e5 3e5; do
    case "$lr_tag" in
        1e5) lr=0.00001 ;;
        3e5) lr=0.00003 ;;
    esac
    for mode in mlp self full; do
        printf '\nTRAINING mode=%s lr=%s seed=42\n' "$mode" "$lr"
        python -u train_head.py \
            --train "${cache_dir}/piqa_train.pt" "${cache_dir}/cqa_train.pt" \
            --val "${cache_dir}/piqa_val.pt" "${cache_dir}/cqa_val.pt" \
            --mode "$mode" \
            --output "${run_dir}/${mode}_seed42_lr${lr_tag}" \
            --model-dim 256 \
            --heads 4 \
            --ffn-dim 1024 \
            --dropout 0.1 \
            --weight-decay 0.01 \
            --lr "$lr" \
            --epochs 10 \
            --patience 2 \
            --batch-size 128 \
            --sampling balanced \
            --seed 42 \
            --threads 4 \
            --device cuda:0
    done
done

# Report every run; use only validation metrics for this tuning step.
python -u - "$reference_dir" "$run_dir" "$SECONDS" <<'PY_REPORT'
import csv
import json
from pathlib import Path
import sys

reference_dir, run_dir = map(Path, sys.argv[1:3])
baseline = json.loads((reference_dir / "mlp_seed42_lr1e4" / "summary.json").read_text())["baseline_source_validation"]

def record(mode, lr, epoch, metrics, source, parameters=None):
    return {
        "mode": mode, "lr": lr, "seed": 42, "best_epoch": epoch,
        "piqa_accuracy": metrics["per_dataset"]["piqa"]["accuracy"],
        "cqa_accuracy": metrics["per_dataset"]["commonsenseqa"]["accuracy"],
        "macro_accuracy": metrics["macro_accuracy"], "macro_nll": metrics["macro_nll"],
        "delta_macro_pp": 100 * (metrics["macro_accuracy"] - baseline["macro_accuracy"]),
        "parameter_count": parameters, "run_dir": str(source),
    }

records = [record("baseline", None, 0, baseline, reference_dir)]
sources = [(reference_dir, "1e4", 0.0001), (run_dir, "3e5", 0.00003), (run_dir, "1e5", 0.00001)]
for source, tag, lr in sources:
    for mode in ("mlp", "self", "full"):
        folder = source / f"{mode}_seed42_lr{tag}"
        summary = json.loads((folder / "summary.json").read_text())
        if summary.get("synthetic_only"):
            raise ValueError(f"Synthetic results cannot enter the real experiment report: {folder}")
        if summary["baseline_source_validation"] != baseline:
            raise ValueError(f"Reference validation baseline differs: {folder}")
        records.append(record(mode, lr, summary["best_epoch"], summary["best_source_validation"], folder, summary["parameter_count"]))
with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
trained = records[1:]
winner = max(trained, key=lambda row: (row["macro_accuracy"], -row["macro_nll"]))
report = {
    "scope": "Source validation tuning, seed 42 only; not target generalization results",
    "elapsed_seconds": int(sys.argv[3]), "comparison": records,
    "best_trained_on_source_val": winner,
    "best_trained_accuracy_above_baseline": winner["macro_accuracy"] > baseline["macro_accuracy"],
    "selection_uses_source_cal": False,
}
(run_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("SOURCE VALIDATION LEARNING RATE COMPARISON", flush=True)
for row in records:
    lr = "--" if row["lr"] is None else f"{row['lr']:.0e}"
    print(f"{row['mode']:8s} lr={lr:5s} epoch={row['best_epoch']:2d} "
          f"PIQA={100*row['piqa_accuracy']:.2f}% CQA={100*row['cqa_accuracy']:.2f}% "
          f"Macro={100*row['macro_accuracy']:.2f}% Delta={row['delta_macro_pp']:+.2f}pp "
          f"NLL={row['macro_nll']:.4f}", flush=True)
print("BEST_TRAINED_ABOVE_BASELINE=" + str(report["best_trained_accuracy_above_baseline"]), flush=True)
print(f"RESULTS_DIR={run_dir}", flush=True)
print("PILOT HEAD TUNING COMPLETE", flush=True)
PY_REPORT
