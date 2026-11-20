#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=unimc_center
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
# Submit from repository root after: mkdir -p logs
export MKL_THREADING_LAYER=GNU
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${UNIMC_CONDA_ENV:-unimc}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "${UNIMC_CODE_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
parent_cache="cache/mlm_pilot_55518_55555"
reference_dir="runs/compact_pilot_55518_55555/mlm"
centered_cache="cache/centered_mlm_${SLURM_JOB_ID}"
run_dir="runs/centered_mlm_${SLURM_JOB_ID}"
mkdir "$run_dir"
printf 'RESULTS_DIR=%s\nCENTERED_CACHE_DIR=%s\n' "$run_dir" "$centered_cache"

# Freeze all settings used by the completed compact-MLM control.
python -u - "$reference_dir" "$parent_cache" <<'PY_PREFLIGHT'
import json
from pathlib import Path
import sys
from data_io import FeatureDataset, assert_disjoint, file_sha256

reference, cache = map(Path, sys.argv[1:])
expected = {"model_dim": 64, "heads": 4, "ffn_dim": 256, "dropout": 0.1,
            "lr": 0.0001, "weight_decay": 0.01, "epochs": 10, "patience": 2,
            "batch_size": 128, "sampling": "balanced", "seed": 42, "gate": False}
previous = json.loads((reference.parent / "comparison.json").read_text())
for name in ("decision_head.py", "train_head.py", "data_io.py"):
    if file_sha256(name) != previous["scripts_sha256"][name]:
        raise ValueError(f"Reference model code changed: {name}")
for mode in ("mlp", "self", "full"):
    config = json.loads((reference / f"{mode}_seed42_lr1e4/config.json").read_text())
    if any(config["arguments"][key] != value for key, value in expected.items()):
        raise ValueError(f"Reference training settings differ: {mode}")
    for path, digest in config["data_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError(f"Reference cache changed: {path}")
train = FeatureDataset([cache / f"{task}_train.pt" for task in ("piqa", "cqa")], "source_train")
val = FeatureDataset([cache / f"{task}_val.pt" for task in ("piqa", "cqa")], "source_val")
assert_disjoint(train, val)
if len(train) != 4000 or len(val) != 1517 or train.signature != val.signature:
    raise ValueError("Incorrect source cache sizes/signatures")
print("REFERENCE CHECK PASSED", flush=True)
PY_PREFLIGHT

python -u center_option_features.py \
    --cache "${parent_cache}/piqa_train.pt" "${parent_cache}/cqa_train.pt" \
            "${parent_cache}/piqa_val.pt" "${parent_cache}/cqa_val.pt" \
    --output-dir "$centered_cache"

for mode in mlp self full; do
    printf '\nTRAINING centered_mlm mode=%s lr=0.0001 seed=42\n' "$mode"
    python -u train_head.py \
        --train "${centered_cache}/piqa_train.pt" "${centered_cache}/cqa_train.pt" \
        --val "${centered_cache}/piqa_val.pt" "${centered_cache}/cqa_val.pt" \
        --mode "$mode" \
        --output "${run_dir}/${mode}_seed42_lr1e4" \
        --model-dim 64 --heads 4 --ffn-dim 256 --dropout 0.1 \
        --lr 0.0001 --weight-decay 0.01 --epochs 10 --patience 2 \
        --batch-size 128 --sampling balanced --seed 42 --threads 4 --device cuda:0

    python -u predict_head.py \
        --cache "${centered_cache}/piqa_val.pt" "${centered_cache}/cqa_val.pt" \
        --checkpoint "${run_dir}/${mode}_seed42_lr1e4/best.pt" \
        --output "${run_dir}/${mode}_val.jsonl" --threads 4 --device cuda:0
done

python -u - "$reference_dir" "$run_dir" "$centered_cache" <<'PY_REPORT'
import csv
import json
from pathlib import Path
import sys

reference, run_dir, centered_cache = map(Path, sys.argv[1:])
baseline = json.loads((reference / "full_seed42_lr1e4/summary.json").read_text())["baseline_source_validation"]
rows = []
def add(label, mode, metrics, epoch, parameters, source):
    rows.append({"variant": label, "mode": mode, "best_epoch": epoch, "parameters": parameters,
                 "piqa_accuracy": metrics["per_dataset"]["piqa"]["accuracy"],
                 "cqa_accuracy": metrics["per_dataset"]["commonsenseqa"]["accuracy"],
                 "macro_accuracy": metrics["macro_accuracy"], "macro_nll": metrics["macro_nll"],
                 "delta_macro_pp": 100*(metrics["macro_accuracy"]-baseline["macro_accuracy"]), "run_dir": str(source)})
add("baseline", "baseline", baseline, 0, 0, reference)
for label, parent in (("uncentered", reference), ("centered", run_dir)):
    for mode in ("mlp", "self", "full"):
        folder = parent / f"{mode}_seed42_lr1e4"
        summary = json.loads((folder / "summary.json").read_text())
        if summary.get("synthetic_only") or summary["baseline_source_validation"] != baseline:
            raise ValueError(f"Synthetic or different baseline: {folder}")
        add(label, mode, summary["best_source_validation"], summary["best_epoch"], summary["parameter_count"], folder)
with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
report = {"scope": "Source validation only, seed 42; no calibration or target data used",
          "centering_semantics": "All centered variants use set information; self removes cross-option attention only.",
          "centered_cache": str(centered_cache), "comparison": rows,
          "centering_manifest": json.loads((centered_cache / "centering_manifest.json").read_text())}
(run_dir / "comparison.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
print("CENTERING SOURCE VALIDATION COMPARISON", flush=True)
for row in rows:
    print(f"{row['variant']:10s} {row['mode']:8s} epoch={row['best_epoch']:2d} "
          f"PIQA={100*row['piqa_accuracy']:.2f}% CQA={100*row['cqa_accuracy']:.2f}% "
          f"Macro={100*row['macro_accuracy']:.2f}% Delta={row['delta_macro_pp']:+.2f}pp NLL={row['macro_nll']:.4f}", flush=True)
print(f"RESULTS_DIR={run_dir}", flush=True)
print("CENTERED HEAD EXPERIMENT COMPLETE", flush=True)
PY_REPORT
