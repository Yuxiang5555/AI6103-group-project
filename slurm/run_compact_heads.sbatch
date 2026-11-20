#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=unimc_compact
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
raw_cache_dir="cache/pilot_55518"
reference_dir="runs/pilot_55518"
mlm_cache_dir="cache/mlm_pilot_55518_${SLURM_JOB_ID}"
run_dir="runs/compact_pilot_55518_${SLURM_JOB_ID}"
mkdir "$run_dir"
mkdir "$run_dir/raw" "$run_dir/mlm"
printf 'REFERENCE_DIR=%s\nRESULTS_DIR=%s\nMLM_CACHE_DIR=%s\n' "$reference_dir" "$run_dir" "$mlm_cache_dir"

python -u - "$raw_cache_dir" "$reference_dir" <<'PY_PREFLIGHT'
import json
from pathlib import Path
import sys
from data_io import FeatureDataset, assert_disjoint, file_sha256

cache_dir, reference_dir = map(Path, sys.argv[1:])
manifest = json.loads((reference_dir / "experiment_manifest.json").read_text())
for filename in ("decision_head.py", "train_head.py", "data_io.py"):
    if file_sha256(filename) != manifest["scripts_sha256"][filename]:
        raise ValueError(f"Reference model/training code changed: {filename}")
config = json.loads((reference_dir / "full_seed42_lr1e4" / "config.json").read_text())
for path, digest in config["data_sha256"].items():
    if file_sha256(path) != digest:
        raise ValueError(f"Reference cache changed: {path}")
train = FeatureDataset([cache_dir / f"{task}_train.pt" for task in ("piqa", "cqa")], "source_train")
val = FeatureDataset([cache_dir / f"{task}_val.pt" for task in ("piqa", "cqa")], "source_val")
if len(train) != 4000 or len(val) != 1517 or train.signature != val.signature:
    raise ValueError("Incorrect pilot data counts/signatures")
if train.signature != manifest["feature_signature"]:
    raise ValueError("Reference/raw cache signatures differ")
assert_disjoint(train, val)
print("RAW CACHE CHECK PASSED: train=4000 val=1517", flush=True)
PY_PREFLIGHT

python -u project_mlm_features.py \
    --cache "${raw_cache_dir}/piqa_train.pt" "${raw_cache_dir}/cqa_train.pt" \
            "${raw_cache_dir}/piqa_val.pt" "${raw_cache_dir}/cqa_val.pt" \
    --output-dir "$mlm_cache_dir" \
    --model ../models/unimc-albert-235m \
    --fengshen-root ../Fengshenbang-LM-main \
    --batch-size 128 \
    --threads 4 \
    --device cuda:0

# The new representation must preserve every original score and sample exactly.
python -u - "$raw_cache_dir" "$mlm_cache_dir" <<'PY_PROJECTED'
from pathlib import Path
import sys
import torch
from data_io import FeatureDataset

raw_dir, mlm_dir = map(Path, sys.argv[1:])
signature = None
for split in ("train", "val"):
    for task in ("piqa", "cqa"):
        raw = FeatureDataset([raw_dir / f"{task}_{split}.pt"], f"source_{split}")
        mlm = FeatureDataset([mlm_dir / f"{task}_{split}.pt"], f"source_{split}")
        if len(raw) != len(mlm):
            raise ValueError("Projection changed sample counts")
        if signature is None:
            signature = mlm.signature
        if mlm.signature != signature:
            raise ValueError("Projected signatures differ across splits/tasks")
        for before, after in zip(raw.samples, mlm.samples):
            for key in ("id", "dataset", "choices", "gold_label", "content_hash"):
                if before[key] != after[key]:
                    raise ValueError(f"Projection changed {key}")
            torch.testing.assert_close(before["base_logits"], after["base_logits"], atol=0, rtol=0)
print("PROJECTED CACHE CHECK PASSED: original logits and rows preserved", flush=True)
print(f"MLM_FEATURE_DIM={signature['hidden_dim']}", flush=True)
PY_PROJECTED

for representation in raw mlm; do
    case "$representation" in
        raw) feature_dir="$raw_cache_dir" ;;
        mlm) feature_dir="$mlm_cache_dir" ;;
    esac
    for mode in mlp self full; do
        printf '\nTRAINING representation=%s mode=%s dim=64 lr=0.0001 seed=42\n' "$representation" "$mode"
        python -u train_head.py \
            --train "${feature_dir}/piqa_train.pt" "${feature_dir}/cqa_train.pt" \
            --val "${feature_dir}/piqa_val.pt" "${feature_dir}/cqa_val.pt" \
            --mode "$mode" \
            --output "${run_dir}/${representation}/${mode}_seed42_lr1e4" \
            --model-dim 64 \
            --heads 4 \
            --ffn-dim 256 \
            --dropout 0.1 \
            --lr 0.0001 \
            --weight-decay 0.01 \
            --epochs 10 \
            --patience 2 \
            --batch-size 128 \
            --sampling balanced \
            --seed 42 \
            --threads 4 \
            --device cuda:0
    done
done

python -u - "$reference_dir" "$run_dir" "$mlm_cache_dir" "$SECONDS" <<'PY_REPORT'
import csv
import json
from pathlib import Path
import sys
from data_io import file_sha256

reference_dir, run_dir, mlm_cache_dir = map(Path, sys.argv[1:4])
baseline = json.loads((reference_dir / "full_seed42_lr1e4" / "summary.json").read_text())["baseline_source_validation"]

def record(name, mode, width, epoch, metrics, parameters, source):
    return {"representation": name, "mode": mode, "model_dim": width, "best_epoch": epoch,
            "parameter_count": parameters,
            "piqa_accuracy": metrics["per_dataset"]["piqa"]["accuracy"],
            "cqa_accuracy": metrics["per_dataset"]["commonsenseqa"]["accuracy"],
            "macro_accuracy": metrics["macro_accuracy"], "macro_nll": metrics["macro_nll"],
            "delta_macro_pp": 100 * (metrics["macro_accuracy"] - baseline["macro_accuracy"]),
            "run_dir": str(source)}

rows = [record("baseline", "baseline", None, 0, baseline, 0, reference_dir)]
for source, representation, width in ((reference_dir, "raw", 256),
                                      (run_dir / "raw", "raw", 64),
                                      (run_dir / "mlm", "mlm", 64)):
    for mode in ("mlp", "self", "full"):
        folder = source / f"{mode}_seed42_lr1e4"
        summary = json.loads((folder / "summary.json").read_text())
        if summary.get("synthetic_only") or summary["baseline_source_validation"] != baseline:
            raise ValueError(f"Synthetic or different validation baseline: {folder}")
        rows.append(record(representation, mode, width, summary["best_epoch"],
                           summary["best_source_validation"], summary["parameter_count"], folder))
with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
best = max(rows[1:], key=lambda row: (row["macro_accuracy"], -row["macro_nll"]))
report = {"scope": "Source validation structural pilot, seed 42 only", "lr": 0.0001,
          "calibration_data_used": False, "target_data_used": False,
          "reference_dir": str(reference_dir), "mlm_cache_dir": str(mlm_cache_dir),
          "projection_manifest": json.loads((mlm_cache_dir / "projection_manifest.json").read_text()),
          "scripts_sha256": {name: file_sha256(name) for name in
                             ("project_mlm_features.py", "decision_head.py", "train_head.py", "data_io.py")},
          "elapsed_seconds": int(sys.argv[4]), "comparison": rows, "best_trained_on_source_val": best,
          "best_trained_above_baseline": best["macro_accuracy"] > baseline["macro_accuracy"]}
(run_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("SOURCE VALIDATION COMPACT HEAD COMPARISON", flush=True)
for row in rows:
    width = "--" if row["model_dim"] is None else str(row["model_dim"])
    print(f"{row['representation']:8s} {row['mode']:8s} d={width:3s} "
          f"params={row['parameter_count']:7d} epoch={row['best_epoch']:2d} "
          f"PIQA={100*row['piqa_accuracy']:.2f}% CQA={100*row['cqa_accuracy']:.2f}% "
          f"Macro={100*row['macro_accuracy']:.2f}% Delta={row['delta_macro_pp']:+.2f}pp "
          f"NLL={row['macro_nll']:.4f}", flush=True)
print("BEST_TRAINED_ABOVE_BASELINE=" + str(report["best_trained_above_baseline"]), flush=True)
print(f"RESULTS_DIR={run_dir}", flush=True)
print("COMPACT HEAD EXPERIMENT COMPLETE", flush=True)
PY_REPORT
