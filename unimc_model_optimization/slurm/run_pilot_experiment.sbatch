#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=unimc_pilot
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
data_dir="data/pilot_seed42_v2"
cache_dir="cache/pilot_${SLURM_JOB_ID}"
run_dir="runs/pilot_${SLURM_JOB_ID}"
prediction_dir="predictions/pilot_${SLURM_JOB_ID}"
mkdir -p "$cache_dir" "$run_dir" "$prediction_dir"
printf 'CACHE_DIR=%s\nRUN_DIR=%s\nPREDICTION_DIR=%s\n' "$cache_dir" "$run_dir" "$prediction_dir"

# Check the prepared files before extracting features.
python -u - "$data_dir" <<'PY_PREFLIGHT'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
expected_counts = {"piqa": {"train": 2000, "val": 919, "cal": 919},
                   "cqa": {"train": 2000, "val": 598, "cal": 599}}
for task, splits in expected_counts.items():
    for split, expected in splits.items():
        path = root / task / f"{split}.jsonl"
        payload = path.read_bytes()
        recorded = provenance["files"][f"{task}/{split}.jsonl"]
        if hashlib.sha256(payload).hexdigest() != recorded["sha256"]:
            raise ValueError(f"Prepared data hash changed: {path}")
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != expected or any(row["role"] != f"source_{split}" for row in rows):
            raise ValueError(f"Unexpected example count or source role: {path}")
print("INPUT DATA CHECK PASSED: 7035 source examples", flush=True)
PY_PREFLIGHT

# All six caches share the tested encoder precision, batch size, and truncation policy.
for task in piqa cqa; do
    for split in train val cal; do
        printf '\nEXTRACTING %s/%s\n' "$task" "$split"
        python -u extract_features.py \
            --fengshen-root ../Fengshenbang-LM-main \
            --model ../models/unimc-albert-235m \
            --input "${data_dir}/${task}/${split}.jsonl" \
            --output "${cache_dir}/${task}_${split}.pt" \
            --batch-size 2 \
            --max-length 512 \
            --amp none \
            --verify-official \
            --allow-option-truncation \
            --device cuda:0
    done
done

# Validate signatures, roles, counts, official checks, and all split intersections.
python -u - "$cache_dir" "$data_dir" "$run_dir" <<'PY_CACHES'
import hashlib
import json
from pathlib import Path
import sys

from data_io import FeatureDataset, assert_disjoint

cache_dir, data_dir, run_dir = map(Path, sys.argv[1:])
datasets = {}
provenance = json.loads((data_dir / "provenance.json").read_text(encoding="utf-8"))
summary = []
signature = None
for split in ("train", "val", "cal"):
    paths = [cache_dir / f"{task}_{split}.pt" for task in ("piqa", "cqa")]
    datasets[split] = FeatureDataset(paths, required_role=f"source_{split}")
    expected_n = sum(provenance["files"][f"{task}/{split}.jsonl"]["n"] for task in ("piqa", "cqa"))
    if len(datasets[split]) != expected_n:
        raise ValueError(f"Actual cache sample count differs from the prepared data: {split}")
    if signature is None:
        signature = datasets[split].signature
    if signature != datasets[split].signature:
        raise ValueError("Extraction signatures differ across splits")
    for task, metadata in zip(("piqa", "cqa"), datasets[split].metadata):
        expected = provenance["files"][f"{task}/{split}.jsonl"]
        if metadata["n"] != expected["n"] or metadata["input_sha256"] != expected["sha256"]:
            raise ValueError(f"Cache/input provenance mismatch: {task}/{split}")
        if not metadata["official_prediction_check"]["passed"] or metadata["limit"] is not None:
            raise ValueError(f"Incomplete or unverified cache: {task}/{split}")
        summary.append({"task": task, "split": split, "n": metadata["n"],
                        "truncated_option_examples": metadata["truncated_option_examples"]})
assert_disjoint(datasets["train"], datasets["val"])
assert_disjoint(datasets["train"], datasets["cal"])
assert_disjoint(datasets["val"], datasets["cal"])
code_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(Path(".").glob("*.py"))}
manifest = {"data_root": str(data_dir), "cache_root": str(cache_dir), "run_root": str(run_dir),
            "feature_signature": signature, "caches": summary, "scripts_sha256": code_hashes,
            "data_provenance_sha256": hashlib.sha256((data_dir / "provenance.json").read_bytes()).hexdigest(),
            "selection": "Source validation macro accuracy; tie-break macro NLL.",
            "calibration": "Calibration labels are not used for model selection.",
            "old_backbone_exposure": provenance["old_backbone_exposure"]}
(run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
print("PILOT FEATURES COMPLETE", flush=True)
PY_CACHES

# Export baseline logits for source validation and the separate calibration pool.
for split in val cal; do
    python -u predict_head.py \
        --cache "${cache_dir}/piqa_${split}.pt" "${cache_dir}/cqa_${split}.pt" \
        --baseline \
        --output "${prediction_dir}/baseline_${split}.jsonl" \
        --threads 4 \
        --device cuda:0
done

# Same source caches and hyperparameters; each variant is trained independently.
for mode in mlp self full; do
    printf '\nTRAINING %s\n' "$mode"
    python -u train_head.py \
        --train "${cache_dir}/piqa_train.pt" "${cache_dir}/cqa_train.pt" \
        --val "${cache_dir}/piqa_val.pt" "${cache_dir}/cqa_val.pt" \
        --mode "$mode" \
        --output "${run_dir}/${mode}_seed42_lr1e4" \
        --lr 0.0001 \
        --epochs 10 \
        --patience 2 \
        --batch-size 128 \
        --sampling balanced \
        --seed 42 \
        --threads 4 \
        --device cuda:0

    for split in val cal; do
        python -u predict_head.py \
            --cache "${cache_dir}/piqa_${split}.pt" "${cache_dir}/cqa_${split}.pt" \
            --checkpoint "${run_dir}/${mode}_seed42_lr1e4/best.pt" \
            --output "${prediction_dir}/${mode}_${split}.jsonl" \
            --threads 4 \
            --device cuda:0
    done
done

# Summarize only source validation performance; calibration files are separate.
python -u - "$run_dir" "$prediction_dir" "$SECONDS" <<'PY_REPORT'
import csv
import json
from pathlib import Path
import sys

run_dir, prediction_dir = map(Path, sys.argv[1:3])

def read_predictions(mode):
    rows = [json.loads(line) for line in (prediction_dir / f"{mode}_val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {(row["dataset"], row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate prediction IDs: {mode}")
    return result

baseline = read_predictions("baseline")
baseline_metrics = json.loads((prediction_dir / "baseline_val.jsonl.metrics.json").read_text())
records = []
for mode in ("baseline", "mlp", "self", "full"):
    metrics = json.loads((prediction_dir / f"{mode}_val.jsonl.metrics.json").read_text())
    predictions = read_predictions(mode)
    if predictions.keys() != baseline.keys():
        raise ValueError(f"Different evaluation rows: {mode}")
    corrected, harmed = 0, 0
    for key, original in baseline.items():
        candidate = predictions[key]
        if (original["choices"], original["gold_label"]) != (candidate["choices"], candidate["gold_label"]):
            raise ValueError(f"Different evaluation choices or gold label: {mode}/{key}")
        old_correct = original["prediction"] == original["gold_label"]
        new_correct = candidate["prediction"] == candidate["gold_label"]
        corrected += int(new_correct and not old_correct)
        harmed += int(old_correct and not new_correct)
    epoch = 0
    if mode != "baseline":
        training = json.loads((run_dir / f"{mode}_seed42_lr1e4" / "summary.json").read_text())
        epoch = training["best_epoch"]
        if training["best_source_validation"]["macro_accuracy"] != metrics["macro_accuracy"]:
            raise ValueError(f"Best checkpoint/exported validation accuracy mismatch: {mode}")
    records.append({
        "mode": mode, "best_epoch": epoch,
        "piqa_accuracy": metrics["per_dataset"]["piqa"]["accuracy"],
        "cqa_accuracy": metrics["per_dataset"]["commonsenseqa"]["accuracy"],
        "macro_accuracy": metrics["macro_accuracy"], "macro_nll": metrics["macro_nll"],
        "delta_macro_pp": 100 * (metrics["macro_accuracy"] - baseline_metrics["macro_accuracy"]),
        "corrected_vs_baseline": corrected, "harmed_vs_baseline": harmed,
    })
with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
report = {"scope": "Source validation pilot; not target zero-shot results", "seed": 42,
          "lr": 0.0001, "elapsed_seconds": int(sys.argv[3]),
          "prediction_root": str(prediction_dir), "comparison": records}
(run_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("SOURCE VALIDATION COMPARISON", flush=True)
for row in records:
    print(f"{row['mode']:8s} PIQA={100*row['piqa_accuracy']:.2f}% "
          f"CQA={100*row['cqa_accuracy']:.2f}% "
          f"Macro={100*row['macro_accuracy']:.2f}% "
          f"Delta={row['delta_macro_pp']:+.2f}pp "
          f"Corrected={row['corrected_vs_baseline']} Harmed={row['harmed_vs_baseline']}", flush=True)
print(f"RESULTS_DIR={run_dir}", flush=True)
print(f"CALIBRATION_LOGITS_DIR={prediction_dir}", flush=True)
print("PILOT EXPERIMENT COMPLETE", flush=True)
PY_REPORT
