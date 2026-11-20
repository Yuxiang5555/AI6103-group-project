"""Add exactly three interaction-gate trainings to the fixed centered-64 experiment."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import time
import zipfile

import numpy as np
import torch

from data_io import FeatureDataset, assert_disjoint, collate_features, evaluate, file_sha256, select_device, write_json
from decision_head import HeadConfig, ResidualOptionHead
from interaction_gate_head import InteractionGatedOptionHead, MODEL_KIND, definition_hashes
from predict_interaction_gate import export_predictions
import train_interaction_gate

SEEDS = (42, 43, 44)
MODES = ("mlp", "self", "full", "gated")
FIXED = {"model_dim": 64, "heads": 4, "ffn_dim": 256, "dropout": 0.1, "lr": 1e-4,
         "weight_decay": 0.01, "epochs": 10, "patience": 2, "batch_size": 128,
         "sampling": "balanced", "gate": False, "threads": 4}
NEW_FILES = ("interaction_gate_head.py", "train_interaction_gate.py", "predict_interaction_gate.py", "run_interaction_gate.py")
LEGACY_FILES = ("decision_head.py", "data_io.py", "train_head.py", "predict_head.py")
INFERENCE_FILES = ("extract_features.py", "project_mlm_features.py", "center_option_features.py", "prepare_data.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_predictions(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed = {(row["dataset"], str(row["id"])): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    return indexed


def preflight(report_path, cache):
    report = read_json(report_path)
    original = read_json("runs/pilot_55518/experiment_manifest.json")
    for name in LEGACY_FILES:
        if file_sha256(name) != original["scripts_sha256"][name]:
            raise ValueError(f"Legacy core code changed: {name}")
    train_paths = [str(cache / f"{task}_train.pt") for task in ("piqa", "cqa")]
    val_paths = [str(cache / f"{task}_val.pt") for task in ("piqa", "cqa")]
    train, val = FeatureDataset(train_paths, "source_train"), FeatureDataset(val_paths, "source_val")
    if len(train) != 4000 or len(val) != 1517 or train.signature != val.signature:
        raise ValueError("Expected the fixed 4000/1517 source experiment")
    if train.signature.get("synthetic") or train.signature["hidden_dim"] != 128 or "option_centering" not in train.signature:
        raise ValueError("Expected real centered 128-dimensional MLM caches")
    if train.signature["option_centering"]["converter_sha256"] != file_sha256("center_option_features.py"):
        raise ValueError("Centering implementation differs from the reference caches")
    if train.signature["option_representation"]["converter_sha256"] != file_sha256("project_mlm_features.py"):
        raise ValueError("MLM implementation differs from the reference caches")
    assert_disjoint(train, val)
    hashes = {path: file_sha256(path) for path in train_paths + val_paths}
    references = {}
    for mode in ("mlp", "self", "full"):
        for seed in SEEDS:
            selected = [row for row in report["seed_results"] if
                        (row["variant"], row["mode"], row["seed"]) == ("centered", mode, seed)]
            if len(selected) != 1:
                raise ValueError("Missing/duplicate centered reference run")
            row = selected[0]
            folder = Path(row["run_dir"])
            if not folder.resolve().is_relative_to(Path("runs").resolve()):
                raise ValueError("Reference is outside runs/")
            config, summary = read_json(folder / "config.json"), read_json(folder / "summary.json")
            if config["arguments"]["seed"] != seed or config["arguments"]["mode"] != mode:
                raise ValueError("Reference seed/mode differs")
            if any(config["arguments"][key] != value for key, value in FIXED.items()):
                raise ValueError("Reference training settings differ")
            if config["data_sha256"] != hashes or config["feature_signature"] != train.signature:
                raise ValueError("Reference source caches differ")
            if config["head_config"] != HeadConfig(128, 64, 4, 256, 0.1, mode, False).to_dict():
                raise ValueError("Reference head architecture differs")
            if summary.get("synthetic_only") or summary["best_source_validation"]["macro_accuracy"] != row["macro_accuracy"]:
                raise ValueError("Reference source result differs")
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            if (checkpoint["seed"] != seed or checkpoint["head_config"] != config["head_config"] or
                    checkpoint["feature_signature"] != train.signature or checkpoint["source_validation"] != summary["best_source_validation"]):
                raise ValueError("Reference checkpoint differs from its records")
            if mode == "full" and not (folder / "initial.pt").is_file():
                raise FileNotFoundError(folder / "initial.pt")
            references[(mode, seed)] = folder
    return report, train, val, train_paths, val_paths, hashes, references


def endpoint_probe(data, device):
    # Use nonzero scoring weights: all-zero scores would make an endpoint check trivial.
    examples = []
    for task in ("piqa", "commonsenseqa"):
        examples.extend([row for row in data.samples if row["dataset"] == task][:4])
    if len(examples) != 8:
        raise ValueError("Need both source tasks for endpoint verification")
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collate_features(examples).items()}
    config = HeadConfig(hidden_dim=128, model_dim=64, heads=4, feedforward_dim=256, dropout=0.1)
    gated = InteractionGatedOptionHead(config).to(device).eval()
    with torch.no_grad():
        gated.scorer.weight.copy_(torch.linspace(-0.2, 0.2, 64, device=device)[None])
    common = {key: value for key, value in gated.state_dict().items() if key != "interaction_logit"}
    errors = {}
    with torch.inference_mode():
        for mode, value in (("self", 0.0), ("full", 1.0)):
            parent = ResidualOptionHead(HeadConfig(**{**config.to_dict(), "mode": mode})).to(device).eval()
            parent.load_state_dict(common, strict=True)
            expected = parent(batch["hidden"], batch["base_logits"], batch["valid"])
            actual = gated(batch["hidden"], batch["base_logits"], batch["valid"], gate_override=value)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            errors[mode] = float((actual[batch["valid"]]-expected[batch["valid"]]).abs().max().cpu())
    return {"nonzero_scorer": True, "n": len(examples), "max_endpoint_logit_error": errors}


def training_args(seed, folder, reference, train, val, device):
    return train_interaction_gate.parser().parse_args([
        "--train", *train, "--val", *val, "--output", str(folder), "--reference-dir", str(reference),
        "--mode", "full", "--model-dim", "64", "--heads", "4", "--ffn-dim", "256",
        "--dropout", "0.1", "--lr", "0.0001", "--weight-decay", "0.01", "--epochs", "10",
        "--patience", "2", "--batch-size", "128", "--sampling", "balanced", "--threads", "4",
        "--seed", str(seed), "--gate-init", "0.5", "--device", str(device)])


def verify_completed(folder, seed, signature, hashes, require_real=True):
    config, summary = read_json(folder / "config.json"), read_json(folder / "summary.json")
    if config["model_kind"] != MODEL_KIND or config["arguments"]["seed"] != seed:
        raise ValueError("Wrong completed interaction-gate run")
    if any(config["arguments"][key] != value for key, value in FIXED.items()):
        raise ValueError("Completed gate settings differ")
    if config["gate_init"] != 0.5 or config["definition_sha256"] != definition_hashes():
        raise ValueError("Gate initialization/implementation differs")
    if config["data_sha256"] != hashes or config["feature_signature"] != signature:
        raise ValueError("Completed gate data differs")
    if (require_real and summary.get("synthetic_only")) or not summary["paired_initialization"]["shared_initial_weights_equal"]:
        raise ValueError("Expected paired real-data gate training")
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    if (checkpoint["seed"] != seed or checkpoint["epoch"] != summary["best_epoch"] or
            checkpoint["source_validation"] != summary["best_source_validation"]):
        raise ValueError("Completed gate checkpoint/summary differs")
    return summary


def paired_diagnostics(previous, current):
    if previous.keys() != current.keys():
        raise ValueError("Paired prediction IDs differ")
    corrected = harmed = changed = 0
    tasks = {}
    for key, old in previous.items():
        row = current[key]
        if any(old[field] != row[field] for field in ("choices", "gold_label", "base_logits")):
            raise ValueError("Paired labels/options/base logits differ")
        old_ok, new_ok = old["prediction"] == old["gold_label"], row["prediction"] == row["gold_label"]
        corrected += int(new_ok and not old_ok)
        harmed += int(old_ok and not new_ok)
        changed += int(old["prediction"] != row["prediction"])
        scores = sorted(old["base_logits"], reverse=True)
        delta = np.asarray(row["logits"])-np.asarray(row["base_logits"])
        group = tasks.setdefault(key[0], {"base_top2_gaps": [], "delta_spans": []})
        group["base_top2_gaps"].append(scores[0]-scores[1])
        group["delta_spans"].append(float(np.ptp(delta)))
    return {"corrected": corrected, "harmed": harmed, "changed_predictions": changed,
            "per_task_score_scale": {task: {"median_base_top2_gap": float(np.median(v["base_top2_gaps"])),
                                            "median_correction_span": float(np.median(v["delta_spans"]))}
                                     for task, v in tasks.items()}}


def summarize(records, predictions, baseline, draws=20000):
    lookup = {(row["mode"], row["seed"]): row for row in records}
    if len(lookup) != len(records) or set(lookup) != {(mode, seed) for mode in MODES for seed in SEEDS}:
        raise ValueError("Need all four heads and three paired seeds")
    groups = []
    for mode in MODES:
        selected = [lookup[(mode, seed)] for seed in SEEDS]
        values = [row["macro_accuracy"] for row in selected]
        groups.append({"mode": mode, "mean_macro_accuracy": statistics.mean(values),
                       "sd_macro_accuracy": statistics.stdev(values),
                       "mean_macro_nll": statistics.mean(row["macro_nll"] for row in selected),
                       "delta_vs_baseline_pp": 100*(statistics.mean(values)-baseline["macro_accuracy"])})
    contrasts = []
    keys = sorted(predictions[("gated", 42)])
    task_positions = [np.array([i for i, key in enumerate(keys) if key[0] == task]) for task in sorted({k[0] for k in keys})]
    rng = np.random.default_rng(20261118)
    for control in ("full", "self", "mlp"):
        delta = [100*(lookup[("gated", seed)]["macro_accuracy"]-lookup[(control, seed)]["macro_accuracy"]) for seed in SEEDS]
        per_question = np.mean([[int(predictions[("gated", seed)][key]["prediction"] == predictions[("gated", seed)][key]["gold_label"])
                                 - int(predictions[(control, seed)][key]["prediction"] == predictions[(control, seed)][key]["gold_label"])
                                 for key in keys] for seed in SEEDS], axis=0)
        boot = np.zeros(draws)
        for start in range(0, draws, 500):
            stop = min(start+500, draws)
            for positions in task_positions:
                sample = rng.integers(0, len(positions), size=(stop-start, len(positions)))
                boot[start:stop] += per_question[positions][sample].mean(1) / len(task_positions)
        contrasts.append({"comparison": f"gated minus {control}", "delta_by_seed_pp": dict(zip(map(str, SEEDS), delta)),
                          "mean_delta_pp": statistics.mean(delta), "sd_delta_pp": statistics.stdev(delta),
                          "positive_seeds": sum(v > 0 for v in delta),
                          "exploratory_question_bootstrap_ci95_pp": (100*np.quantile(boot, [0.025, 0.975])).tolist()})
    return {"groups": groups, "paired_gate_comparisons": contrasts,
            "bootstrap": {"draws": draws, "seed": 20261118, "method": "Task-stratified paired questions; three seeds carried together; fixed models"}}


def package_results(output, prediction_dir, folders):
    archive = Path.cwd().parent / "UniMC_interaction_gate_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in output.iterdir():
            if path.is_file():
                bundle.write(path, path.name)
        for path in prediction_dir.iterdir():
            if path.is_file():
                bundle.write(path, "predictions/"+path.name)
        for (mode, seed), folder in folders.items():
            bundle.write(folder / "best.pt", f"models/{mode}_seed{seed}.pt")
            for name in ("config.json", "history.json", "summary.json"):
                bundle.write(folder / name, f"training_records/{mode}/seed{seed}/{name}")
        for name in NEW_FILES + LEGACY_FILES + INFERENCE_FILES:
            bundle.write(name, "code/"+name)
        bundle.write("README_交互Gate实施.md", "README.md")
        bundle.write("tests/test_interaction_gate.py", "code/tests/test_interaction_gate.py")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip():
            raise ValueError("Interaction-gate result archive is corrupted")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="cache/centered_mlm_55752")
    parser.add_argument("--reference-report", default="runs/centering_confirm_55757/comparison.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output, cache, device = Path(args.output), Path(args.cache), select_device(args.device)
    if not args.report_only and output.exists():
        raise FileExistsError(output)
    report, train, val, train_paths, val_paths, hashes, references = preflight(args.reference_report, cache)
    implementations = {name: file_sha256(name) for name in NEW_FILES + LEGACY_FILES}
    reference_hashes = {f"{mode}_seed{seed}": {
        "best_sha256": file_sha256(folder / "best.pt"), "config_sha256": file_sha256(folder / "config.json"),
        "initial_sha256": file_sha256(folder / "initial.pt") if mode == "full" else None}
        for (mode, seed), folder in references.items()}
    if args.report_only:
        previous = read_json(output / "experiment_manifest.json")
        if previous["code_sha256"] != implementations or previous["source_cache_sha256"] != hashes:
            raise ValueError("Report-only recovery code/data differs from the completed experiment")
        if (previous["reference_report_sha256"] != file_sha256(args.reference_report) or
                previous["reference_checkpoint_sha256"] != reference_hashes):
            raise ValueError("Report-only recovery reference models differ")
        print("REPORT ONLY: reusing all three completed interaction-gate trainings", flush=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "experiment_manifest.json", {
            "model_kind": MODEL_KIND, "seeds": SEEDS, "fixed_settings": FIXED, "gate_init": 0.5,
            "code_sha256": implementations, "source_cache_sha256": hashes,
            "reference_report": args.reference_report, "reference_report_sha256": file_sha256(args.reference_report),
            "reference_checkpoint_sha256": reference_hashes,
            "source_cal_used": False, "target_data_used": False,
        })
    probe = endpoint_probe(val, device)
    write_json(output / "endpoint_check.json", probe)
    print("NONZERO-SCORER GATE ENDPOINT CHECK PASSED", flush=True)
    folders = dict(references)
    for seed in SEEDS:
        folder = output / f"gated_seed{seed}"
        if not args.report_only:
            print(f"TRAINING interaction_gate seed={seed} gate_init=0.5", flush=True)
            train_interaction_gate.run(training_args(seed, folder, references[("full", seed)], train_paths, val_paths, device))
        verify_completed(folder, seed, train.signature, hashes)
        folders[("gated", seed)] = folder
    tag = f"predictions_resume_{os.environ.get('SLURM_JOB_ID') or time.time_ns()}" if args.report_only else "predictions"
    prediction_dir = output / tag
    prediction_dir.mkdir(exist_ok=False)
    base_path = prediction_dir / "baseline_val.jsonl"
    baseline = export_predictions(val, base_path, device)
    if abs(baseline["macro_accuracy"]-report["baseline"]["macro_accuracy"]) > 1e-12:
        raise ValueError("Reference baseline accuracy differs")
    base_rows = read_predictions(base_path)
    predictions, records = {}, []
    for mode in MODES:
        for seed in SEEDS:
            folder = folders[(mode, seed)]
            target = prediction_dir / f"{mode}_seed{seed}_val.jsonl"
            metrics = export_predictions(val, target, device, folder / "best.pt")
            current = read_predictions(target)
            diagnostics = paired_diagnostics(base_rows, current)
            summary = read_json(folder / "summary.json")
            if metrics["macro_accuracy"] != summary["best_source_validation"]["macro_accuracy"]:
                raise ValueError("Exported accuracy differs from trained checkpoint")
            records.append({"mode": mode, "seed": seed, "run_dir": str(folder), "best_epoch": summary["best_epoch"],
                            "parameters": summary["parameter_count"], "gate_value": summary.get("best_gate_value"),
                            "macro_accuracy": metrics["macro_accuracy"], "macro_nll": metrics["macro_nll"],
                            "piqa_accuracy": metrics["per_dataset"]["piqa"]["accuracy"],
                            "cqa_accuracy": metrics["per_dataset"]["commonsenseqa"]["accuracy"], **diagnostics})
            predictions[(mode, seed)] = current
    result = summarize(records, predictions, baseline)
    write_json(output / "comparison.json", {
        "scope": "Exploratory source-development comparison; not independent target evidence",
        "source_cal_used": False, "target_data_used": False, "baseline": baseline,
        "seed_results": records, **result, "prediction_dir": str(prediction_dir),
        "gate_semantics": "One global learned scalar interpolates shared-weight self/full attention; not a per-example confidence or rejection gate",
        "centering_semantics": "All compared heads use the same centered features, which already contain set information",
    })
    fields = [key for key in records[0] if key != "per_task_score_scale"]
    with (output / "seed_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Interaction-gate source-development experiment", "",
             "Three new gated trainings; fixed centered-64 controls reused. No source calibration or target selection.", "",
             "| Head | Macro Accuracy (%) mean ± seed SD | Change vs baseline (pp) | Mean NLL |",
             "|---|---:|---:|---:|", f"| baseline | {100*baseline['macro_accuracy']:.4f} | — | {baseline['macro_nll']:.6f} |"]
    for group in result["groups"]:
        print(f"GROUP {group['mode']} Macro={100*group['mean_macro_accuracy']:.4f}% +/- {100*group['sd_macro_accuracy']:.4f}pp Delta={group['delta_vs_baseline_pp']:+.4f}pp", flush=True)
        lines.append(f"| {group['mode']} | {100*group['mean_macro_accuracy']:.4f} ± {100*group['sd_macro_accuracy']:.4f} | {group['delta_vs_baseline_pp']:+.4f} | {group['mean_macro_nll']:.6f} |")
    lines.extend(["", "Paired changes (percentage points), fixed models; exploratory intervals after adaptive source-development work:", ""])
    for contrast in result["paired_gate_comparisons"]:
        lo, hi = contrast["exploratory_question_bootstrap_ci95_pp"]
        line = f"{contrast['comparison']}: {contrast['mean_delta_pp']:+.4f} ± {contrast['sd_delta_pp']:.4f}pp; paired question interval [{lo:+.4f}, {hi:+.4f}]pp"
        print("PAIRED " + line, flush=True)
        lines.append("- " + line)
    lines.extend(["", "Gate values at selected checkpoints:", ""])
    for record in records:
        if record["mode"] == "gated":
            line = f"seed={record['seed']} g={record['gate_value']:.6f} epoch={record['best_epoch']} corrected={record['corrected']} harmed={record['harmed']}"
            print("GATE " + line, flush=True)
            lines.append("- " + line)
    lines.extend(["", "Gate size alone does not demonstrate performance gains. Centered self/MLP already use set information. Independent target evaluation is required."])
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    archive = package_results(output, prediction_dir, folders)
    print(f"RESULTS_DIR={output}", flush=True)
    print(f"REPORT_ARCHIVE={archive}", flush=True)
    print("INTERACTION GATE EXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
