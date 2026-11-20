"""Paired source-validation confirmation of option centering, seeds 42/43/44.

Reuse completed seed-42 runs. Train seeds 43/44 for both input variants with all
other settings fixed. Never use calibration/target data or pool seeds as samples.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile

SEEDS = (42, 43, 44)
MODES = ("mlp", "self", "full")
VARIANTS = ("uncentered", "centered")
FIXED = {"model_dim": 64, "heads": 4, "ffn_dim": 256, "dropout": 0.1,
         "lr": 0.0001, "weight_decay": 0.01, "epochs": 10, "patience": 2,
         "batch_size": 128, "sampling": "balanced", "gate": False}


def plan_runs(output, references):
    return [{"seed": seed, "variant": variant, "mode": mode,
             "path": references[variant] / f"{mode}_seed42_lr1e4" if seed == 42 else
                     output / variant / f"{mode}_seed{seed}_lr1e4",
             "train_needed": seed != 42}
            for seed in SEEDS for variant in VARIANTS for mode in MODES]


def training_args(item, cache, device, threads):
    return ["--train", str(cache / "piqa_train.pt"), str(cache / "cqa_train.pt"),
            "--val", str(cache / "piqa_val.pt"), str(cache / "cqa_val.pt"),
            "--output", str(item["path"]), "--mode", item["mode"],
            "--seed", str(item["seed"]), "--model-dim", "64", "--heads", "4",
            "--ffn-dim", "256", "--dropout", "0.1", "--lr", "0.0001",
            "--weight-decay", "0.01", "--epochs", "10", "--patience", "2",
            "--batch-size", "128", "--sampling", "balanced", "--threads", str(threads), "--device", device]


def mean_sd(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values)}


def summarize(records, baseline):
    index = {(row["variant"], row["mode"], row["seed"]): row for row in records}
    expected = {(variant, mode, seed) for variant in VARIANTS for mode in MODES for seed in SEEDS}
    if len(index) != len(records) or set(index) != expected:
        raise ValueError("Need exactly one result per variant/mode/seed")
    groups = []
    for variant in VARIANTS:
        for mode in MODES:
            rows = [index[(variant, mode, seed)] for seed in SEEDS]
            groups.append({"variant": variant, "mode": mode,
                           **{metric: mean_sd([row[metric] for row in rows]) for metric in
                              ("piqa_accuracy", "cqa_accuracy", "macro_accuracy", "macro_nll")},
                           "delta_macro_pp": 100 * (statistics.mean(row["macro_accuracy"] for row in rows)
                                                     - baseline["macro_accuracy"])})
    paired = []
    for mode in MODES:
        deltas = [100 * (index[("centered", mode, seed)]["macro_accuracy"] -
                         index[("uncentered", mode, seed)]["macro_accuracy"]) for seed in SEEDS]
        paired.append({"mode": mode, "delta_by_seed_pp": dict(zip(map(str, SEEDS), deltas)),
                       **mean_sd(deltas), "positive_seeds": sum(value > 0 for value in deltas),
                       "zero_seeds": sum(value == 0 for value in deltas)})
    attention = []
    for control in ("self", "mlp"):
        deltas = [100 * (index[("centered", "full", seed)]["macro_accuracy"] -
                         index[("centered", control, seed)]["macro_accuracy"]) for seed in SEEDS]
        attention.append({"comparison": f"centered full minus {control}",
                          "delta_by_seed_pp": dict(zip(map(str, SEEDS), deltas)), **mean_sd(deltas)})
    return {"groups": groups, "paired_centering_delta_pp": paired, "attention_delta_pp": attention}


def load_predictions(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed = {(row["dataset"], row["id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate prediction IDs: {path}")
    return indexed


def check_initial_pair(a, b):
    import torch
    first = torch.load(a / "initial.pt", map_location="cpu", weights_only=True)
    second = torch.load(b / "initial.pt", map_location="cpu", weights_only=True)
    if first["head_config"] != second["head_config"] or first["state_dict"].keys() != second["state_dict"].keys():
        raise ValueError("Paired model architectures differ")
    if any(not torch.equal(first["state_dict"][key], second["state_dict"][key]) for key in first["state_dict"]):
        raise ValueError("Paired model initializations differ")


def preflight(caches, references):
    import torch
    from data_io import FeatureDataset, assert_disjoint, file_sha256
    original = json.loads(Path("runs/pilot_55518/experiment_manifest.json").read_text(encoding="utf-8"))
    for name in ("decision_head.py", "train_head.py", "data_io.py", "predict_head.py"):
        if file_sha256(name) != original["scripts_sha256"][name]:
            raise ValueError(f"Core code changed since original pilot: {name}")
    datasets = {}
    for variant, cache in caches.items():
        for split, count in (("train", 4000), ("val", 1517)):
            dataset = FeatureDataset([cache / f"{task}_{split}.pt" for task in ("piqa", "cqa")], f"source_{split}")
            if len(dataset) != count:
                raise ValueError(f"Incorrect data count: {variant}/{split}")
            datasets[(variant, split)] = dataset
        assert_disjoint(datasets[(variant, "train")], datasets[(variant, "val")])
        if datasets[(variant, "train")].signature != datasets[(variant, "val")].signature:
            raise ValueError("Train/val feature signatures differ")
        for mode in MODES:
            config = json.loads((references[variant] / f"{mode}_seed42_lr1e4/config.json").read_text(encoding="utf-8"))
            if config["arguments"]["seed"] != 42 or config["arguments"]["mode"] != mode:
                raise ValueError("Incorrect seed-42 reference")
            if any(config["arguments"][key] != value for key, value in FIXED.items()):
                raise ValueError("Reference hyperparameters differ")
            for path, digest in config["data_sha256"].items():
                if file_sha256(path) != digest:
                    raise ValueError(f"Reference cache changed: {path}")
    centered_signature = datasets[("centered", "train")].signature
    if centered_signature["option_centering"]["converter_sha256"] != file_sha256("center_option_features.py"):
        raise ValueError("Centering converter changed")
    for split in ("train", "val"):
        before, after = datasets[("uncentered", split)], datasets[("centered", split)]
        for a, b in zip(before.samples, after.samples):
            if any(a[key] != b[key] for key in ("id", "dataset", "choices", "gold_label", "content_hash")):
                raise ValueError("Paired input rows differ")
            torch.testing.assert_close(a["base_logits"], b["base_logits"], atol=0, rtol=0)
            expected = a["hidden"].float() - a["hidden"].float().mean(0, keepdim=True)
            torch.testing.assert_close(expected, b["hidden"], atol=0, rtol=0)
    for mode in MODES:
        check_initial_pair(references["uncentered"] / f"{mode}_seed42_lr1e4", references["centered"] / f"{mode}_seed42_lr1e4")
    print("PAIRED INPUT AND INITIALIZATION CHECK PASSED", flush=True)


def prediction_environment():
    child_env = os.environ.copy()
    if sys.platform.startswith("linux"):
        child_env["MKL_THREADING_LAYER"] = "GNU"
    return child_env


def export(caches, output, device, threads, checkpoint=None):
    command = [sys.executable, "predict_head.py", "--cache", str(caches / "piqa_val.pt"), str(caches / "cqa_val.pt"),
               "--output", str(output), "--threads", str(threads), "--device", device]
    command += ["--checkpoint", str(checkpoint)] if checkpoint else ["--baseline"]
    # NumPy/MKL can set INTEL in the parent environment; PyTorch's Linux child
    # loads libgomp. Select the matching GNU MKL layer before child imports.
    subprocess.run(command, check=True, env=prediction_environment())


def validate_completed_runs(plan, caches):
    from data_io import file_sha256
    expected_hashes = {variant: {
        str(cache / f"{task}_{split}.pt"): file_sha256(cache / f"{task}_{split}.pt")
        for split in ("train", "val") for task in ("piqa", "cqa")}
        for variant, cache in caches.items()}
    for item in plan:
        folder = item["path"]
        for filename in ("initial.pt", "best.pt", "config.json", "history.json", "summary.json"):
            path = folder / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Incomplete training result; recovery will not retrain it: {path}")
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if config["arguments"]["seed"] != item["seed"] or config["arguments"]["mode"] != item["mode"]:
            raise ValueError(f"Completed run has different mode/seed: {folder}")
        if any(config["arguments"][key] != value for key, value in FIXED.items()):
            raise ValueError(f"Completed run has different hyperparameters: {folder}")
        if config["data_sha256"] != expected_hashes[item["variant"]]:
            raise ValueError(f"Completed run has different training/validation inputs: {folder}")
        if summary.get("synthetic_only") or summary["best_epoch"] < 1:
            raise ValueError(f"Expected completed real-data training: {folder}")


def ensure_training_completed(plan, caches, device, threads, report_only):
    if not report_only:
        import train_head
        for item in plan:
            if item["train_needed"]:
                print(f"TRAINING variant={item['variant']} mode={item['mode']} seed={item['seed']}", flush=True)
                train_head.run(train_head.parser().parse_args(training_args(item, caches[item["variant"]], device, threads)))
    validate_completed_runs(plan, caches)
    print(f"ALL {len(plan)} TRAINING RESULTS VERIFIED", flush=True)


def prediction_directory(output, report_only):
    if report_only:
        tag = os.environ.get("SLURM_JOB_ID") or str(time.time_ns())
        if not tag.isdigit():
            raise ValueError("Expected a numeric recovery run identifier")
        path = output / f"predictions_resume_{tag}"
    else:
        path = output / "predictions"
    path.mkdir(exist_ok=False)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report-only", action="store_true", help="Require completed runs; export/summarize without training")
    args = parser.parse_args()
    from data_io import file_sha256, write_json
    output = Path(args.output)
    if args.threads < 1:
        raise ValueError("Threads must be positive")
    if args.report_only:
        if not output.is_dir() or not (output / "confirmation_manifest.json").is_file():
            raise FileNotFoundError(f"Expected existing confirmation run: {output}")
        print(f"REPORT ONLY: reusing completed checkpoints in {output}", flush=True)
    elif output.exists():
        raise FileExistsError(f"Existing run; use --report-only for export recovery: {output}")
    caches = {"uncentered": Path("cache/mlm_pilot_55518_55555"), "centered": Path("cache/centered_mlm_55752")}
    references = {"uncentered": Path("runs/compact_pilot_55518_55555/mlm"), "centered": Path("runs/centered_mlm_55752")}
    preflight(caches, references)
    if not args.report_only:
        output.mkdir(parents=True, exist_ok=False)
        for variant in VARIANTS:
            (output / variant).mkdir()
    plan = plan_runs(output, references)
    manifest_path = output / "confirmation_manifest.json"
    if args.report_only:
        manifest_path = output / f"report_resume_{os.environ.get('SLURM_JOB_ID') or time.time_ns()}.json"
    write_json(manifest_path, {
        "seeds": SEEDS, "fixed_settings": FIXED, "reuse_seed42": True,
        "source_cal_used": False, "target_data_used": False,
        "report_only": args.report_only,
        "export_mkl_threading_layer": prediction_environment().get("MKL_THREADING_LAYER"),
        "caches": {name: str(path) for name, path in caches.items()},
        "references": {name: str(path) for name, path in references.items()},
        "code_sha256": {name: file_sha256(name) for name in
                        ("confirm_centering.py", "decision_head.py", "train_head.py", "data_io.py", "predict_head.py", "center_option_features.py")},
    })
    ensure_training_completed(plan, caches, args.device, args.threads, args.report_only)
    paths = {(item["variant"], item["mode"], item["seed"]): item["path"] for item in plan}
    for seed in SEEDS:
        for mode in MODES:
            check_initial_pair(paths[("uncentered", mode, seed)], paths[("centered", mode, seed)])
    prediction_dir = prediction_directory(output, args.report_only)
    baseline_file = prediction_dir / "baseline_val.jsonl"
    print("EXPORTING baseline", flush=True)
    export(caches["uncentered"], baseline_file, args.device, args.threads)
    baseline_predictions = load_predictions(baseline_file)
    baseline = json.loads(Path(str(baseline_file) + ".metrics.json").read_text(encoding="utf-8"))
    records = []
    for item in plan:
        name = f"{item['variant']}_{item['mode']}_seed{item['seed']}_val.jsonl"
        prediction = prediction_dir / name
        print(f"EXPORTING {item['variant']} {item['mode']} seed={item['seed']}", flush=True)
        export(caches[item["variant"]], prediction, args.device, args.threads, item["path"] / "best.pt")
        metrics = json.loads(Path(str(prediction) + ".metrics.json").read_text(encoding="utf-8"))
        training = json.loads((item["path"] / "summary.json").read_text(encoding="utf-8"))
        if training.get("synthetic_only") or training["baseline_source_validation"] != baseline:
            raise ValueError("Synthetic or different baseline results")
        if training["best_source_validation"]["macro_accuracy"] != metrics["macro_accuracy"]:
            raise ValueError("Exported checkpoint accuracy differs from training summary")
        predictions = load_predictions(prediction)
        if predictions.keys() != baseline_predictions.keys():
            raise ValueError("Prediction ID sets differ")
        corrected = harmed = 0
        for key, previous in baseline_predictions.items():
            current = predictions[key]
            if any(previous[field] != current[field] for field in ("choices", "gold_label", "base_logits")):
                raise ValueError("Paired choices/labels/base logits differ")
            old_ok = previous["prediction"] == previous["gold_label"]
            new_ok = current["prediction"] == current["gold_label"]
            corrected += int(new_ok and not old_ok)
            harmed += int(old_ok and not new_ok)
        records.append({"variant": item["variant"], "mode": item["mode"], "seed": item["seed"],
                        "best_epoch": training["best_epoch"], "run_dir": str(item["path"]),
                        "piqa_accuracy": metrics["per_dataset"]["piqa"]["accuracy"],
                        "cqa_accuracy": metrics["per_dataset"]["commonsenseqa"]["accuracy"],
                        "macro_accuracy": metrics["macro_accuracy"], "macro_nll": metrics["macro_nll"],
                        "corrected_vs_baseline": corrected, "harmed_vs_baseline": harmed})
    with (output / "seed_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records, baseline)
    write_json(output / "comparison.json", {
        "scope": "Paired source validation, fixed seeds; SD describes seed variation, not a generalization confidence interval",
        "seeds": SEEDS, "baseline": baseline, "seed_results": records, **summary,
        "prediction_dir": str(prediction_dir), "report_only_recovery": args.report_only,
        "centering_semantics": "Centered self/MLP use set information; self blocks cross-option attention only.",
    })
    print("THREE SEED PAIRED COMPARISON", flush=True)
    print(f"BASELINE Macro={100*baseline['macro_accuracy']:.2f}% NLL={baseline['macro_nll']:.4f}", flush=True)
    for row in summary["groups"]:
        metric = row["macro_accuracy"]
        print(f"GROUP {row['variant']:10s} {row['mode']:5s} Macro={100*metric['mean']:.2f}% +/- {100*metric['sd']:.2f}pp "
              f"PIQA={100*row['piqa_accuracy']['mean']:.2f}% CQA={100*row['cqa_accuracy']['mean']:.2f}% "
              f"Delta={row['delta_macro_pp']:+.2f}pp NLL={row['macro_nll']['mean']:.4f}", flush=True)
    for row in summary["paired_centering_delta_pp"]:
        print(f"PAIRED {row['mode']:5s} center-minus-control={row['mean']:+.2f} +/- {row['sd']:.2f}pp "
              f"positive_seeds={row['positive_seeds']}/3", flush=True)
    for row in summary["attention_delta_pp"]:
        print(f"ATTENTION {row['comparison']}={row['mean']:+.2f} +/- {row['sd']:.2f}pp", flush=True)
    archive = Path.cwd().parent / "UniMC_centering_confirmation.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in list(output.iterdir()) + list(prediction_dir.iterdir()):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".csv"):
                z.write(path, path.relative_to(output).as_posix())
        for item in plan:
            for filename in ("config.json", "history.json", "summary.json"):
                z.write(item["path"] / filename, f"training_records/{item['variant']}/{item['mode']}/seed{item['seed']}/{filename}")
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise ValueError("Confirmation archive integrity check failed")
    print(f"RESULTS_DIR={output}", flush=True)
    print(f"REPORT_ARCHIVE={archive}", flush=True)
    print("CENTERING CONFIRMATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
