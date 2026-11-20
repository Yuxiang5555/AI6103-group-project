"""Package frozen model-group candidates and export held-out-source calibration logits.

No optimizer, calibration fit, target evaluation, or new model selection is run.
The primary candidate is centered full, fixed before reading calibration scores.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

RUNTIME_FILES = ("decision_head.py", "data_io.py", "train_head.py", "predict_head.py",
                 "prepare_data.py", "extract_features.py", "project_mlm_features.py", "center_option_features.py")


def python_stage(arguments, log_file):
    environment = os.environ.copy()
    if sys.platform.startswith("linux"):
        environment["MKL_THREADING_LAYER"] = "GNU"
    with Path(log_file).open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, *map(str, arguments)], check=True, env=environment, stdout=stream)


def validate_source_splits(data_root):
    pools = {split: set() for split in ("train", "val", "cal")}
    counts = {}
    for task in ("piqa", "cqa"):
        for split in pools:
            path = data_root / task / f"{split}.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows or any(row["role"] != f"source_{split}" for row in rows):
                raise ValueError(f"Incorrect prepared data role: {path}")
            hashes = {row["content_hash"] for row in rows}
            if len(hashes) != len(rows):
                raise ValueError(f"Repeated question contents in {path}")
            pools[split].update(hashes)
            counts[f"{task}/{split}"] = len(rows)
    if any(pools[a] & pools[b] for a, b in (("train", "val"), ("train", "cal"), ("val", "cal"))):
        raise ValueError("Source train/validation/calibration contents overlap")
    if counts != {"piqa/train": 2000, "piqa/val": 919, "piqa/cal": 919,
                  "cqa/train": 2000, "cqa/val": 598, "cqa/cal": 599}:
        raise ValueError("Prepared source splits differ from the confirmed pilot")
    return counts


def audit_calibration_exports(folder, seeds=(42, 43, 44)):
    def load(name):
        rows = [json.loads(line) for line in (folder / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        result = {(row["dataset"], row["id"]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate calibration prediction IDs")
        return result
    baseline = load("baseline.jsonl")
    if len(baseline) != 1518:
        raise ValueError("Expected 1518 source-calibration examples")
    names = ["baseline.jsonl", *[f"centered_full_seed{seed}.jsonl" for seed in seeds]]
    for name in names:
        predicted = load(name)
        if predicted.keys() != baseline.keys():
            raise ValueError("Calibration model ID sets differ")
        for key, row in predicted.items():
            if any(row[field] != baseline[key][field] for field in ("choices", "gold_label", "base_logits")):
                raise ValueError("Calibration choices/labels/base scores differ")
        metadata = json.loads((folder / f"{name}.metadata.json").read_text(encoding="utf-8"))
        if not metadata["uncalibrated"] or metadata["cache_roles"] != ["source_cal", "source_cal"]:
            raise ValueError("Expected raw source-calibration logits")
    return {"n": len(baseline), "matched_model_views": len(names), "uncalibrated": True}


def validate_calibration_cache(data, data_root):
    expected = {}
    for task in ("piqa", "cqa"):
        for line in (data_root / task / "cal.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = (row["dataset"], str(row["id"]))
                if key in expected:
                    raise ValueError("Duplicate prepared calibration IDs")
                expected[key] = row
    observed = {(row["dataset"], str(row["id"])): row for row in data.samples}
    if observed.keys() != expected.keys():
        raise ValueError("Calibration cache/prepared data IDs differ")
    for key, row in observed.items():
        if any(row[field] != expected[key][field] for field in ("content_hash", "choices", "gold_label")):
            raise ValueError("Calibration cache/prepared contents differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New handoff directory")
    parser.add_argument("--report", default="runs/centering_confirm_55757/comparison.json")
    parser.add_argument("--model", default="../models/unimc-albert-235m")
    parser.add_argument("--fengshen-root", default="../Fengshenbang-LM-main")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    from data_io import FeatureDataset, file_sha256, write_json
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = report["seed_results"]
    expected = {(variant, mode, seed) for variant in ("uncentered", "centered")
                for mode in ("mlp", "self", "full") for seed in (42, 43, 44)}
    keys = [(row["variant"], row["mode"], row["seed"]) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Expected the complete paired three-seed source report")
    counts = validate_source_splits(Path("data/pilot_seed42_v2"))
    cal = [Path("cache/pilot_55518") / f"{task}_cal.pt" for task in ("piqa", "cqa")]
    cal_data = FeatureDataset(cal, "source_cal")
    if len(cal_data) != 1518:
        raise ValueError("Incorrect source-calibration cache size")
    validate_calibration_cache(cal_data, Path("data/pilot_seed42_v2"))
    output.mkdir(parents=True, exist_ok=False)
    for name in ("models", "calibration", "provenance", "work"):
        (output / name).mkdir()
    for name in RUNTIME_FILES:
        shutil.copyfile(name, output / name)
    shutil.copyfile("README_模型优化组交接.md", output / "README.md")
    shutil.copytree("data/pilot_seed42_v2", output / "data/pilot_seed42_v2")
    if Path("handoff_reference").is_dir():
        shutil.copytree("handoff_reference", output / "source_results")
    else:
        (output / "source_results").mkdir()
    shutil.copyfile(args.report, output / "source_results/three_seed_confirmation.json")
    shutil.copytree(report["prediction_dir"], output / "source_results/validation_predictions")
    models = []
    primary = {}
    for row in rows:
        source = Path(row["run_dir"])
        if not source.resolve().is_relative_to(Path("runs").resolve()):
            raise ValueError("Checkpoint source is outside the experiment runs directory")
        checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
        if checkpoint["seed"] != row["seed"] or checkpoint["head_config"]["mode"] != row["mode"]:
            raise ValueError("Report/checkpoint seed or mode mismatch")
        if checkpoint["source_validation"]["macro_accuracy"] != row["macro_accuracy"]:
            raise ValueError("Checkpoint/report validation accuracy differs")
        if ("option_centering" in checkpoint["feature_signature"]) != (row["variant"] == "centered"):
            raise ValueError("Unexpected checkpoint representation")
        filename = f"{row['variant']}_{row['mode']}_seed{row['seed']}.pt"
        target = output / "models" / filename
        shutil.copyfile(source / "best.pt", target)
        provenance = output / "provenance" / row["variant"] / row["mode"] / f"seed{row['seed']}"
        provenance.mkdir(parents=True)
        for name in ("config.json", "history.json", "summary.json"):
            shutil.copyfile(source / name, provenance / name)
        models.append({"file": f"models/{filename}", "sha256": file_sha256(target),
                       "variant": row["variant"], "mode": row["mode"], "seed": row["seed"]})
        if row["variant"] == "centered" and row["mode"] == "full":
            primary[row["seed"]] = target
    signature = torch.load(primary[42], map_location="cpu", weights_only=True)["feature_signature"]
    mlm_signature = signature["option_centering"]["parent_feature_signature"]
    raw_signature = mlm_signature["option_representation"]["parent_feature_signature"]
    if raw_signature != cal_data.signature:
        raise ValueError("Calibration cache and frozen model signatures differ")
    if file_sha256("project_mlm_features.py") != mlm_signature["option_representation"]["converter_sha256"]:
        raise ValueError("MLM projection code differs from the frozen checkpoint")
    if file_sha256("center_option_features.py") != signature["option_centering"]["converter_sha256"]:
        raise ValueError("Centering code differs from the frozen checkpoint")
    del cal_data
    print("HANDOFF CHECKPOINTS AND SOURCE SPLITS VERIFIED", flush=True)
    work = output / "work"
    python_stage(["project_mlm_features.py", "--cache", *cal,
                  "--output-dir", work / "mlm_cal", "--model", args.model,
                  "--fengshen-root", args.fengshen_root, "--device", args.device], work / "project.log")
    projected = [work / "mlm_cal" / path.name for path in cal]
    python_stage(["center_option_features.py", "--cache", *projected,
                  "--output-dir", work / "centered_cal"], work / "center.log")
    centered = [work / "centered_cal" / path.name for path in cal]
    python_stage(["predict_head.py", "--cache", *cal, "--baseline", "--device", args.device,
                  "--output", output / "calibration/baseline.jsonl"], work / "baseline.log")
    for seed in (42, 43, 44):
        print(f"EXPORTING SOURCE CALIBRATION full seed={seed}", flush=True)
        python_stage(["predict_head.py", "--cache", *centered, "--checkpoint", primary[seed],
                      "--device", args.device, "--output", output / f"calibration/centered_full_seed{seed}.jsonl"],
                     work / f"seed{seed}.log")
    audit = audit_calibration_exports(output / "calibration")
    revision_file = Path(args.model) / "model_revision.txt"
    versions = {}
    for package in ("transformers", "pytorch-lightning", "torchmetrics", "sentencepiece", "protobuf", "huggingface-hub", "jieba", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(output / "model_manifest.json", {
        "status": "Candidate for independent evaluation; not a confirmed accuracy improvement",
        "primary": {"variant": "centered", "mode": "full", "seeds": [42, 43, 44]},
        "selection_basis": "Fixed after the three-seed source-development comparison; no calibration or target selection",
        "models": models, "base_model_repo": "IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English",
        "base_model_revision": revision_file.read_text().strip() if revision_file.is_file() else None,
        "raw_feature_signature": raw_signature, "mlm_feature_signature": mlm_signature,
        "primary_feature_signature": signature, "source_split_counts": counts,
        "calibration_export_audit": audit, "calibration_fit_performed": False,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda_runtime": torch.version.cuda, **versions},
        "runtime_file_sha256": {name: file_sha256(output / name) for name in RUNTIME_FILES},
    })
    archive = Path.cwd().parent / "UniMC_model_optimization_handoff.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in output.rglob("*"):
            relative = path.relative_to(output)
            if path.is_file() and relative.parts[0] != "work":
                z.write(path, relative.as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise ValueError("Handoff archive integrity check failed")
    print(f"SOURCE_CALIBRATION_EXPORTS n={audit['n']} model_views={audit['matched_model_views']} uncalibrated=True", flush=True)
    print(f"HANDOFF_ARCHIVE={archive}", flush=True)
    print("MODEL HANDOFF COMPLETE", flush=True)


if __name__ == "__main__":
    main()
