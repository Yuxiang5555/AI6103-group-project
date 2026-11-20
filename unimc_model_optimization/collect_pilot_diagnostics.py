"""Collect the completed source pilots for local diagnosis; no model forward/training.

Includes source-validation caches, selected small-head/full-head checkpoints,
training histories, and 64 training examples per task/representation as caches.
The full ALBERT checkpoint and calibration/target datasets are not collected.
"""
import argparse
import copy
import io
import json
from pathlib import Path
import zipfile


def collect(project_root, reference_job=55518, tuning_job=55537, compact_job=55555,
            subset_size=64):
    import torch
    torch.set_num_threads(1)
    if subset_size < 1:
        raise ValueError("Subset size must be positive")
    project_root = Path(project_root).resolve()
    code_root = project_root / "unimc_model_optimization"
    reference = code_root / "runs" / f"pilot_{reference_job}"
    tuning = code_root / "runs" / f"tune_pilot_{reference_job}_{tuning_job}"
    compact = code_root / "runs" / f"compact_pilot_{reference_job}_{compact_job}"
    raw_cache = code_root / "cache" / f"pilot_{reference_job}"
    mlm_cache = code_root / "cache" / f"mlm_pilot_{reference_job}_{compact_job}"
    files, required = set(), []
    for run_root in (reference, tuning, compact):
        required.append(run_root / "comparison.json")
        files.update(run_root.rglob("*.json"))
        files.update(run_root.glob("comparison.csv"))
    # Inspect an original full head, its two lower-LR variants, and compact heads.
    cases = [reference / "full_seed42_lr1e4", compact / "raw" / "full_seed42_lr1e4"]
    cases += [compact / "mlm" / f"{mode}_seed42_lr1e4" for mode in ("mlp", "self", "full")]
    for case in cases:
        required.extend([case / "initial.pt", case / "best.pt"])
    required += [tuning / f"full_seed42_lr{tag}" / "best.pt" for tag in ("1e5", "3e5")]
    for cache_dir in (raw_cache, mlm_cache):
        for task in ("piqa", "cqa"):
            required.append(cache_dir / f"{task}_val.pt")
            files.update(cache_dir.glob(f"{task}_val.pt.metadata.json"))
        files.update(cache_dir.glob("projection_manifest.json"))
    files.update(required)
    for task in ("piqa", "cqa"):
        path = code_root / "data" / "pilot_seed42_v2" / task / "val.jsonl"
        if path.is_file():
            files.add(path)
    prediction_dir = code_root / "predictions" / f"pilot_{reference_job}"
    files.update(prediction_dir.glob("*_val.jsonl*"))
    for name in ("decision_head.py", "train_head.py", "predict_head.py", "data_io.py", "project_mlm_features.py"):
        path = code_root / name
        if path.is_file():
            files.add(path)
    for prefix, job in (("unimc_pilot", reference_job), ("unimc_tune", tuning_job), ("unimc_compact", compact_job)):
        for extension in ("out", "err"):
            path = project_root / f"{prefix}_{job}.{extension}"
            if path.is_file():
                files.add(path)
    training_inputs = [(view, task, folder / f"{task}_train.pt")
                       for view, folder in (("raw", raw_cache), ("mlm", mlm_cache))
                       for task in ("piqa", "cqa")]
    missing = [str(path) for path in required + [path for _, _, path in training_inputs] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing expected experiment files:\n" + "\n".join(missing))
    for path in files | {path for _, _, path in training_inputs}:
        if not path.resolve().is_relative_to(project_root):
            raise ValueError(f"Input resolves outside the selected project: {path}")
    output = project_root / "UniMC_pilot_diagnostics.zip"
    manifest = {
        "purpose": "Read-only source-pilot diagnosis; not an additional training run",
        "reference_job": reference_job, "tuning_job": tuning_job, "compact_job": compact_job,
        "torch_version": str(torch.__version__), "subset_limit_per_task_view": subset_size,
        "calibration_data_included": False, "target_data_included": False,
        "full_backbone_weights_included": False, "files": [], "training_subsets": [],
    }
    print(f"Collecting {len(files)} existing files plus four training subsets...", flush=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(files):
            name = path.relative_to(project_root).as_posix()
            archive.write(path, name)
            manifest["files"].append({"path": name, "bytes": path.stat().st_size})
        for view, task, path in training_inputs:
            cache = torch.load(path, map_location="cpu", weights_only=True)
            if cache["metadata"]["role"] != "source_train":
                raise ValueError(f"Expected source training cache: {path}")
            metadata = copy.deepcopy(cache["metadata"])
            metadata.update(role="diagnostic_train_subset", source_role="source_train",
                            diagnostic_only=True, n=min(subset_size, len(cache["samples"])),
                            source_cache=path.relative_to(project_root).as_posix())
            # Clone tensor slices to avoid serializing any shared full-batch storage.
            samples = [dict(row, hidden=row["hidden"].clone(), base_logits=row["base_logits"].clone())
                       for row in cache["samples"][:subset_size]]
            buffer = io.BytesIO()
            torch.save({"schema_version": 1, "metadata": metadata, "samples": samples}, buffer)
            name = f"diagnostic_train_subsets/{view}/{task}_first{subset_size}.pt"
            archive.writestr(name, buffer.getvalue())
            manifest["training_subsets"].append({"path": name, "n": len(samples),
                                                  "selection": "First examples from the fixed pilot cache"})
            del cache, samples, buffer
        archive.writestr("diagnostics_manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Archive integrity check failed: {bad}")
    print(f"DIAGNOSTICS READY: {output}", flush=True)
    print(f"SIZE_MIB={output.stat().st_size / 1024**2:.1f}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--reference-job", type=int, default=55518)
    parser.add_argument("--tuning-job", type=int, default=55537)
    parser.add_argument("--compact-job", type=int, default=55555)
    args = parser.parse_args()
    collect(args.project_root, args.reference_job, args.tuning_job, args.compact_job)


if __name__ == "__main__":
    main()
