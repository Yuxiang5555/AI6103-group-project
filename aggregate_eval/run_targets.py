"""Run the fixed comparison inside the historical UniMC CUDA environment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from prepare_targets import ROOT, SOURCES

COMP = ROOT / "unimc_model_optimization"
SEEDS = (42, 43, 44)
HEADS = ("centered_full", "uncentered_full", "centered_self", "centered_mlp")
MODELS = ["baseline", *(f"{head}_seed{seed}" for head in HEADS for seed in SEEDS)]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def run(script, *args):
    subprocess.run([sys.executable, str(COMP / script), *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "handoff", "model", "fengshen-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.inputs / "manifest.json")
    handoff = read(args.handoff / "model_manifest.json")
    models = {Path(m["file"]).stem: m for m in handoff["models"]}
    hashes = {}
    for name in MODELS[1:]:
        checkpoint = args.handoff / "models" / (name + ".pt")
        hashes[name] = sha(checkpoint)
        if hashes[name] != models[name]["sha256"]:
            raise ValueError(f"Model hash mismatch: {name}")
    expected = read(COMP / "results/base_model_signature.json")["feature_signature"]
    if expected != handoff["raw_feature_signature"]:
        raise ValueError("Handoff backbone signature differs from the archived experiment")
    args.out.mkdir(parents=True, exist_ok=True)
    for task in SOURCES:
        inp = args.inputs / (task + ".jsonl")
        fingerprint = dict(input=sha(inp), heads=hashes, signature=expected)
        if fingerprint["input"] != manifest[task]["input_sha256"]:
            raise ValueError(f"Input hash differs from manifest for {task}")
        target = args.out / task
        if target.exists():
            record = read(target / "complete.json")
            if record["fingerprint"] != fingerprint or any(
                sha(target / (name + ".jsonl")) != record["predictions"][name] for name in MODELS
            ):
                raise ValueError(f"Existing outputs for {task} differ")
            continue
        with tempfile.TemporaryDirectory(dir=args.out, prefix=task + "-") as tmp:
            work = Path(tmp)
            results = work / "results"
            results.mkdir()
            raw = work / "raw.pt"
            start = time.monotonic()
            run("extract_features.py", "--input", inp, "--output", raw, "--model", args.model,
                "--fengshen-root", args.fengshen_root, "--device", "cuda:0", "--batch-size", 2,
                "--amp", "none", "--max-length", 512, "--verify-official", "--allow-option-truncation")
            meta = read(work / "raw.pt.metadata.json")
            if (meta["feature_signature"] != expected or not meta["official_prediction_check"]["passed"]
                    or meta["n"] != sum(manifest[task]["conditions"].values())):
                raise ValueError("Target extraction does not match the frozen protocol")
            run("project_mlm_features.py", "--cache", raw, "--output-dir", work / "mlm",
                "--model", args.model, "--fengshen-root", args.fengshen_root, "--device", "cuda:0")
            mlm = work / "mlm" / "raw.pt"
            run("center_option_features.py", "--cache", mlm, "--output-dir", work / "centered")
            for name in MODELS:
                if name == "baseline":
                    cache, flags = raw, ["--baseline"]
                else:
                    cache = mlm if name.startswith("uncentered_") else work / "centered" / "raw.pt"
                    flags = ["--checkpoint", args.handoff / "models" / (name + ".pt")]
                run("predict_head.py", "--cache", cache, *flags, "--device", "cuda:0",
                    "--output", results / (name + ".jsonl"))
            record = dict(fingerprint=fingerprint, extraction=meta,
                          predictions={n: sha(results / (n + ".jsonl")) for n in MODELS},
                          shared_pipeline_wall_seconds=time.monotonic() - start)
            (results / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
            results.rename(target)


if __name__ == "__main__":
    main()
