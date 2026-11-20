"""Run all frozen target comparisons and package the peer-reviewable results."""
import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from run_targets import COMP, MODELS, ROOT, SOURCES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("handoff", type=Path, nargs="?", default=COMP / "models",
                   help="Optional model directory. Defaults to the audited repository heads.")
    p.add_argument("base", type=Path, help="Local UniMC ALBERT backbone directory")
    p.add_argument("fengshen", type=Path, help="Matching Fengshen checkout")
    args = p.parse_args()
    if not any((args.handoff / name).is_file() for name in ("manifest.json", "model_manifest.json")):
        p.error(f"Missing checkpoint manifest in {args.handoff}")
    for path in (args.base / "config.json",
                 args.fengshen / "fengshen"):
        if not path.exists():
            p.error(f"Missing required path: {path}")

    import torch
    if not torch.cuda.is_available():
        p.error("Activate the original UniMC environment on an allocated CUDA node")

    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    inputs = ROOT / "data/aggregate_targets"
    predictions = ROOT / "predictions/aggregate_targets"
    scripts = Path(__file__).resolve().parent
    subprocess.run([sys.executable, scripts / "run_targets.py", "--inputs", inputs,
                    "--handoff", args.handoff, "--model", args.base, "--fengshen-root", args.fengshen,
                    "--out", predictions], check=True)

    archive = output / "UniMC_target_results.tar.gz"
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        summary = Path(tmp) / "summary"
        subprocess.run([sys.executable, scripts / "summarize.py", "--inputs", inputs,
                        "--predictions", predictions, "--output", summary], check=True)
        packed = Path(tmp) / archive.name
        with tarfile.open(packed, "w:gz", compresslevel=4) as tar:
            tar.add(summary, arcname="summary")
            tar.add(inputs / "manifest.json", arcname="target_manifest.json")
            for task in SOURCES:
                files = ["complete.json", *(name + ".jsonl" for name in MODELS)]
                for filename in files:
                    tar.add(predictions / task / filename, arcname=f"predictions/{task}/{filename}")
        packed.replace(archive)
    print(f"Results ready: {archive}", flush=True)


if __name__ == "__main__":
    main()
