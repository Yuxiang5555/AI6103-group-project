"""Summarize exactly three fixed-configuration LoRA runs; no training or selection."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import zipfile

import torch

from data_io import file_sha256, write_json
from run_unimc_lora import CODE_FILES
from unimc_online import metrics_from_predictions


def collect(paths):
    records, predictions, prototype, seen = [], {}, None, set()
    fixed_keys = ("rank", "alpha", "lora_dropout", "lr", "weight_decay", "epochs", "patience", "micro_batch", "grad_accum", "amp", "max_length", "no_gradient_checkpointing")
    for path in map(Path, paths):
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary.get("synthetic_only") or summary["seed"] in seen or summary["seed"] not in (42, 43, 44):
            raise ValueError("Need unique real-data seeds 42/43/44")
        seen.add(summary["seed"])
        settings = {"fixed_arguments": {k: config["arguments"][k] for k in fixed_keys},
                    **{k:config[k] for k in ("adapter_spec", "base_signature", "source_data_sha256", "code_sha256", "baseline_source_validation")}}
        if prototype is None:
            prototype = settings
        elif prototype != settings:
            raise ValueError("The three runs do not use the same data/settings/base/code")
        rows = [json.loads(line) for line in (path / "best_val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        indexed = {(r["dataset"], str(r["id"])): r for r in rows}
        if len(indexed) != len(rows) or len(rows) != 1517:
            raise ValueError("Expected exactly 1517 unique source-validation questions")
        for other in predictions.values():
            if indexed.keys() != other.keys() or any(any(indexed[k][f] != other[k][f] for f in ("choices", "gold_label", "base_logits")) for k in indexed):
                raise ValueError("Paired predictions differ in IDs/options/labels/baseline")
        predictions[summary["seed"]] = indexed
        metrics = metrics_from_predictions(rows)
        if abs(metrics["macro_accuracy"]-summary["best_source_validation"]["macro_accuracy"]) > 1e-12:
            raise ValueError("Recomputed source accuracy differs")
        saved = torch.load(path / "best_adapter.pt", map_location="cpu", weights_only=True)
        if saved["seed"] != summary["seed"] or saved["source_validation"] != summary["best_source_validation"]:
            raise ValueError("Adapter checkpoint differs from its summary")
        metadata = json.loads((path / "best_val.jsonl.metadata.json").read_text(encoding="utf-8"))
        if (metadata["role"] != "source_val" or not metadata["uncalibrated"] or metadata["evaluation_amp"] != "none"
                or metadata["checkpoint_sha256"] != file_sha256(path / "best_adapter.pt")
                or saved["base_signature"] != config["base_signature"] or saved["adapter_spec"] != config["adapter_spec"]):
            raise ValueError("Export metadata or saved adapter signature differs")
        corrected = sum(r["prediction"] == r["gold_label"] and max(range(len(r["base_logits"])), key=r["base_logits"].__getitem__) != r["gold_label"] for r in rows)
        harmed = sum(r["prediction"] != r["gold_label"] and max(range(len(r["base_logits"])), key=r["base_logits"].__getitem__) == r["gold_label"] for r in rows)
        records.append({"seed": summary["seed"], "run_dir": str(path), "best_epoch": summary["best_epoch"],
                        "macro_accuracy": metrics["macro_accuracy"], "macro_nll": metrics["macro_nll"],
                        "delta_macro_pp": summary["delta_macro_pp"], "corrected": corrected, "harmed": harmed,
                        "checkpoint_sha256": file_sha256(path / "best_adapter.pt")})
    if seen != {42, 43, 44}:
        raise ValueError("Need all three seeds")
    records.sort(key=lambda r:r["seed"])
    result = {"scope": "Fixed-configuration source-development evaluation; independent target effects untested",
              "seed_results": records, "mean_macro_accuracy": statistics.mean(r["macro_accuracy"] for r in records),
              "sd_macro_accuracy": statistics.stdev(r["macro_accuracy"] for r in records),
              "mean_macro_nll": statistics.mean(r["macro_nll"] for r in records),
              "mean_delta_macro_pp": statistics.mean(r["delta_macro_pp"] for r in records),
              "settings": prototype, "source_cal_used": False, "target_data_used": False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = collect(args.runs)
    output.mkdir(parents=True)
    write_json(output / "comparison.json", result)
    with (output / "seed_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["seed_results"][0]))
        writer.writeheader()
        writer.writerows(result["seed_results"])
    archive = Path.cwd().parent / "UniMC_lora_three_seed_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in output.iterdir():
            bundle.write(path, path.name)
        for row in result["seed_results"]:
            folder = Path(row["run_dir"])
            for name in ("config.json", "summary.json", "history.json", "best_adapter.pt", "best_val.jsonl", "best_val.jsonl.metadata.json"):
                bundle.write(folder / name, f"seed{row['seed']}/{name}")
        for name in CODE_FILES+("data_io.py",):
            bundle.write(name, "code/"+name)
        bundle.write("README_UniMC_LoRA实施.md", "README.md")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip():
            raise ValueError("Result archive CRC failed")
    print(f"LORA THREE SEEDS Macro={100*result['mean_macro_accuracy']:.4f}% +/- {100*result['sd_macro_accuracy']:.4f}pp Delta={result['mean_delta_macro_pp']:+.4f}pp")
    print(f"REPORT_ARCHIVE={archive}")


if __name__ == "__main__":
    main()
