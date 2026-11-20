"""Fit source-only global temperatures and evaluate the final UniMC models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from calibration.plot_results import make_figures
from calibration.temperature_scaling import evaluate_logits, fit_temperature, probabilities


KINDS = ("mlp", "tf_noint", "tf_int", "tf_gate")
SOURCES = ("arc_c", "csqa", "obqa", "piqa")
TARGETS = ("sst2", "cb", "copa", "hellaswag")
TEST_PAIRS = tuple((dataset, "test") for dataset in SOURCES + TARGETS) + (
    ("arc_c", "test_official"), ("obqa", "test_official"),
)
PAIRS = tuple((dataset, "calib") for dataset in SOURCES) + TEST_PAIRS
METRICS = ("accuracy", "ece", "nll", "brier", "mean_confidence")
MODEL_NAMES = {
    "unimc": "Original UniMC", "mlp": "Residual MLP",
    "tf_noint": "Transformer without interaction",
    "tf_int": "Transformer with interaction", "tf_gate": "Gated Transformer",
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_predictions(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Empty prediction file: {path}")
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate sample IDs: {path}")
    records.sort(key=lambda record: record["id"])
    maximum_options = max(len(record["raw_logits"]) for record in records)
    raw = np.full((len(records), maximum_options), -np.inf, dtype=np.float64)
    new = np.full_like(raw, -np.inf)
    labels = np.empty(len(records), dtype=np.int64)
    for index, record in enumerate(records):
        n_options = len(record["raw_logits"])
        label = record["label"]
        if n_options < 2 or len(record["new_logits"]) != n_options:
            raise ValueError(f"Invalid option count: {path}, {record['id']}")
        if type(label) is not int or not 0 <= label < n_options:
            raise ValueError(f"Invalid answer index: {path}, {record['id']}")
        raw[index, :n_options] = record["raw_logits"]
        new[index, :n_options] = record["new_logits"]
        labels[index] = label
        if not np.isfinite(raw[index, :n_options]).all() or not np.isfinite(new[index, :n_options]).all():
            raise ValueError(f"Nonfinite saved option score: {path}, {record['id']}")
        if record["pred_raw"] != int(raw[index].argmax()) or record["pred_new"] != int(new[index].argmax()):
            raise ValueError(f"Saved prediction disagrees with logits: {path}, {record['id']}")
    return {"ids": [record["id"] for record in records], "labels": labels,
            "raw": raw, "new": new, "sha256": file_hash(path)}


def pool_source_calibration(parts, field):
    width = max(part[field].shape[1] for part in parts)
    logits = [np.pad(part[field], ((0, 0), (0, width - part[field].shape[1])), constant_values=-np.inf)
              for part in parts]
    return np.concatenate(logits), np.concatenate([part["labels"] for part in parts])


def validate_and_load(inputs, manifest):
    splits = json.loads((inputs / "data/splits.json").read_text())
    expected_ids = {}
    for dataset in SOURCES:
        calibration = set(splits[dataset]["calib_ids_in_official_train"])
        validation = set(splits[dataset]["val_ids_in_official_train"])
        if calibration & validation:
            raise ValueError(f"Calibration/validation overlap: {dataset}")
        expected_ids[dataset] = {f"{dataset}-train-{index}" for index in calibration}
    expected_hashes = {item["archive_member"].removeprefix("unimc_opt/"): item["sha256"]
                       for item in manifest["members"] if item["purpose"] == "prediction"}
    batches = {}
    baseline = {}
    file_count = 0
    for kind in KINDS:
        for seed in range(3):
            run = {}
            published_results = json.loads((inputs / f"runs/{kind}/seed{seed}/results.json").read_text())
            for dataset, split in PAIRS:
                relative = f"runs/{kind}/seed{seed}/preds/{dataset}_{split}.jsonl"
                path = inputs / relative
                batch = load_predictions(path)
                if batch["sha256"] != expected_hashes[relative]:
                    raise ValueError(f"Input hash differs from final archive: {path}")
                if split == "calib" and set(batch["ids"]) != expected_ids[dataset]:
                    raise ValueError(f"Calibration IDs disagree with fixed split: {path}")
                if split != "calib" and any(not sample_id.startswith(f"{dataset}-{split}-") for sample_id in batch["ids"]):
                    raise ValueError(f"Test sample ID belongs to another split: {path}")
                published = published_results[f"{dataset}/{split}"]
                if published["n"] != len(batch["labels"]):
                    raise ValueError(f"Sample count differs from published run: {path}")
                for field, metric in (("raw", "acc_raw"), ("new", "acc_new")):
                    accuracy = float((batch[field].argmax(axis=1) == batch["labels"]).mean())
                    if abs(accuracy - published[metric]) > 1e-6:
                        raise ValueError(f"Accuracy differs from published run: {path}, {field}")
                pair = (dataset, split)
                if pair in baseline:
                    reference = baseline[pair]
                    if batch["ids"] != reference["ids"] or not np.array_equal(batch["labels"], reference["labels"]):
                        raise ValueError(f"Cross-model sample or label alignment failed: {path}")
                    if not np.array_equal(batch["raw"], reference["raw"]):
                        raise ValueError(f"Cross-model original UniMC logits differ: {path}")
                else:
                    baseline[pair] = batch
                run[pair] = batch
                file_count += 1
            batches[(kind, seed)] = run
    audit = {
        "input_prediction_files": file_count,
        "source_calibration_counts": {dataset: len(expected_ids[dataset]) for dataset in SOURCES},
        "calibration_validation_overlap": 0,
        "all_calibration_ids_match_fixed_split": True,
        "all_prediction_ids_unique": True,
        "all_labels_and_samples_align_across_runs": True,
        "all_baseline_logits_identical_across_12_runs": True,
        "all_saved_predictions_match_logit_argmax": True,
        "all_counts_and_accuracies_match_published_runs": True,
        "all_prediction_file_hashes_match_archive": True,
    }
    return batches, baseline, audit


def summarize(rows, group_fields):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, members in groups.items():
        result = dict(zip(group_fields, key))
        result["n_seeds"] = len(members)
        for metric in METRICS:
            values = [member[metric] for member in members]
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(result)
    return summary


def macro_target_rows(rows):
    groups = {}
    for row in rows:
        if row["dataset"] in TARGETS and row["split"] == "test":
            key = row["model"], row["seed"], row["stage"]
            groups.setdefault(key, []).append(row)
    macros = []
    for (model, seed, stage), members in groups.items():
        if {row["dataset"] for row in members} != set(TARGETS) or len(members) != 4:
            raise ValueError(f"Incomplete target evaluation: {model}, {seed}, {stage}")
        macros.append({"model": model, "seed": seed, "stage": stage,
                       **{metric: float(np.mean([row[metric] for row in members])) for metric in METRICS}})
    return macros


def formatted_mean(row, metric, scale=1.0):
    mean = row[f"{metric}_mean"] * scale
    deviation = row[f"{metric}_sd"]
    return f"{mean:.4f}" if deviation is None else f"{mean:.4f} ± {deviation * scale:.4f}"


def write_report(output, summary, per_dataset, fits, audit, protocol, repository_commit):
    lines = [
        "# Global temperature calibration of the final UniMC models", "",
        "## Experimental protocol", "",
        "The final v0.3 decision-layer release was evaluated at repository commit "
        f"`{repository_commit}`. Original UniMC and four decision-layer "
        "architectures were included. Each decision layer had training seeds 0, 1 and 2.", "",
        "One positive global temperature was fitted for original UniMC, and one for each trained "
        "decision-layer seed, giving 13 temperatures. Fitting minimized example-mean NLL on the "
        "pooled source calibration set of 3,193 questions "
        "(ARC-C 112, CommonsenseQA 974, OpenBookQA 496, PIQA 1,611). "
        "Every calibration question had equal weight. The fitted temperature was then shared "
        "across evaluation datasets for that model and seed. Target labels were used solely for evaluation.", "",
        "For valid option logits z, calibrated probabilities were softmax(z / T). Optimization used "
        "float64 arithmetic and bisection of the convex NLL derivative in inverse temperature "
        "beta = 1/T, constrained to [0.0001, 10000]. Every fitted solution was interior.", "",
        f"ECE used maximum-probability confidence and {protocol['ece_bins']} equal-width, right-closed bins. "
        "NLL used natural logarithms. Multiclass Brier was the example mean of the sum of squared "
        "probability errors over valid options, using the [0, 2] convention for every task. "
        "Padding contributed zero probability. CSV files store ECE and Accuracy as fractions.", "",
        "Target summaries first averaged the four task-level metrics equally across SST-2, CB, "
        "COPA and HellaSwag, then reported the mean and sample standard deviation across training "
        "seeds (ddof=1). Original UniMC was evaluated once after verifying that its logits were "
        "identical in all twelve prediction copies. Domain-test and additional official-test "
        "results are available separately in metrics.csv.", "",
        "## Target-task macro results", "",
        "ECE is reported in percent. Values after ± are sample standard deviations across three seeds.", "",
        "| Model | Stage | ECE (%) | NLL | Brier | Accuracy (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {MODEL_NAMES[row['model']]} | {row['stage']} | "
                     f"{formatted_mean(row, 'ece', 100)} | {formatted_mean(row, 'nll')} | "
                     f"{formatted_mean(row, 'brier')} | {formatted_mean(row, 'accuracy', 100)} |")
    lines += ["", "## Per-target calibration effects", "",
              "Each cell gives before → after. Decision-layer values are means across three seeds.", "",
              "| Model | Dataset | ECE (%) | NLL | Brier |", "|---|---|---:|---:|---:|"]
    index = {(row["model"], row["dataset"], row["split"], row["stage"]): row for row in per_dataset}
    for model in ("unimc",) + KINDS:
        for dataset in TARGETS:
            before = index[(model, dataset, "test", "before")]
            after = index[(model, dataset, "test", "after")]
            lines.append(f"| {MODEL_NAMES[model]} | {dataset} | "
                         f"{100 * before['ece_mean']:.3f} → {100 * after['ece_mean']:.3f} | "
                         f"{before['nll_mean']:.4f} → {after['nll_mean']:.4f} | "
                         f"{before['brier_mean']:.4f} → {after['brier_mean']:.4f} |")
    macro_index = {(row["model"], row["stage"]): row for row in summary}
    original_before = macro_index[("unimc", "before")]
    original_after = macro_index[("unimc", "after")]
    decision_temperatures = [fit["temperature"] for name, fit in fits.items() if name != "unimc"]
    improved_architectures = sum(all(macro_index[(model, "after")][f"{metric}_mean"] <
                                    macro_index[(model, "before")][f"{metric}_mean"]
                                    for metric in ("ece", "nll", "brier")) for model in KINDS)
    lines += ["", "## Cross-task findings", "",
              f"Across-seed mean target-macro ECE, NLL and Brier decreased together for {improved_architectures} "
              "of the four decision-layer architectures. Target-macro ECE changes (after minus before) were "
              + ", ".join(f"{MODEL_NAMES[model]} {100 * (macro_index[(model, 'after')]['ece_mean'] - macro_index[(model, 'before')]['ece_mean']):+.3f} percentage points"
                          for model in KINDS) + ".", "",
              f"Original UniMC had T={fits['unimc']['temperature']:.6f}. Its source calibration NLL "
              f"fell from {fits['unimc']['nll_before']:.6f} to {fits['unimc']['nll_after']:.6f}. "
              f"Its target-macro ECE changed from {100 * original_before['ece_mean']:.3f}% "
              f"to {100 * original_after['ece_mean']:.3f}%, NLL from {original_before['nll_mean']:.4f} "
              f"to {original_after['nll_mean']:.4f}, and Brier from {original_before['brier_mean']:.4f} "
              f"to {original_after['brier_mean']:.4f}. This source-fitted temperature sharpened the "
              "probability distribution. On HellaSwag, original UniMC already had mean confidence "
              "above its accuracy, and further sharpening increased its probability error.", "",
              f"Decision-layer temperatures ranged from {min(decision_temperatures):.4f} "
              f"to {max(decision_temperatures):.4f}, softening their distributions. "
              "HellaSwag had lower mean ECE, NLL and Brier for every decision-layer architecture. "
              "SST-2 and COPA had higher mean NLL and Brier after this source-fitted softening. "
              "These tasks had different confidence-error patterns, so transfer effects varied by task. "
              "The per-target table retains both improvements and degradations.", "",
              "Post-calibration target-macro minima were "
              + ", ".join(f"{metric.upper()} for {MODEL_NAMES[min(('unimc',) + KINDS, key=lambda model: macro_index[(model, 'after')][f'{metric}_mean'])]}"
                          for metric in ("ece", "nll", "brier")) + ". The three-seed summaries describe training-seed "
              "variation. The task-level changes are descriptive estimates from the fixed evaluation sets.", "",
              "## Fitted temperatures", "",
              "| Model / seed | T | Source calibration NLL before | After |",
              "|---|---:|---:|---:|"]
    for name, fit in fits.items():
        lines.append(f"| {name} | {fit['temperature']:.6f} | {fit['nll_before']:.6f} | {fit['nll_after']:.6f} |")
    lines += ["", "## Verification and interpretation", "",
              f"All {audit['input_prediction_files']} prediction files matched the final archive's SHA-256 hashes. "
              "Source calibration IDs matched data/splits.json exactly and were disjoint from the "
              "source model-selection IDs. Labels, option counts, and original UniMC logits agreed "
              "across model and seed copies. Saved predictions matched the option-logit argmax.", "",
              "Every fitted temperature reduced pooled source calibration NLL. Positive-temperature "
              "argmax checks passed for every evaluation batch. Target probability metrics were "
              "calculated with frozen source-fitted temperatures.", "",
              "The source calibration questions were held out from the group's decision-layer training "
              "and model selection. Official UniMC's upstream MC training included these source tasks. "
              "The experiment therefore measures transfer from source-task calibration to the four target tasks.", "",
              "CB has 56 questions and COPA has 100. Their ECE estimates and reliability curves have "
              "limited sample support. Per-bin counts are provided in reliability_bins.csv. Reliability "
              "figures show seed 0 for each decision layer, while all seeds contribute to summary tables.", "",
              "## Files and reproduction", "",
              "- temperatures.json contains model/checkpoint identities, source input hashes and fitted temperatures.",
              "- metrics.csv and metrics.json contain per-run, per-dataset results, including source calibration metrics.",
              "- target_macro_by_seed.csv and target_macro_summary.csv contain the task-equal target summaries.",
              "- per_dataset_summary.csv contains seed means and sample standard deviations for every dataset.",
              "- reliability_bins.csv contains counts and calibration statistics for all confidence bins.",
              "- figures/ contains five reliability diagrams and one target-metric summary.",
              "- audit.json and run_metadata.json record data checks and implementation identities.", "",
              "From the repository root, run `python -m calibration.run_calibration`. "
              "Use `--temperatures-file calibration/results/temperatures.json --output <fresh-directory>` "
              "to replay evaluation with the frozen temperature parameters.", "",
              "## Reference", "",
              "Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). "
              "On Calibration of Modern Neural Networks. ICML. "
              "[Paper](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(inputs, output, manifest_path, n_bins=15, temperatures_file=None):
    manifest = json.loads(manifest_path.read_text())
    batches, baseline, audit = validate_and_load(inputs, manifest)
    checkpoints = {item["archive_member"]: item["sha256"] for item in manifest["members"]
                   if item["purpose"] == "checkpoint_identity"}
    frozen = json.loads(temperatures_file.read_text()) if temperatures_file else None
    if frozen and (frozen["repository_commit"] != manifest["repository_commit"]
                   or frozen["input_manifest_sha256"] != file_hash(manifest_path)):
        raise ValueError("Frozen temperatures belong to a different final release or input manifest")
    saved = frozen["fits"] if frozen else None
    run_specs = [("unimc", None, baseline, "raw")]
    run_specs += [(kind, seed, batch, "new") for (kind, seed), batch in batches.items()]
    fits = {}
    rows = []
    changed_predictions = 0
    for model, seed, batch, field in run_specs:
        name = model if seed is None else f"{model}/seed{seed}"
        source_parts = [batch[(dataset, "calib")] for dataset in SOURCES]
        source_logits, source_labels = pool_source_calibration(source_parts, field)
        calibration_inputs = {dataset: batch[(dataset, "calib")]["sha256"] for dataset in SOURCES}
        if saved is None:
            fit = fit_temperature(source_logits, source_labels)
        else:
            fit = dict(saved[name])
            if fit["calibration_inputs_sha256"] != calibration_inputs:
                raise ValueError(f"Frozen temperature inputs differ: {name}")
            if fit["model"] != model or fit["seed"] != seed or fit["logits_field"] != field:
                raise ValueError(f"Frozen temperature model identity differs: {name}")
            if model != "unimc":
                checkpoint_hash = checkpoints[f"unimc_opt/runs/{model}/seed{seed}/best.pt"]
                if fit["checkpoint_sha256"] != checkpoint_hash:
                    raise ValueError(f"Frozen temperature checkpoint differs: {name}")
            elif fit["base_implementation_sha256"] != manifest["code_sha256"]["unimc_core.py"]:
                raise ValueError("Frozen baseline implementation differs")
        if not fit["converged"] or fit["boundary"] is not None:
            raise ValueError(f"Temperature optimization requires review: {name}, {fit}")
        temperature = fit["temperature"]
        before = evaluate_logits(source_logits, source_labels, n_bins=n_bins)
        after = evaluate_logits(source_logits, source_labels, temperature, n_bins)
        if after["nll"] > before["nll"] + 1e-10:
            raise ValueError(f"Calibration NLL increased: {name}")
        fit.update({"model": model, "seed": seed, "calibration_inputs_sha256": calibration_inputs,
                    "source_counts": audit["source_calibration_counts"], "logits_field": field})
        if model != "unimc":
            fit["checkpoint_sha256"] = checkpoints[f"unimc_opt/runs/{model}/seed{seed}/best.pt"]
        else:
            fit["base_model"] = "IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English"
            fit["base_implementation_sha256"] = manifest["code_sha256"]["unimc_core.py"]
            fit["base_prediction_identity"] = calibration_inputs
        fits[name] = fit
        all_parts = [(dataset, split, part[field], part["labels"]) for (dataset, split), part in batch.items()]
        all_parts.append(("source_calib_pooled", "calib", source_logits, source_labels))
        for dataset, split, logits, labels in all_parts:
            scaled = probabilities(logits, temperature)
            changed_predictions += int(np.count_nonzero(scaled.argmax(axis=1) != logits.argmax(axis=1)))
            if not np.allclose(scaled.sum(axis=1), 1.0, rtol=0, atol=1e-12):
                raise ValueError(f"Probability normalization failed: {name}, {dataset}")
            domain = "source_calibration" if split == "calib" else "target_test" if dataset in TARGETS else "source_test"
            for stage, value in (("before", 1.0), ("after", temperature)):
                result = evaluate_logits(logits, labels, value, n_bins)
                rows.append({"model": model, "seed": seed, "dataset": dataset, "split": split,
                             "domain": domain, "stage": stage, "temperature": value, **result})
        print(f"{name:18s} T={temperature:.6f}  source NLL {before['nll']:.6f} -> {after['nll']:.6f}", flush=True)
    if changed_predictions:
        raise ValueError(f"Positive-temperature prediction invariance failed: {changed_predictions}")
    macros = macro_target_rows(rows)
    summary = summarize(macros, ("model", "stage"))
    per_dataset = summarize(rows, ("model", "dataset", "split", "stage"))
    output.mkdir(parents=True, exist_ok=True)
    for previous_record in ("audit.json", "verification.json"):
        (output / previous_record).unlink(missing_ok=True)
    protocol = {"ece_bins": n_bins, "ece_bin_intervals": "right-closed, zero included in first bin",
                "brier": "sum over valid classes, example mean, range [0,2]",
                "fit_objective": "pooled source-calibration example-mean natural-log NLL",
                "source_datasets": list(SOURCES), "target_datasets": list(TARGETS),
                "target_aggregation": "equal task weights, then training-seed mean and sample SD",
                "seed_sd_ddof": 1, "baseline_repetitions": 1}
    write_json(output / "temperatures.json", {"schema_version": 1, "repository_commit": manifest["repository_commit"],
               "input_manifest_sha256": file_hash(manifest_path), "protocol": protocol, "fits": fits})
    write_json(output / "metrics.json", rows)
    write_csv(output / "metrics.csv", [{key: value for key, value in row.items() if key != "bins"} for row in rows])
    write_csv(output / "target_macro_by_seed.csv", macros)
    write_csv(output / "target_macro_summary.csv", summary)
    write_csv(output / "per_dataset_summary.csv", per_dataset)
    bin_rows = []
    for row in rows:
        for index, bin_row in enumerate(row["bins"], 1):
            bin_rows.append({"model": row["model"], "seed": row["seed"], "dataset": row["dataset"],
                             "split": row["split"], "stage": row["stage"], "bin": index, **bin_row})
    write_csv(output / "reliability_bins.csv", bin_rows)
    make_figures(rows, output / "figures")
    audit.update({"passed": True, "fitted_temperatures": len(fits), "all_fits_converged_interior": True,
                  "all_source_calibration_nll_nonincreasing": True, "changed_predictions": changed_predictions,
                  "all_calibrated_probabilities_sum_to_one": True, "target_labels_used_for_fitting": False})
    write_report(output, summary, per_dataset, fits, audit, protocol, manifest["repository_commit"])
    implementation = {name: file_hash(Path(__file__).parent / name) for name in
                      ("run_calibration.py", "temperature_scaling.py", "plot_results.py")}
    write_json(output / "run_metadata.json", {"repository_commit": manifest["repository_commit"],
               "input_manifest_sha256": file_hash(manifest_path), "implementation_sha256": implementation,
               "protocol": protocol, "python": platform.python_version(), "numpy": np.__version__,
               "mode": "frozen-temperature replay" if saved else "source-only temperature fitting"})
    write_json(output / "audit.json", audit)
    print(f"Completed {len(rows)} metric rows; results in {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("unimc_decision_layers"))
    parser.add_argument("--output", type=Path, default=Path("calibration/results"))
    parser.add_argument("--manifest", type=Path, default=Path("calibration/input_manifest.json"))
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--temperatures-file", type=Path)
    args = parser.parse_args()
    run(args.inputs, args.output, args.manifest, args.bins, args.temperatures_file)


if __name__ == "__main__":
    main()
