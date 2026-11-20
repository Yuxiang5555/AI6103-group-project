"""Question-paired source-development analysis, keeping all three seeds together."""
import argparse
import csv
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "output/UniMC_centering_analysis_55757"
SEEDS = (42, 43, 44)
TASKS = ("piqa", "commonsenseqa")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    global DEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(value is not None for value in vars(args).values()):
        if any(value is None for value in vars(args).values()):
            parser.error("Supply all of --predictions, --comparison, --source-data and --output")
        DEST = args.output
        DEST.mkdir(parents=True, exist_ok=False)
        source = json.loads(args.comparison.read_text(encoding="utf-8"))
        predictions, source_data = args.predictions, args.source_data
    else:
        DEST.mkdir(exist_ok=False)
        with zipfile.ZipFile(ROOT / "output/UniMC_centering_confirmation.zip") as archive:
            for item in archive.infolist():
                if not (DEST / item.filename).resolve().is_relative_to(DEST.resolve()):
                    raise ValueError(item.filename)
            archive.extractall(DEST)
        source = json.loads((DEST / "comparison.json").read_text(encoding="utf-8"))
        predictions = DEST / Path(source["prediction_dir"]).name
        source_data = ROOT / "output/UniMC_diagnostics_55518_55555/unimc_model_optimization/data/pilot_seed42_v2"
    baseline_rows = read_rows(predictions / "baseline_val.jsonl")
    base_index = {(row["dataset"], row["id"]): row for row in baseline_rows}
    if len(base_index) != 1517:
        raise ValueError("Incorrect baseline input count/unique IDs")
    keys = sorted(base_index)
    gold = np.array([base_index[key]["gold_label"] for key in keys])
    base_pred = np.array([base_index[key]["prediction"] for key in keys])
    base_ok = base_pred == gold
    groups = {task: np.array([i for i, key in enumerate(keys) if key[0] == task]) for task in TASKS}
    if [len(groups[task]) for task in TASKS] != [919, 598]:
        raise ValueError("Unexpected task sizes")
    labels, correct = {}, {}
    checks = []
    per_seed = []
    for variant in ("uncentered", "centered"):
        for mode in ("mlp", "self", "full"):
            labels[(variant, mode)] = []
            for seed in SEEDS:
                rows = read_rows(predictions / f"{variant}_{mode}_seed{seed}_val.jsonl")
                indexed = {(row["dataset"], row["id"]): row for row in rows}
                if len(indexed) != len(rows) or indexed.keys() != base_index.keys():
                    raise ValueError("Different/duplicate prediction IDs")
                for key, previous in base_index.items():
                    if any(indexed[key][field] != previous[field] for field in ("gold_label", "choices", "base_logits")):
                        raise ValueError(f"Different labels/options/base scores: {variant}/{mode}/{seed}/{key}")
                pred = np.array([indexed[key]["prediction"] for key in keys])
                ok = pred == gold
                labels[(variant, mode)].append(pred)
                task_metrics = {}
                for task, positions in groups.items():
                    losses = []
                    for i in positions:
                        score = np.array(indexed[keys[i]]["logits"], dtype=np.float64)
                        log_sum_exp = score.max() + np.log(np.exp(score - score.max()).sum())
                        losses.append(float(log_sum_exp - score[gold[i]]))
                    task_metrics[task] = {"n": len(positions), "accuracy": float(ok[positions].mean()),
                                          "nll": float(np.mean(losses)),
                                          "changed": int((pred[positions] != base_pred[positions]).sum()),
                                          "corrected": int((ok[positions] & ~base_ok[positions]).sum()),
                                          "harmed": int((~ok[positions] & base_ok[positions]).sum())}
                macro = np.mean([row["accuracy"] for row in task_metrics.values()])
                nll = np.mean([row["nll"] for row in task_metrics.values()])
                reference = next(row for row in source["seed_results"] if
                                 (row["variant"], row["mode"], row["seed"]) == (variant, mode, seed))
                if abs(macro - reference["macro_accuracy"]) > 1e-12 or abs(nll - reference["macro_nll"]) > 3e-7:
                    raise ValueError("Recomputed metrics differ from exported report")
                per_seed.append({"variant": variant, "mode": mode, "seed": seed,
                                 "best_epoch": reference["best_epoch"], "macro_accuracy": float(macro),
                                 "macro_nll": float(nll), "per_dataset": task_metrics})
            labels[(variant, mode)] = np.stack(labels[(variant, mode)])
            correct[(variant, mode)] = labels[(variant, mode)] == gold[None, :]
    # Resample questions independently within each task. Carry the three seed
    # predictions together; do not treat three copies of one question as new data.
    contrasts = {}
    for mode in ("mlp", "self", "full"):
        contrasts[f"centered_{mode}_minus_baseline"] = correct[("centered", mode)].mean(0) - base_ok
        contrasts[f"centered_{mode}_minus_uncentered"] = (correct[("centered", mode)].astype(float) -
                                                          correct[("uncentered", mode)]).mean(0)
    for mode in ("mlp", "self"):
        contrasts[f"centered_full_minus_{mode}"] = (correct[("centered", "full")].astype(float) -
                                                    correct[("centered", mode)]).mean(0)
    names = list(contrasts)
    matrix = np.stack([contrasts[name] for name in names])
    rng = np.random.default_rng(20261118)
    draws = np.zeros((20000, len(names)))
    for start in range(0, len(draws), 500):
        stop = min(start + 500, len(draws))
        for positions in groups.values():
            indexes = rng.integers(0, len(positions), size=(stop - start, len(positions)))
            for j in range(len(names)):
                draws[start:stop, j] += 0.5 * matrix[j, positions][indexes].mean(1)
    bootstrap = {}
    for j, name in enumerate(names):
        point = np.mean([matrix[j, positions].mean() for positions in groups.values()])
        ci = np.quantile(draws[:, j], [0.025, 0.975])
        bootstrap[name] = {"mean_delta_pp": float(100 * point), "exploratory_ci95_pp": list(100 * ci)}
    stability = {}
    for mode in ("mlp", "self", "full"):
        ok = correct[("centered", mode)]
        pred = labels[("centered", mode)]
        stability[mode] = {"stable_corrected_all_seeds": int((~base_ok & ok.all(0)).sum()),
                           "stable_harmed_all_seeds": int((base_ok & (~ok).all(0)).sum()),
                           "corrected_in_any_seed": int((~base_ok & ok.any(0)).sum()),
                           "harmed_in_any_seed": int((base_ok & (~ok).any(0)).sum()),
                           "prediction_disagreement_across_seeds": int((pred != pred[:1]).any(0).sum())}
    text_index = {}
    for directory in ("piqa", "cqa"):
        for row in read_rows(source_data / directory / "val.jsonl"):
            text_index[(row["dataset"], row["id"])] = row
    error_rows = []
    full_ok, full_pred = correct[("centered", "full")], labels[("centered", "full")]
    for i, key in enumerate(keys):
        if not (full_pred[:, i] != base_pred[i]).any():
            continue
        fixed = bool(not base_ok[i] and full_ok[:, i].all())
        harmed = bool(base_ok[i] and (~full_ok[:, i]).all())
        category = "stable_corrected" if fixed else "stable_harmed" if harmed else "seed_dependent_change"
        row, original = base_index[key], text_index[key]
        sorted_scores = sorted(row["base_logits"], reverse=True)
        error_rows.append({"dataset": key[0], "id": key[1], "category": category,
                           "texta": original["texta"], "question": original["question"],
                           "choices": json.dumps(row["choices"], ensure_ascii=False), "gold_label": int(gold[i]),
                           "baseline_prediction": int(base_pred[i]), "seed42_prediction": int(full_pred[0, i]),
                           "seed43_prediction": int(full_pred[1, i]), "seed44_prediction": int(full_pred[2, i]),
                           "base_top2_margin": sorted_scores[0] - sorted_scores[1],
                           "correct_seed_count": int(full_ok[:, i].sum())})
    error_rows.sort(key=lambda row: (row["category"], row["dataset"], row["id"]))
    with (DEST / "full_changed_examples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(error_rows[0]))
        writer.writeheader()
        writer.writerows(error_rows)
    audit = {
        "scope": "Exploratory source-development analysis after adaptive model selection, not confirmatory test evidence",
        "audit": {"unique_validation_questions": len(keys), "models": len(per_seed),
                  "all_prediction_ids_labels_choices_and_base_logits_match": True,
                  "all_metrics_recomputed": True},
        "seed_results": per_seed, "bootstrap_seed": 20261118, "bootstrap_draws": len(draws),
        "bootstrap_method": "Task-stratified paired question bootstrap, seeds carried together, fixed trained models",
        "exploratory_bootstrap": bootstrap, "stability": stability,
        "original_summary": source,
    }
    (DEST / "paired_analysis.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("AUDIT", json.dumps(audit["audit"]))
    print("BOOTSTRAP", json.dumps(bootstrap))
    print("STABILITY", json.dumps(stability))
    for row in per_seed:
        if row["variant"] == "centered":
            print("CENTERED_SEED", json.dumps(row))
    print("ANALYSIS_READY", DEST)


if __name__ == "__main__":
    main()
