"""Score fixed target predictions, aligned by task, condition and original question."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from run_targets import HEADS, MODELS, SEEDS, SOURCES, read, sha


def records(path, task):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            if r["dataset"] != task or r["id"] in rows:
                raise ValueError(f"Wrong task or duplicate ID in {path}")
            rows[r["id"]] = r
    return rows


def ece15(conf, correct):
    bins = np.minimum(14, np.asarray(conf) * 15).astype(int)
    error = np.asarray(conf) - correct
    return float(np.abs(np.bincount(bins, weights=error)).sum() / len(bins))


def rejection_metrics(positives, negatives):
    # Missing-gold is positive. Process tied scores as a group.
    scores = defaultdict(lambda: [0, 0])
    for k, values in enumerate((positives, negatives)):
        for score in values:
            scores[score][k] += 1
    p, n = len(positives), len(negatives)
    tp = fp = area = ap = 0
    for score in sorted(scores, reverse=True):
        dp, dn = scores[score]
        area += (tp + dp / 2) * dn
        tp, fp = tp + dp, fp + dn
        ap += dp * tp / (tp + fp)
    return dict(n_missing=p, n_answerable=n, auroc=area / (p * n), average_precision=ap / p)


def task_scores(data, pred, baseline, calibration=None, calibrated_baseline=None):
    if pred.keys() != data.keys() or baseline.keys() != data.keys():
        raise ValueError("Incomplete or extra predictions")
    if calibration is not None and calibration.keys() != data.keys():
        raise ValueError("Incomplete or extra calibrated predictions")
    grouped = defaultdict(list)
    changes = []
    correct_original = {}
    reject = defaultdict(list)
    for key, src in data.items():
        row, base = pred[key], baseline[key]
        k = len(src["choices"])
        gold = src["gold_label"]
        condition = src["condition"]
        logits = np.asarray(row["logits"], dtype=float)
        if (logits.shape != (k,) or not np.isfinite(logits).all() or row["gold_label"] != gold
                or row["prediction"] != int(logits.argmax()) or row["choices"] != base["choices"]
                or len(row["choices"]) != k or row["base_logits"] != base["logits"]):
            raise ValueError(f"Prediction alignment/score mismatch: {key}")
        if row["choices"] != src["choices"] and not all(
            old.startswith(new) for old, new in zip(src["choices"], row["choices"])
        ):
            raise ValueError(f"Unexpected option text/order: {key}")
        logp = logits - np.logaddexp.reduce(logits)
        probs = np.exp(logp)
        if calibration is not None:
            cal = calibration[key]
            probs = np.asarray(cal["probabilities"], dtype=float)
            if (cal["choices"] != row["choices"] or probs.shape != (k,) or not np.isfinite(probs).all()
                    or np.any(probs < 0) or not np.isclose(probs.sum(), 1, atol=1e-6, rtol=0)):
                raise ValueError(f"Invalid calibrated probabilities: {key}")
            with np.errstate(divide="ignore"):
                logp = np.log(probs)
            reject_score = float(cal["rejection_score"])
        else:
            reject_score = 1 - max(probs)
        if not math.isfinite(reject_score):
            raise ValueError(f"Invalid rejection score: {key}")
        reject[condition].append(reject_score)
        if gold is None:
            if condition != "drop_gold":
                raise ValueError(f"Unexpected missing target: {key}")
            continue
        label = int(probs.argmax())
        parent = src["original_id"]
        base_label = int(base["prediction"] if calibrated_baseline is None
                         else np.argmax(calibrated_baseline[key]["probabilities"]))
        original_key = "original/" + parent
        reference = (calibration[original_key]["probabilities"] if calibration is not None
                     else pred[original_key]["logits"])
        stable = src["original_indices"][label] == int(np.argmax(reference))
        correct = int(label == gold)
        nll = -float(logp[gold])
        if not math.isfinite(nll):
            raise ValueError(f"Zero assigned probability on true class: {key}")
        brier = float(np.sum(probs**2) - 2 * probs[gold] + 1)
        grouped[condition].append((correct, nll, brier, max(probs), stable))
        if condition == "original":
            correct_original[parent] = correct
            if label != base_label:
                ranked = sorted(base["logits"], reverse=True)
                changes.append(dict(id=parent, num_options=k, gold=gold, baseline=base_label,
                                    prediction=label, corrected=int(label == gold), harmed=int(base_label == gold),
                                    baseline_top2_logit_gap=ranked[0] - ranked[1]))
    metrics = {}
    for condition, values in grouped.items():
        acc, nll, brier, conf, stable = np.asarray(values).T
        metrics[condition] = dict(n=len(values), accuracy=float(acc.mean()), nll=float(nll.mean()),
                                  ece15=ece15(conf, acc), brier=float(brier.mean()),
                                  same_original_choice=float(stable.mean()))
    if reject["drop_gold"]:
        metrics["rejection"] = rejection_metrics(reject["drop_gold"], reject["drop_distractor"])
    return metrics, changes, correct_original


def paired_interval(task_deltas, rng):
    # Each question carries all three seed differences. Sampling counts of identical
    # differences is exactly the paired question bootstrap within each task.
    means = np.zeros(20000)
    for deltas in task_deltas:
        vals, freq = np.unique(deltas, return_counts=True)
        sampled = rng.multinomial(len(deltas), freq / len(deltas), size=len(means))
        means += (sampled @ vals) / len(deltas)
    return (100 * np.quantile(means / len(task_deltas), [0.025, 0.975])).tolist()


def summarize(inputs, predictions, output, calibrated=None):
    manifest = read(inputs / "manifest.json")
    all_metrics = {}
    changes = []
    deltas = defaultdict(list)
    for task in SOURCES:
        path = inputs / (task + ".jsonl")
        if sha(path) != manifest[task]["input_sha256"]:
            raise ValueError(f"Target input checksum mismatch: {task}")
        data = records(path, task)
        base = records(predictions / task / "baseline.jsonl", task)
        cal_base = records(calibrated / task / "baseline.jsonl", task) if calibrated else None
        correct = {}
        all_metrics[task] = {}
        for name in MODELS:
            pred = base if name == "baseline" else records(predictions / task / (name + ".jsonl"), task)
            cal = records(calibrated / task / (name + ".jsonl"), task) if calibrated else None
            scores, diff, correct[name] = task_scores(data, pred, base, cal, cal_base)
            all_metrics[task][name] = scores
            changes.extend(dict(task=task, model=name, **d) for d in diff)
        for control in ("baseline", "uncentered_full", "centered_self", "centered_mlp"):
            by_seed = []
            for seed in SEEDS:
                primary = correct[f"centered_full_seed{seed}"]
                ref = correct["baseline" if control == "baseline" else f"{control}_seed{seed}"]
                by_seed.append([v - ref[q] for q, v in primary.items()])
            deltas[control].append(np.mean(by_seed, axis=0))
    macro = {}
    for name in MODELS:
        macro[name] = {metric: mean(all_metrics[t][name]["original"][metric] for t in SOURCES)
                       for metric in ("accuracy", "nll", "ece15", "brier")}
    rng = np.random.default_rng(20261120)
    contrasts = {c: dict(delta_pp=100 * mean(float(d.mean()) for d in tasks),
                         paired_95pct_pp=paired_interval(tasks, rng)) for c, tasks in deltas.items()}
    result = dict(score_source="team_calibrated" if calibrated else "raw", per_task=all_metrics,
                  macro_original=macro, centered_full_contrasts=contrasts)
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = ["task", "model", "id", "num_options", "gold", "baseline", "prediction", "corrected",
              "harmed", "baseline_top2_logit_gap"]
    with (output / "changed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(changes)
    title = "校准组输出" if calibrated else "原始分数"
    lines = ["# 四任务独立评估（" + title + "）", "", "| 模型 | 宏准确率（%） | 种子标准差（百分点） | 宏 NLL |",
             "|---|---:|---:|---:|"]
    for head in ("baseline", *HEADS):
        rows = [macro["baseline"]] if head == "baseline" else [macro[f"{head}_seed{s}"] for s in SEEDS]
        acc = [100 * r["accuracy"] for r in rows]
        sd = "—" if len(rows) == 1 else f"{stdev(acc):.4f}"
        lines.append(f'| {head} | {mean(acc):.4f} | {sd} | {mean(r["nll"] for r in rows):.6f} |')
    lines.extend(["", "## 主候选的配对比较", "", "| 对照 | 宏准确率差（百分点） | 逐题配对 95% 区间 |",
                  "|---|---:|---:|"])
    for control, values in contrasts.items():
        lo, hi = values["paired_95pct_pp"]
        lines.append(f'| {control} | {values["delta_pp"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    lines.extend(["", "## 主候选三个种子的平均值", "",
                  "| 任务 | 原题准确率 | 轮换后准确率 | 轮换后选中同一答案 | 删干扰项准确率 | 缺答案 AUROC | 缺答案 AP |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for task in SOURCES:
        rows = [all_metrics[task][f"centered_full_seed{s}"] for s in SEEDS]
        def fmt(group, field):
            return f"{mean(r[group][field] for r in rows):.4f}" if group in rows[0] else "—"
        cells = [fmt("original", "accuracy"), fmt("rotate", "accuracy"), fmt("rotate", "same_original_choice"),
                 fmt("drop_distractor", "accuracy"), fmt("rejection", "auroc"), fmt("rejection", "average_precision")]
        lines.append("| " + " | ".join([task, *cells]) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--calibrated", type=Path, help="Per-model probabilities and rejection_score from the calibration team")
    summarize(**vars(parser.parse_args()))
