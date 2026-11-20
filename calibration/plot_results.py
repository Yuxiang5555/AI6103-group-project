"""Create reliability diagrams and target-task calibration summary figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MODELS = ("unimc", "mlp", "tf_noint", "tf_int", "tf_gate")
TARGETS = ("sst2", "cb", "copa", "hellaswag")
MODEL_NAMES = {
    "unimc": "Original UniMC",
    "mlp": "Residual MLP",
    "tf_noint": "Transformer without option interaction",
    "tf_int": "Transformer with option interaction",
    "tf_gate": "Gated option-interaction Transformer",
}
DATASET_NAMES = {"sst2": "SST-2", "cb": "CB", "copa": "COPA", "hellaswag": "HellaSwag"}
STAGES = ("before", "after")
COLORS = {"before": "#D55E00", "after": "#0072B2"}
STAGE_NAMES = {"before": "Before calibration", "after": "After calibration"}


def _prediction_row(index, model, seed, dataset, stage):
    key = (model, seed, dataset, stage)
    if key not in index:
        raise ValueError(f"Missing target metric row: {key}")
    return index[key]


def _reliability_figure(index, model, output_dir):
    seed = None if model == "unimc" else 0
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    temperature = _prediction_row(index, model, seed, TARGETS[0], "after")["temperature"]

    for axis, dataset in zip(axes.flat, TARGETS):
        axis.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=1.4)
        before = _prediction_row(index, model, seed, dataset, "before")
        after = _prediction_row(index, model, seed, dataset, "after")
        for stage, row in (("before", before), ("after", after)):
            populated = [bin_row for bin_row in row["bins"] if bin_row["count"] > 0]
            axis.plot(
                [bin_row["confidence"] for bin_row in populated],
                [bin_row["accuracy"] for bin_row in populated],
                color=COLORS[stage],
                linestyle="--" if stage == "before" else "-",
                marker="o" if stage == "before" else "s",
                markersize=4.5,
                linewidth=1.8,
                label=STAGE_NAMES[stage],
            )
        axis.set_title(f"{DATASET_NAMES[dataset]}  (n = {before['n']:,})", pad=9)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xticks(np.linspace(0, 1, 5))
        axis.set_yticks(np.linspace(0, 1, 5))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#E5E7EB", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.text(
            0.04,
            0.96,
            f"ECE  {100 * before['ece']:.2f}% → {100 * after['ece']:.2f}%",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3},
        )

    for axis in axes[-1, :]:
        axis.set_xlabel("Mean confidence in bin")
    for axis in axes[:, 0]:
        axis.set_ylabel("Accuracy in bin")

    run_label = "Frozen baseline" if seed is None else "Representative seed 0"
    figure.suptitle(f"{MODEL_NAMES[model]}\n{run_label} · global temperature T = {temperature:.4f}", y=0.98)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=2, frameon=False)
    n_bins = len(_prediction_row(index, model, seed, TARGETS[0], "before")["bins"])
    figure.text(
        0.5,
        0.025,
        f"{n_bins} equal-width confidence bins. Empty bins omitted. Dotted line indicates perfect calibration.\n"
        "Temperature fitted on source calibration data. Target labels used for evaluation.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.12, wspace=0.15, hspace=0.24)
    path = output_dir / f"reliability_{model}.png"
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)
    return str(path.resolve())


def _target_summary_figure(index, seeds_by_model, output_dir):
    metrics = (("ece", "ECE (%)", 100), ("nll", "Negative log-likelihood", 1), ("brier", "Multiclass Brier score", 1))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    positions = np.arange(len(MODELS))
    width = 0.36

    for axis, (metric, label, scale) in zip(axes, metrics):
        for stage, offset in (("before", -width / 2), ("after", width / 2)):
            means = []
            deviations = []
            for model in MODELS:
                task_means = [
                    np.mean([
                        _prediction_row(index, model, seed, dataset, stage)[metric]
                        for dataset in TARGETS
                    ]) * scale
                    for seed in seeds_by_model[model]
                ]
                means.append(float(np.mean(task_means)))
                deviations.append(float(np.std(task_means, ddof=1)) if len(task_means) > 1 else None)
            axis.bar(
                positions + offset,
                means,
                width,
                color=COLORS[stage],
                alpha=0.9,
                label=STAGE_NAMES[stage],
            )
            for position, mean, deviation in zip(positions + offset, means, deviations):
                if deviation is not None:
                    axis.errorbar(position, mean, yerr=deviation, fmt="none", color="#222222", capsize=3, linewidth=1.2)
        axis.set_xticks(positions, ("UniMC", "MLP", "tf_noint", "tf_int", "tf_gate"), rotation=25, ha="right")
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.set_title(label)

    figure.suptitle("Calibration on unseen target tasks", y=0.98, fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=False)
    seed_counts = sorted({len(seeds_by_model[model]) for model in MODELS if model != "unimc"})
    count_text = "/".join(str(count) for count in seed_counts)
    figure.text(
        0.5,
        0.025,
        "Equal-weight task averages over SST-2, CB, COPA and HellaSwag. Lower is better.\n"
        f"Error bars show sample SD across {count_text} training seeds. UniMC is one frozen baseline.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.24, wspace=0.35)
    path = output_dir / "target_metrics.png"
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)
    return str(path.resolve())


def make_figures(rows, output_dir):
    """Write seed-0 reliability diagrams and task-macro summary PNGs.

    Reliability curves show a single run, while summary error bars measure
    variation across training seeds after averaging the four target tasks.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for row in rows:
        if row["split"] != "test" or row["dataset"] not in TARGETS or row["model"] not in MODELS:
            continue
        if row["stage"] not in STAGES:
            raise ValueError(f"Unexpected calibration stage: {row['stage']}")
        key = (row["model"], row["seed"], row["dataset"], row["stage"])
        if key in index:
            raise ValueError(f"Duplicate target metric row: {key}")
        index[key] = row

    seeds_by_model = {}
    for model in MODELS:
        seeds = {key[1] for key in index if key[0] == model}
        expected = {None} if model == "unimc" else {0, 1, 2}
        if seeds != expected:
            raise ValueError(f"Expected seeds {expected} for {model}; received {seeds}")
        seeds_by_model[model] = [None] if model == "unimc" else sorted(seeds)
        for seed in seeds_by_model[model]:
            for dataset in TARGETS:
                for stage in STAGES:
                    _prediction_row(index, model, seed, dataset, stage)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        paths = [_reliability_figure(index, model, output_dir) for model in MODELS]
        paths.append(_target_summary_figure(index, seeds_by_model, output_dir))
    return paths
