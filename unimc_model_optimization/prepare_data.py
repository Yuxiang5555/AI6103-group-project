"""Convert local public data exports to a fixed, versioned UniMC input format.

Source train is never split implicitly. Supply a separate source holdout pool;
it is split into source_val/source_cal. Check its exposure to the old checkpoint.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def read_records(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    return json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]


def get_choices(value):
    if isinstance(value, dict):
        return list(value["text"]), list(value["label"])
    if value and isinstance(value[0], dict):
        return [x["text"] for x in value], [x["label"] for x in value]
    return list(value), [str(i) for i in range(len(value))]


def convert(row, task, index, external_label=None):
    question = ""
    if task == "canonical":
        text = row["texta"]
        question = row.get("question", "")
        choices = row.get("choices", row.get("choice"))
        gold = row.get("gold_label", row.get("label"))
        dataset = row["dataset"]
    elif task == "piqa":
        text, choices = row["goal"], [row["sol1"], row["sol2"]]
        gold = external_label if external_label is not None else row.get("label")
        dataset = task
    elif task in {"commonsenseqa", "arc", "openbookqa"}:
        rawq = row.get("question", row.get("question_stem"))
        if isinstance(rawq, dict):
            text, raw_choices = rawq["stem"], rawq["choices"]
        else:
            text, raw_choices = rawq, row["choices"]
        choices, keys = get_choices(raw_choices)
        answer = row.get("answerKey")
        if answer is None or str(answer) not in [str(k) for k in keys]:
            raise ValueError(f"{task} example {index}: missing/invalid answerKey; use a labeled split")
        gold = [str(k) for k in keys].index(str(answer))
        dataset = task
    elif task == "sst2":
        text = row["sentence"]
        question = "What is the sentiment of the review?"
        choices = ["it's great.", "it's terrible."]
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("SST-2 requires a publicly labeled split")
        gold, dataset = 1 - label, task
    elif task == "cb":
        text, hypothesis = row["premise"], row["hypothesis"]
        question = "Based on the paragraph."
        choices = [f"We can infer that {hypothesis}", f"We can not infer that {hypothesis}",
                   f"It is difficult for us to infer that {hypothesis}"]
        gold, dataset = row["label"], task  # HF SuperGLUE: entailment, contradiction, neutral
    elif task == "copa":
        text, choices = row["premise"], [row["choice1"], row["choice2"]]
        if row["question"] not in {"cause", "effect"}:
            raise ValueError("COPA question must be cause/effect")
        question = "What was the cause?" if row["question"] == "cause" else "What happened as a result?"
        gold, dataset = row["label"], task
    elif task == "hellaswag":
        text = row.get("ctx", (row.get("ctx_a", "") + " " + row.get("ctx_b", "")).strip())
        choices, gold, dataset = row["endings"], row["label"], task
    else:
        raise ValueError(task)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing text in {task} example {index}")
    if not isinstance(choices, list) or len(choices) < 2 or any(not isinstance(c, str) or not c.strip() for c in choices):
        raise ValueError(f"Invalid choices in {task} example {index}")
    if gold is None:
        raise ValueError(f"Missing gold label in {task} example {index}; use a labeled split")
    gold = int(gold)
    if not 0 <= gold < len(choices):
        raise ValueError(f"Invalid gold label in {task} example {index}: {gold}")
    record = {"id": str(row.get("id", row.get("idx", row.get("ind", index)))), "dataset": dataset,
              "texta": text, "textb": row.get("textb", "") if task == "canonical" else "",
              "question": question, "choices": choices, "gold_label": gold}
    payload = {k: record[k] for k in ("texta", "textb", "question", "choices")}
    record["content_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return record


def load_and_convert(path, task, labels_path=None):
    raw = read_records(path)
    labels = None if labels_path is None else [int(x) for x in Path(labels_path).read_text().splitlines() if x.strip()]
    if labels is not None and len(labels) != len(raw):
        raise ValueError("External label file length does not match examples")
    result, seen = [], {}
    for i, row in enumerate(raw):
        item = convert(row, task, i, None if labels is None else labels[i])
        if item["content_hash"] in seen:
            if seen[item["content_hash"]] != item["gold_label"]:
                raise ValueError(f"Conflicting gold labels for duplicate question at {path}:{i}")
            continue  # exact duplicate questions are kept once, before splitting
        seen[item["content_hash"]] = item["gold_label"]
        result.append(item)
    return result


def write_records(path, rows, role, origin):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            # Official files sometimes have only per-split row numbers as IDs.
            row = dict(row, id=f"{origin}:{row['id']}", role=role)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=["canonical", "piqa", "commonsenseqa", "arc", "openbookqa", "sst2", "cb", "copa", "hellaswag"])
    p.add_argument("--input", required=True)
    p.add_argument("--labels", help="PIQA raw train-labels.lst, if labels are separate")
    p.add_argument("--holdout", help="Separate labeled source pool, split into validation and calibration")
    p.add_argument("--holdout-labels")
    p.add_argument("--role", choices=["source", "target"], required=True)
    p.add_argument("--output", required=True, help="New output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.5)
    p.add_argument("--limit-train", type=int, help="Optional pilot size, chosen before training")
    args = p.parse_args()
    rows = load_and_convert(args.input, args.task, args.labels)
    rng = random.Random(args.seed)
    if args.role == "source":
        if args.task in {"sst2", "cb", "copa", "hellaswag"}:
            raise ValueError("These tasks are reserved for final target evaluation in this project")
        if not args.holdout:
            raise ValueError("Supply --holdout; the program does not assume old checkpoint train data are held out")
        holdout = load_and_convert(args.holdout, args.task, args.holdout_labels)
        if {x["content_hash"] for x in rows} & {x["content_hash"] for x in holdout}:
            raise ValueError("Source train and holdout question contents overlap")
        if not 0 < args.val_fraction < 1 or len(holdout) < 2:
            raise ValueError("Need 0 < val-fraction < 1 and at least two holdout examples")
        rng.shuffle(holdout)
        cut = max(1, min(len(holdout) - 1, round(len(holdout) * args.val_fraction)))
        if args.limit_train is not None:
            if args.limit_train < 1:
                raise ValueError("limit-train must be positive")
            rng.shuffle(rows)
            rows = rows[:args.limit_train]
        splits = {"train": (rows, "source_train", "train"),
                  "val": (holdout[:cut], "source_val", "holdout"),
                  "cal": (holdout[cut:], "source_cal", "holdout")}
    else:
        if args.holdout:
            raise ValueError("Target evaluation does not use a calibration/selection holdout")
        splits = {"eval": (rows, "target_eval", "eval")}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for name, (items, role, origin) in splits.items():
        write_records(out / f"{name}.jsonl", items, role, origin)
        summary[name] = {"n": len(items), "role": role,
                         "option_counts": dict(Counter(len(x["choices"]) for x in items))}
    manifest = {"task": args.task, "arguments": vars(args), "splits": summary,
                "template_version": "project-v1-fixed",
                "exposure_note": "Independent of the NEW head; old backbone exposure must be audited separately.",
                "template_note": "These are fixed project templates, not a claim of exact original-paper template reproduction."}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
