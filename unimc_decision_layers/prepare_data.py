"""
prepare_data.py -- download the 8 datasets, convert to the UniMC input format, fix the splits.

Output:  data/<dataset>/<split>.jsonl   one json per line:
   {"id": "<dataset>-<split>-<k>", "dataset": ..., "split": ...,
    "texta": str, "textb": "", "question": str, "choice": [str, ...], "label": int}

Splits
  source tasks (used to train the new decision layers):
     ARC-Challenge, CommonsenseQA, OpenBookQA, PIQA
        train  = 80 % of the official train split      (trains the new layers)
        val    = 10 % of the official train split      (model selection / early stopping)
        calib  = 10 % of the official train split      (reserved for temperature scaling, member D)
        test   = the official validation split, untouched  -> in-domain test
        test_official = official test split where it has labels (ARC-C, OBQA), extra in-domain test
  target tasks (never used for training / tuning / temperature):
     SST-2, CB, COPA, HellaSwag   -> split "test" = the official labelled evaluation split
The train/val/calib split is deterministic (seed 0) and is written to data/splits.json.

Prompt templates follow the UniMC paper (EMNLP 2022), Table 9 and Figure 3:
   NLI (CB)   : question "Base on the paragraph."  options "We can infer that x2" /
                "We can not infer that x2" / "It is difficult for us to infer x2"
   Sentiment  : question "What is sentiment of reviews?"  options "It's great." / "It's terrible."
   Hellaswag  : no question; passage = context, options = the 4 endings (Figure 3)
   COPA       : not shown in the paper; we use the cause/effect question wording below.
Keep them fixed for all models so that comparisons are fair.
"""
import argparse, json, os, random
from datasets import load_dataset

TEMPLATES = {
    "sst2":      {"question": "What is sentiment of reviews?",
                  "choice": ["It's great.", "It's terrible."]},          # paper order; sst2 label 1=positive -> idx 0
    "cb":        {"question": "Base on the paragraph.",
                  "choice_fmt": ["We can infer that {h}", "We can not infer that {h}", "It is difficult for us to infer {h}"]},
                  # super_glue cb labels: 0 entailment, 1 contradiction, 2 neutral  (same order as the options)
    "copa":      {"question": {"cause": "What was the cause of this?", "effect": "What happened as a result?"}},
    "hellaswag": {"question": ""},
}


def dump(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):6d} -> {path}")


def mk(ds, split, k, texta, question, choice, label, textb=""):
    return {"id": f"{ds}-{split}-{k}", "dataset": ds, "split": split,
            "texta": texta, "textb": textb, "question": question, "choice": list(choice), "label": int(label)}


def letter_to_idx(labels, key):
    labels = list(labels)
    if key in labels:
        return labels.index(key)
    # some ARC items use "1".."4" instead of "A".."D"
    alt = {"1": "A", "2": "B", "3": "C", "4": "D", "A": "1", "B": "2", "C": "3", "D": "4"}
    return labels.index(alt[key])


def load_any(name, config=None, split=None):
    """load_dataset with fallbacks for repos that still use loading scripts."""
    tries = [dict(), dict(revision="refs/convert/parquet"), dict(trust_remote_code=True)]
    last = None
    for kw in tries:
        try:
            return load_dataset(name, config, split=split, **kw) if config else load_dataset(name, split=split, **kw)
        except Exception as e:          # noqa
            last = e
    raise last


# ---------------- source tasks ----------------
def arc():
    rows = {}
    for sp, out in [("train", "train"), ("validation", "test"), ("test", "test_official")]:
        d = load_any("allenai/ai2_arc", "ARC-Challenge", sp)
        rows[out] = [mk("arc_c", out, k, "", x["question"], x["choices"]["text"],
                       letter_to_idx(x["choices"]["label"], x["answerKey"])) for k, x in enumerate(d)]
    return rows


def csqa():
    rows = {}
    for sp, out in [("train", "train"), ("validation", "test")]:
        d = load_any("tau/commonsense_qa", None, sp)
        rows[out] = [mk("csqa", out, k, "", x["question"], x["choices"]["text"],
                       letter_to_idx(x["choices"]["label"], x["answerKey"])) for k, x in enumerate(d)]
    return rows


def obqa():
    rows = {}
    for sp, out in [("train", "train"), ("validation", "test"), ("test", "test_official")]:
        d = load_any("allenai/openbookqa", "main", sp)
        rows[out] = [mk("obqa", out, k, "", x["question_stem"], x["choices"]["text"],
                       letter_to_idx(x["choices"]["label"], x["answerKey"])) for k, x in enumerate(d)]
    return rows


def piqa():
    rows = {}
    for sp, out in [("train", "train"), ("validation", "test")]:
        d = load_any("ybisk/piqa", None, sp)
        rows[out] = [mk("piqa", out, k, "", x["goal"], [x["sol1"], x["sol2"]], x["label"]) for k, x in enumerate(d)]
    return rows


# ---------------- target tasks (evaluation only) ----------------
def sst2():
    d = load_any("stanfordnlp/sst2", None, "validation")
    t = TEMPLATES["sst2"]
    # sst2: label 1 = positive -> option 0 "It's great.";  label 0 = negative -> option 1 "It's terrible."
    return {"test": [mk("sst2", "test", k, x["sentence"], t["question"], t["choice"], 1 - x["label"]) for k, x in enumerate(d)]}


def cb():
    d = load_any("aps/super_glue", "cb", "validation")
    fmts = TEMPLATES["cb"]["choice_fmt"]
    return {"test": [mk("cb", "test", k, x["premise"], TEMPLATES["cb"]["question"],
                        [f.format(h=x["hypothesis"]) for f in fmts], x["label"]) for k, x in enumerate(d)]}


def copa():
    d = load_any("aps/super_glue", "copa", "validation")
    q = TEMPLATES["copa"]["question"]
    return {"test": [mk("copa", "test", k, x["premise"], q[x["question"]], [x["choice1"], x["choice2"]], x["label"])
                     for k, x in enumerate(d)]}


def hellaswag():
    d = load_any("Rowan/hellaswag", None, "validation")
    return {"test": [mk("hellaswag", "test", k, x["ctx"], TEMPLATES["hellaswag"]["question"], x["endings"], int(x["label"]))
                     for k, x in enumerate(d)]}


SOURCE = {"arc_c": arc, "csqa": csqa, "obqa": obqa, "piqa": piqa}
TARGET = {"sst2": sst2, "cb": cb, "copa": copa, "hellaswag": hellaswag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--calib_frac", type=float, default=0.1)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="subset of dataset names")
    args = ap.parse_args()

    splits_record = {}
    for name, fn in {**SOURCE, **TARGET}.items():
        if args.only and name not in args.only:
            continue
        print(f"[{name}]")
        rows = fn()
        if name in SOURCE:
            tr = rows.pop("train")
            rng = random.Random(args.seed)
            idx = list(range(len(tr)))
            rng.shuffle(idx)
            n_cal = int(round(len(idx) * args.calib_frac))
            n_val = int(round(len(idx) * args.val_frac))
            cal_idx = sorted(idx[:n_cal]); val_idx = sorted(idx[n_cal:n_cal + n_val]); tr_idx = sorted(idx[n_cal + n_val:])
            rows["calib"] = [tr[i] for i in cal_idx]
            rows["val"] = [tr[i] for i in val_idx]
            rows["train"] = [tr[i] for i in tr_idx]
            for sp in ("calib", "val"):
                for r in rows[sp]:
                    r["split"] = sp
            splits_record[name] = {"seed": args.seed, "calib_frac": args.calib_frac, "val_frac": args.val_frac,
                                   "calib_ids_in_official_train": cal_idx, "val_ids_in_official_train": val_idx}
        for sp, rs in rows.items():
            dump(rs, os.path.join(args.out, name, f"{sp}.jsonl"))
    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump(splits_record, f)
    print("done.")


if __name__ == "__main__":
    main()
