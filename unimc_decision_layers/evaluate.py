"""
evaluate.py -- aggregate all runs into the result tables of the project.

    python evaluate.py --runs runs --cache cache --out results

Produces
  results/table_acc.md / .csv      accuracy per dataset: baseline vs each kind (mean +- std over seeds)
  results/table_f1.md  / .csv      macro-F1 (over option index) in the same layout
  results/cost.md                  parameters, latency per 1000 samples, peak GPU memory of each decision layer
  results/per_run.csv              every run / every split
Macro-F1 is computed over the option index (0..M-1) of each dataset with sklearn; for SST-2
and CB this is the usual class-level macro-F1; for the QA tasks it is macro over positions.
"""
import argparse, csv, json, os, time, statistics as st
import torch
from sklearn.metrics import f1_score
from decision_layers import DecisionLayer, DecisionScorer

ORDER_DS = ["arc_c", "csqa", "obqa", "piqa", "sst2", "cb", "copa", "hellaswag"]
KINDS = list(DecisionLayer.KINDS)


def read_preds(path):
    rows = [json.loads(l) for l in open(path)]
    y = [r["label"] for r in rows]
    return y, [r["pred_raw"] for r in rows], [r["pred_new"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--cache", default="cache"); ap.add_argument("--out", default="results")
    ap.add_argument("--splits", nargs="*", default=["test", "test_official", "val"], help="which splits to tabulate")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    per = []   # (kind, seed, ds, split, acc_raw, acc_new, f1_raw, f1_new)
    for kind in KINDS:
        kd = os.path.join(args.runs, kind)
        if not os.path.isdir(kd):
            continue
        for sd in sorted(os.listdir(kd)):
            pd_ = os.path.join(kd, sd, "preds")
            if not os.path.isdir(pd_):
                continue
            for fn in sorted(os.listdir(pd_)):
                stem = fn[:-6]
                ds = max((d for d in ORDER_DS if stem.startswith(d + "_")), key=len, default=None)
                if ds is None:
                    continue
                sp = stem[len(ds) + 1:]
                y, pr, pn = read_preds(os.path.join(pd_, fn))
                per.append((kind, sd, ds, sp,
                            sum(a == b for a, b in zip(y, pr)) / len(y), sum(a == b for a, b in zip(y, pn)) / len(y),
                            f1_score(y, pr, average="macro"), f1_score(y, pn, average="macro")))
    with open(os.path.join(args.out, "per_run.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["kind", "seed", "dataset", "split", "acc_raw", "acc_new", "f1_raw", "f1_new"]); w.writerows(per)

    def table(metric_idx, name):
        cols = ["dataset", "split", "n_seeds", "UniMC baseline"] + KINDS
        lines, rows_csv = [], []
        for ds in ORDER_DS:
            for sp in args.splits:
                sub = [r for r in per if r[2] == ds and r[3] == sp]
                if not sub:
                    continue
                base = sub[0][metric_idx]
                row = [ds, sp, str(len({r[1] for r in sub})), f"{100*base:.2f}"]
                for k in KINDS:
                    v = [r[metric_idx + 1] for r in sub if r[0] == k]
                    if not v:
                        row.append("-"); continue
                    m = 100 * st.mean(v); s = 100 * st.stdev(v) if len(v) > 1 else 0.0
                    row.append(f"{m:.2f} ± {s:.2f}" if len(v) > 1 else f"{m:.2f}")
                lines.append(row); rows_csv.append(row)
        md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in lines)
        open(os.path.join(args.out, f"table_{name}.md"), "w").write(md + "\n")
        with open(os.path.join(args.out, f"table_{name}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols); w.writerows(rows_csv)
        print(f"\n== {name} ==\n" + md)
    table(4, "acc"); table(6, "f1")

    # ---- cost: params / latency / memory of the decision layer alone (on cached features) ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lines = ["| model | params | latency ms / 1000 samples | peak GPU MB |", "|---|---|---|---|", "| UniMC baseline (argmax of raw logits) | 0 | 0 | 0 |"]
    D = None
    for ds in ORDER_DS:
        p = os.path.join(args.cache, ds, "test.pt")
        if os.path.exists(p):
            D = torch.load(p, map_location="cpu"); break
    if D is not None:
        f, r, m = D["feats"][:1000].to(device), D["raw"][:1000].to(device), D["mask"][:1000].to(device)
        for kind in KINDS:
            p = os.path.join(args.runs, kind, "seed0", "best.pt")
            if not os.path.exists(p):
                continue
            sc = DecisionScorer.load(p, device)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            for _ in range(3):
                sc.score(f, r, m)
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                sc.score(f, r, m)
            if device == "cuda": torch.cuda.synchronize()
            ms = (time.time() - t0) / 10 * 1000 * (1000 / len(f))
            mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
            lines.append(f"| {kind} | {sc.model.n_params():,} | {ms:.1f} | {mem:.0f} |")
    open(os.path.join(args.out, "cost.md"), "w").write("\n".join(lines) + "\n")
    print("\n== cost ==\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
