"""
train_decision.py -- train one decision layer on the cached UniMC features.

    python train_decision.py --kind tf_int --cache cache --out runs --seed 0
    python train_decision.py --kind all   --seeds 0 1 2            # 4 kinds x 3 seeds

Training data  : the "train" split of every source dataset (arc_c, csqa, obqa, piqa), mixed.
Model selection: accuracy on the mixed "val" split of the same source datasets
                 (never on a target task).  The epoch with the best source-val accuracy
                 is saved as best.pt.
Loss           : cross-entropy over the valid options of each sample (padding = -inf).
After training : the best model is applied to EVERY cached split of EVERY dataset and the
                 per-sample logits are written to runs/<kind>/seed<k>/preds/<dataset>_<split>.jsonl
                 (fields: id, label, raw_logits, new_logits, pred_raw, pred_new) -- this is
                 what the calibration (D) and evaluation (E) members consume.
"""
import argparse, csv, json, os, random, time
import torch
import torch.nn.functional as F
from decision_layers import DecisionLayer

SOURCE = ["arc_c", "csqa", "obqa", "piqa"]


def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_split(cache, ds, split):
    p = os.path.join(cache, ds, f"{split}.pt")
    return torch.load(p, map_location="cpu") if os.path.exists(p) else None


def concat(parts, M=None):
    """concatenate cached splits with different option counts -> pad to common M."""
    M = M or max(p["feats"].shape[1] for p in parts)
    N = sum(p["feats"].shape[0] for p in parts)
    H = parts[0]["feats"].shape[2]
    feats = torch.zeros(N, M, H, dtype=torch.float16)
    raw = torch.full((N, M), -1e4)
    mask = torch.zeros(N, M, dtype=torch.bool)
    label = torch.zeros(N, dtype=torch.long)
    ids, src = [], []
    k = 0
    for p in parts:
        n, m = p["feats"].shape[:2]
        feats[k:k + n, :m] = p["feats"]; raw[k:k + n, :m] = p["raw"]; mask[k:k + n, :m] = p["mask"]
        label[k:k + n] = p["label"]; ids += p["ids"]; src += [i.split("-")[0] for i in p["ids"]]
        k += n
    return dict(feats=feats, raw=raw, mask=mask, label=label, ids=ids, src=src)


@torch.no_grad()
def evaluate(model, D, device, bs=1024):
    model.eval()
    out, correct, loss = [], 0, 0.0
    for i in range(0, len(D["label"]), bs):
        f, r, m, y = (D[k][i:i + bs].to(device) for k in ("feats", "raw", "mask", "label"))
        lg = model(f, r, m)
        loss += F.cross_entropy(lg, y, reduction="sum").item()
        correct += (lg.argmax(-1) == y).sum().item()
        out.append(lg.cpu())
    n = len(D["label"])
    return correct / n, loss / n, torch.cat(out)


def train_one(kind, seed, args, device, train, val):
    set_seed(seed)
    model = DecisionLayer(kind, in_dim=train["feats"].shape[2], d=args.d, heads=args.heads, ff=args.ff,
                          dropout=args.dropout, use_raw_feat=not args.no_raw_feat, residual=not args.no_residual).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n = len(train["label"])
    steps_per_epoch = (n + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    out_dir = os.path.join(args.out, kind, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    w = csv.writer(log); w.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "sec"])
    best = {"val_acc": -1}
    # whole training set on the device (fits easily: ~1 GB fp16)
    tr = {k: train[k].to(device) for k in ("feats", "raw", "mask", "label")}
    print(f"[{kind} seed{seed}] params={model.n_params():,}  train n={n}  val n={len(val['label'])}")
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        perm = torch.randperm(n, device=device)
        tl, tc = 0.0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            lg = model(tr["feats"][idx], tr["raw"][idx], tr["mask"][idx])
            y = tr["label"][idx]
            loss = F.cross_entropy(lg, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item() * len(idx); tc += (lg.argmax(-1) == y).sum().item()
        va, vl, _ = evaluate(model, val, device)
        w.writerow([ep, f"{opt.param_groups[0]['lr']:.2e}", f"{tl/n:.4f}", f"{tc/n:.4f}", f"{vl:.4f}", f"{va:.4f}", f"{time.time()-t0:.1f}"]); log.flush()
        print(f"  ep{ep:3d} train {tl/n:.4f}/{tc/n:.4f}  val {vl:.4f}/{va:.4f}")
        if va > best["val_acc"]:
            best = {"val_acc": va, "val_loss": vl, "epoch": ep}
            torch.save({"cfg": model.cfg, "state_dict": model.state_dict(), "best": best,
                        "args": vars(args), "seed": seed}, os.path.join(out_dir, "best.pt"))
    log.close()
    return out_dir, best


@torch.no_grad()
def export_preds(ckpt_dir, cache, device):
    from decision_layers import DecisionScorer
    sc = DecisionScorer.load(os.path.join(ckpt_dir, "best.pt"), device)
    os.makedirs(os.path.join(ckpt_dir, "preds"), exist_ok=True)
    res = {}
    for ds in sorted(os.listdir(cache)):
        if not os.path.isdir(os.path.join(cache, ds)):
            continue
        for fn in sorted(os.listdir(os.path.join(cache, ds))):
            if not fn.endswith(".pt"):
                continue
            sp = fn[:-3]; D = torch.load(os.path.join(cache, ds, fn), map_location="cpu")
            lg = torch.cat([sc.score(D["feats"][i:i + 1024], D["raw"][i:i + 1024], D["mask"][i:i + 1024]).cpu()
                            for i in range(0, len(D["label"]), 1024)])
            pr, pn = D["raw"].masked_fill(~D["mask"], float("-inf")).argmax(-1), lg.argmax(-1)
            with open(os.path.join(ckpt_dir, "preds", f"{ds}_{sp}.jsonl"), "w") as f:
                for i, id_ in enumerate(D["ids"]):
                    m = int(D["n_opt"][i])
                    f.write(json.dumps({"id": id_, "label": int(D["label"][i]), "raw_logits": D["raw"][i, :m].tolist(),
                                        "new_logits": lg[i, :m].tolist(), "pred_raw": int(pr[i]), "pred_new": int(pn[i])}) + "\n")
            res[f"{ds}/{sp}"] = {"n": len(D["label"]), "acc_raw": (pr == D["label"]).float().mean().item(),
                                 "acc_new": (pn == D["label"]).float().mean().item()}
    with open(os.path.join(ckpt_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="tf_int", help="mlp | tf_noint | tf_int | tf_gate | all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--cache", default="cache"); ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=512); ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no_raw_feat", action="store_true"); ap.add_argument("--no_residual", action="store_true")
    ap.add_argument("--source", nargs="*", default=SOURCE)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_parts = [load_split(args.cache, d, "train") for d in args.source]
    va_parts = [load_split(args.cache, d, "val") for d in args.source]
    tr_parts = [p for p in tr_parts if p]; va_parts = [p for p in va_parts if p]
    assert tr_parts and va_parts, "no cached train/val splits found"
    M = max(p["feats"].shape[1] for p in tr_parts + va_parts)
    train, val = concat(tr_parts, M), concat(va_parts, M)
    kinds = DecisionLayer.KINDS if args.kind == "all" else [args.kind]
    table = {}
    for kind in kinds:
        for s in args.seeds:
            d, best = train_one(kind, s, args, device, train, val)
            res = export_preds(d, args.cache, device)
            table[f"{kind}/seed{s}"] = {"best": best, **{k: v["acc_new"] for k, v in res.items()}}
            print(f"  best val acc {best['val_acc']:.4f} @ep{best['epoch']}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train_table.json"), "w") as f:
        json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()
