"""
extract_features.py -- run the frozen official UniMC once and cache, per sample,
    option representations (M x 4096, fp16), raw "yes" logits (M), valid-option mask (M), label.

Usage (on the A40):
    python extract_features.py --model IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English \
        --data data --out cache --dtype bf16 --batch 32
    # quick test on 64 samples per split:   add  --max_samples 64

Output:  cache/<dataset>/<split>.pt  = dict(
            feats   : float16 tensor (N, M, 4096)   padded with 0
            raw     : float32 tensor (N, M)         padded with -1e4
            mask    : bool    tensor (N, M)
            label   : int64   tensor (N,)
            n_opt   : int64   tensor (N,)
            ids     : list[str]
            meta    : dict(model, dtype, max_length, hidden, version))
          cache/<dataset>/<split>.baseline.jsonl   official-style prediction per sample
          cache/summary.json                       baseline accuracy per split
"""
import argparse, json, os, time
import torch
from torch.utils.data import DataLoader, Dataset
from unimc_core import load_unimc, official_probs

VERSION = "cache-v1"


class JsonlDS(Dataset):
    def __init__(self, path, enc, max_samples=None):
        self.rows = [json.loads(l) for l in open(path, encoding="utf8")]
        if max_samples:
            self.rows = self.rows[:max_samples]
        self.enc = enc

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.enc.encode(self.rows[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, enc, model = load_unimc(args.model, device=device, dtype=dtype, max_length=args.max_length)
    print(f"model loaded: hidden={model.hidden_size}, yes_token={enc.yes_token}, device={device}, dtype={args.dtype}")

    summary = {}
    for ds in sorted(os.listdir(args.data)):
        if not os.path.isdir(os.path.join(args.data, ds)) or (args.only and ds not in args.only):
            continue
        for fn in sorted(os.listdir(os.path.join(args.data, ds))):
            if not fn.endswith(".jsonl"):
                continue
            split = fn[:-6]
            path = os.path.join(args.data, ds, fn)
            d = JsonlDS(path, enc, args.max_samples)
            dl = DataLoader(d, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=enc.collate)
            M = max(r["n_opt"] for r in [{"n_opt": len(x["choice"])} for x in d.rows])
            N = len(d)
            feats = torch.zeros(N, M, model.hidden_size, dtype=torch.float16)
            raw = torch.full((N, M), -1e4)
            mask = torch.zeros(N, M, dtype=torch.bool)
            labels = torch.tensor([r["label"] for r in d.rows])
            n_opt = torch.tensor([len(r["choice"]) for r in d.rows])
            t0, k = time.time(), 0
            for batch in dl:
                b = {kk: v.to(device) for kk, v in batch.items()}
                rep, rl = model(**b)
                B, m = rl.shape
                feats[k:k + B, :m] = rep.to(torch.float16).cpu()
                raw[k:k + B, :m] = rl.cpu()
                mask[k:k + B, :m] = b["option_mask"].cpu()
                k += B
            dt = time.time() - t0
            probs = official_probs(raw, mask)
            pred = probs.argmax(-1)
            acc = (pred == labels).float().mean().item()
            summary[f"{ds}/{split}"] = {"n": N, "baseline_acc": acc, "sec": dt}
            print(f"{ds:10s} {split:6s} n={N:6d}  baseline acc={acc:.4f}  ({dt:.0f}s, {N/dt:.1f} samples/s)")

            os.makedirs(os.path.join(args.out, ds), exist_ok=True)
            torch.save({"feats": feats, "raw": raw, "mask": mask, "label": labels, "n_opt": n_opt,
                        "ids": [r["id"] for r in d.rows],
                        "meta": {"model": args.model, "dtype": args.dtype, "max_length": args.max_length,
                                 "hidden": model.hidden_size, "version": VERSION, "yes_token": enc.yes_token}},
                       os.path.join(args.out, ds, f"{split}.pt"))
            with open(os.path.join(args.out, ds, f"{split}.baseline.jsonl"), "w", encoding="utf8") as f:
                for i, r in enumerate(d.rows):
                    f.write(json.dumps({"id": r["id"], "label": r["label"], "pred": int(pred[i]),
                                        "raw_logits": raw[i, :n_opt[i]].tolist(),
                                        "probs": probs[i, :n_opt[i]].tolist()}) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
