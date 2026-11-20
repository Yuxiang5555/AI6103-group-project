"""
smoke_test.py -- end-to-end check WITHOUT the real model / GPU.

1. builds a fake feature cache in the exact format extract_features.py writes
   (random 4096-d option vectors with a planted signal + random raw logits)
2. trains every decision-layer kind for a few epochs
3. exports predictions, runs evaluate.py, checks the inference interface

    python smoke_test.py            # ~1 minute on CPU
"""
import os, shutil, subprocess, sys, torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke")
CACHE, RUNS, RES = (os.path.join(ROOT, d) for d in ("cache", "runs", "results"))
shutil.rmtree(ROOT, ignore_errors=True)
H = 64   # small hidden size for speed (real one is 4096; code is size-agnostic)
g = torch.Generator().manual_seed(0)


def fake_split(ds, split, n, m):
    feats = torch.randn(n, m, H, generator=g)
    label = torch.randint(0, m, (n,), generator=g)
    # planted signal: correct option gets +1 on feature 0 -> a trained layer must beat random
    feats[torch.arange(n), label, 0] += 1.0
    raw = torch.randn(n, m, generator=g)
    raw[torch.arange(n), label] += 0.5          # baseline is better than chance but weak
    mask = torch.ones(n, m, dtype=torch.bool)
    if m > 2:                                   # make some samples have fewer options
        k = n // 4
        mask[:k, -1] = False; feats[:k, -1] = 0; raw[:k, -1] = -1e4
        label[:k] = label[:k].clamp(max=m - 2)
    os.makedirs(os.path.join(CACHE, ds), exist_ok=True)
    torch.save({"feats": feats.half(), "raw": raw, "mask": mask, "label": label, "n_opt": mask.sum(-1),
                "ids": [f"{ds}-{split}-{i}" for i in range(n)],
                "meta": {"model": "fake", "dtype": "fp32", "max_length": 512, "hidden": H, "version": "cache-v1", "yes_token": 0}},
               os.path.join(CACHE, ds, f"{split}.pt"))


for ds, m in [("arc_c", 4), ("csqa", 5), ("obqa", 4), ("piqa", 2)]:
    fake_split(ds, "train", 800, m); fake_split(ds, "val", 200, m); fake_split(ds, "calib", 100, m)
    if ds in ("arc_c", "obqa"):
        fake_split(ds, "test", 200, m)
for ds, m in [("sst2", 2), ("cb", 3), ("copa", 2), ("hellaswag", 4)]:
    fake_split(ds, "test", 150, m)

here = os.path.dirname(os.path.abspath(__file__))
run = lambda *a: subprocess.run([sys.executable, *a], cwd=here, check=True)
run("train_decision.py", "--kind", "all", "--seeds", "0", "1", "--cache", CACHE, "--out", RUNS, "--epochs", "6", "--batch", "64", "--d", "32", "--ff", "64")
run("evaluate.py", "--runs", RUNS, "--cache", CACHE, "--out", RES)

# inference interface
from decision_layers import DecisionScorer
sc = DecisionScorer.load(os.path.join(RUNS, "tf_int", "seed0", "best.pt"))
D = torch.load(os.path.join(CACHE, "arc_c", "test.pt"))
lg = sc.score(D["feats"][:5], D["raw"][:5], D["mask"][:5])
print("\ninterface ok, logits of 5 samples:\n", lg)
single = sc.score(D["feats"][0], D["raw"][0], D["mask"][0])
assert torch.allclose(single, lg[0]), "single-sample path differs"
print("\nSMOKE TEST PASSED")
