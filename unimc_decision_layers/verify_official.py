"""
verify_official.py -- optional: compare our re-implementation with the official fengshen pipeline
on a few hundred samples.  Needs the official package:
    git clone --depth 1 https://github.com/IDEA-CCNL/Fengshenbang-LM.git
    pip install pytorch_lightning==1.9.5    # fengshen's UniMC code uses the PL 1.x API
    PYTHONPATH=Fengshenbang-LM python verify_official.py --data data/sst2/test.jsonl --n 200
It prints the agreement rate of the predicted option and the max abs diff of the probabilities.
If the official import fails (dependency mismatch), skip this: unimc_core.py already copies the
official preprocessing line by line, and the baseline accuracies in cache/summary.json are the
practical check.
"""
import argparse, json, torch
from unimc_core import load_unimc, official_probs

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English")
ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.data)][:a.n]

# ours
tok, enc, model = load_unimc(a.model, "cuda", torch.float32)
ours = []
for i in range(0, len(rows), 16):
    b = enc.collate([enc.encode(r) for r in rows[i:i+16]])
    b = {k: v.cuda() for k, v in b.items()}
    _, raw = model(**b)
    ours.append(official_probs(raw, b["option_mask"]).cpu())
ours = torch.cat(ours)

# official
from fengshen.pipelines.multiplechoice import UniMCPipelines
p = argparse.ArgumentParser(); p = UniMCPipelines.piplines_args(p)
args = p.parse_args([]); args.language = "english"; args.max_length = 512; args.batchsize = 16; args.default_root_dir = "./"
pipe = UniMCPipelines(args, model_path=a.model)
res = pipe.predict([dict(r) for r in rows])
agree, maxdiff = 0, 0.0
for i, (r, o) in enumerate(zip(res, rows)):
    off = torch.tensor([r["score"][c] for c in o["choice"]])
    agree += int(off.argmax() == ours[i, :len(o["choice"])].argmax())
    maxdiff = max(maxdiff, (off - ours[i, :len(o["choice"])]).abs().max().item())
print(f"agreement {agree}/{len(rows)}   max |prob diff| = {maxdiff:.4f}")
