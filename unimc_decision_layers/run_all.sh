#!/bin/bash
# Full pipeline on one A40.  Run from the unimc_opt folder.  Times are rough estimates.
set -e
pip install -q -r requirements.txt

# 0. quick self-test without the model (1 min)
python smoke_test.py

# 1. datasets -> data/<name>/<split>.jsonl  (a few minutes, needs internet)
python prepare_data.py --out data

# 2. official UniMC, frozen -> cache/<name>/<split>.pt   (~30-60 min)
#    first try a tiny run to catch environment problems:
python extract_features.py --data data --out cache_test --max_samples 32 --batch 8 --dtype fp32 --only sst2 copa
python extract_features.py --data data --out cache --dtype bf16 --batch 32

# 3. train the four decision layers, 3 seeds each (< 1 h total; features are cached)
python train_decision.py --kind all --seeds 0 1 2 --cache cache --out runs

# 4. tables
python evaluate.py --runs runs --cache cache --out results

# 5. end-to-end check with text input
python pipeline.py --ckpt runs/tf_int/seed0/best.pt
