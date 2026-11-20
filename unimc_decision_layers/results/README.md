# results/

Tables written by `evaluate.py` on 2026-11-18 (4 decision-layer kinds x seeds 0/1/2, NVIDIA A40):
`table_acc.*` (accuracy), `table_f1.*` (macro-F1), `cost.md` (params / latency / memory), `per_run.csv` (every run and split).

Related files in this folder's parent: `runs/` (weights `best.pt`, `log.csv`, per-sample logits `preds/*.jsonl` for every
split incl. `*_calib.jsonl` for temperature scaling), `runs_check_*/` (sanity checks, see RESULTS §6), `cache/summary.json`
(frozen-UniMC baseline per split), `extract.log` / `train.log`, `data/splits.json` (train/val/calib membership).
The 1.4 GB feature cache and the prepared datasets are not in Git: `prepare_data.py` + `extract_features.py` regenerate them.
Interpretation: `../RESULTS_2026-11-18.md`.
