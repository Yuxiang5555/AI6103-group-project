# Trained heads and LoRA adapters

This folder contains 24 small checkpoints copied byte-for-byte from the audited
local experiment exports. It does not contain the original UniMC backbone.

| Folder | Contents | Purpose |
|---|---|---|
| `heads/` | 18 files: centered/uncentered × MLP/self/full × seeds 42/43/44 | Main candidates and structural controls |
| `gate/` | `gated_seed42.pt`, `gated_seed43.pt`, `gated_seed44.pt` | Interaction-Gate ablation |
| `lora/seed42/` | `initial_adapter.pt`, `best_adapter.pt`, `last_adapter.pt` | Initialization, selected epoch 1, and final epoch 2 |

## Which files to use

The fixed main candidates are `heads/centered_full_seed42.pt`,
`heads/centered_full_seed43.pt`, and `heads/centered_full_seed44.pt`.
They remain candidates for independent evaluation, not confirmed accuracy gains.
Gate did not improve mean accuracy over full attention. The LoRA best adapter is
the best **trained** checkpoint of a negative source experiment; it did not exceed
the original UniMC baseline. Initial/final LoRA files are retained for diagnostics.

These checkpoints are uncalibrated modules. The repository's `calibration/`
deliverable was fitted for the separate v0.3 `unimc_decision_layers` models;
its temperature parameters are not fitted to this collection.

All file sizes, SHA256 digests, epochs, source-validation metrics, and feature
signatures are in [manifest.json](manifest.json). Files are not renamed internally
or reserialized, and their original `output/` backups remain separate.

## Load the matching model path

Run from `unimc_model_optimization/`. Heads require feature caches produced with
the matching official backbone, projection, and centering signatures. The raw
`unimc_decision_layers` cache format cannot be passed directly to these loaders.

```bash
# Centered full head: cache must already contain compatible centered MLM features.
python predict_head.py --cache cache/centered/TASK.pt \
  --checkpoint models/heads/centered_full_seed42.pt \
  --output predictions/TASK_centered_full_seed42.jsonl --device cuda:0

# Interaction Gate uses the dedicated loader and the same centered representation.
python predict_interaction_gate.py --cache cache/centered/TASK.pt \
  --checkpoint models/gate/gated_seed42.pt \
  --output predictions/TASK_gated_seed42.jsonl --device cuda:0

# LoRA runs online from canonical text input, not from frozen feature caches.
python predict_unimc_lora.py --input data/TASK.jsonl --role target_eval \
  --checkpoint models/lora/seed42/best_adapter.pt \
  --output predictions/TASK_lora_seed42.jsonl --device cuda:0
```

Replace `TASK` with actual new paths; output files must not already exist.
Follow [the operations guide](../docs/OPERATIONS.md#reproduction) for the external
base model and verified Fengshen checkout. The external backbone defaults to
`../models/unimc-albert-235m/`, which is a different directory from this one.

## Verification performed when adding this folder

Every copied file matches its source SHA256. All 21 residual/Gate heads load
strictly and produce finite valid logits with padded options excluded in a small
CPU probe. All three LoRA adapters have finite tensors, the expected dimensions
and parameter count, and matching implementation signatures. The full UniMC
backbone was not replayed and no new training or model selection was performed.
