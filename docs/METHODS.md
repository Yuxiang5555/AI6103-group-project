# Model-optimization methods

## Official encoding and baseline

Use the official custom UniMC dataset encoder, tokenizer, position IDs, and attention masks. A complete question contains its candidates in the official structured input; it is not equivalent to independently invoking an ordinary encoder on each candidate. The baseline score is the original yes-token logit at each option marker. Pretrained exposure to the pilot tasks was not established; source-split disjointness concerns training the new modules.

## Frozen-backbone scoring heads

`extract_features.py` stores each option-marker state and its original score while the backbone is in evaluation mode. Raw states have dimension 4096. `decision_head.py` implements MLP, self-only attention, and full option attention without option-position embeddings. The correction head is zero-initialized, and final scores equal baseline scores plus the learned correction. Masks exclude padded options; tests cover permutation equivariance, finite gradients, and all-padding rejection.

The original raw-state model used a 256-dimensional projection. Later compact models use dimension 64. `project_mlm_features.py` applies the original frozen MLM dense/activation/LayerNorm transformation to produce 128-dimensional predecoder states. It preserves the original baseline scores rather than approximating them from cached states.

## Option centering and controls

For question q with K actual candidates, `center_option_features.py` subtracts the mean of the K candidate vectors from every vector. Padding is excluded, and labels are not used. The centered representation is saved as FP32.

Centering introduces candidate-set dependence even for MLP and self-only heads. Therefore, centered full versus centered self/MLP measures the incremental role of cross-option attention under a common set-based representation. Uncentered controls are retained separately. Three-seed studies use seeds 42, 43, and 44, common inputs, and matched initialization where required.

## Interaction Gate

`interaction_gate_head.py` computes self-only and full attention from shared parameters and interpolates their attention outputs:

```text
m = m_self + sigmoid(alpha) * (m_full - m_self)
```

The learned global scalar starts at 0.5 and adds one trainable parameter. With a nonzero scorer, Gate endpoints are checked against the self/full architectures. This differs from the older `train_head.py --gate` flag, which scales the entire correction. The real interaction-Gate experiment used centered MLM features and a 64-dimensional head, not the initial 256-dimensional architecture.

## LoRA adaptation

`unimc_lora.py` adds `(alpha/r) * B * A * dropout(x)` to frozen query/value projections. Only A/B are trainable; B starts at zero. ALBERT's logical layers reuse physical modules, so adapters are injected once per physical projection. For the recorded base, rank 8 yields two modules and 131,072 parameters. Non-reentrant gradient checkpointing supports frozen input embeddings.

`unimc_online.py` reuses official online encoding and loss computation. It deep-copies choices before the upstream encoder can shorten them and corrects padding MLM targets to −100. The native objective is OP plus token-averaged MLM, comprising option-marker yes/no supervision and passage supervision. This differs from training residual heads only on frozen cached states.

`run_unimc_lora.py` performs a real feasibility check before training, verifies zero-adapter baseline parity, and saves adapters separately from the base. Prediction requires the exact backbone/encoding and implementation signatures. No current LoRA result is combined with the centered residual head.

## Read-only loss diagnostics

`diagnose_unimc_lora.py` fixes 16 source-training questions per task and compares initial, best, and final adapters on identical sampled masks and dropout seeds. The formal diagnostic is FP32 and has zero optimizer steps. Option and passage MLM averages are weighted by their target counts to reconstruct the original MLM reduction. Component gradients are measured only with respect to A/B. Per-example cosine averages are not aggregate-batch cosine measurements.

## Selection and downstream boundary

Head training uses source training/validation only, with macro accuracy for selection and NLL as tie-breaker. Source calibration remains separate. The handoff tools export uncalibrated scores for downstream calibration; they do not fit calibration or rejection models. No independent-target evaluation result is included in this repository.
