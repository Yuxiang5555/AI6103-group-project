"""Train the new attention-interaction gate with the legacy source-training protocol."""
import json
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_io import (FeatureDataset, assert_disjoint, collate_features, evaluate,
                     file_sha256, move_batch, seed_everything, select_device, write_json)
from decision_head import HeadConfig
from interaction_gate_head import InteractionGatedOptionHead, MODEL_KIND, definition_hashes
from train_head import parser as legacy_parser


def parser():
    result = legacy_parser()
    result.description = __doc__
    result.add_argument("--reference-dir", required=True, help="Same-seed completed legacy centered-full run")
    result.add_argument("--gate-init", type=float, default=0.5)
    result.set_defaults(model_dim=64, ffn_dim=256, threads=4)
    return result


def check_reference(args, config, signature, hashes, model):
    folder = Path(args.reference_dir)
    reference = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if reference["head_config"] != config.to_dict() or reference["feature_signature"] != signature:
        raise ValueError("Reference architecture or feature signature differs")
    if reference["data_sha256"] != hashes:
        raise ValueError("Reference uses different training/validation caches")
    for key in ("seed", "lr", "weight_decay", "epochs", "patience", "batch_size", "sampling", "threads"):
        if reference["arguments"][key] != getattr(args, key):
            raise ValueError(f"Reference training setting differs: {key}")
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
    if initial["seed"] != args.seed or initial["head_config"] != config.to_dict() or initial["feature_signature"] != signature:
        raise ValueError("Reference initial checkpoint differs")
    shared = {key: value.detach().cpu() for key, value in model.state_dict().items() if key != "interaction_logit"}
    if shared.keys() != initial["state_dict"].keys() or any(not torch.equal(shared[k], initial["state_dict"][k]) for k in shared):
        raise ValueError("Shared initialization differs from same-seed legacy full")
    return {"reference_dir": str(folder), "reference_initial_sha256": file_sha256(folder / "initial.pt"),
            "reference_best_sha256": file_sha256(folder / "best.pt"), "shared_initial_weights_equal": True}


def run(args):
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.lr <= 0:
        raise ValueError("Invalid training settings")
    if args.mode != "full" or args.gate or args.sampling != "balanced":
        raise ValueError("Use --mode full, balanced sampling, and do not pass the legacy --gate flag")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(args.threads)
    train, val = FeatureDataset(args.train, "source_train"), FeatureDataset(args.val, "source_val")
    if train.signature != val.signature:
        raise ValueError("Train/validation feature signatures differ")
    assert_disjoint(train, val)
    hashes = {str(path): file_sha256(path) for path in args.train + args.val}
    seed_everything(args.seed)
    device = select_device(args.device)
    config = HeadConfig(train.signature["hidden_dim"], args.model_dim, args.heads, args.ffn_dim, args.dropout, "full", False)
    model = InteractionGatedOptionHead(config, args.gate_init).to(device)
    paired = check_reference(args, config, train.signature, hashes, model)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(train.sampling_weights(), len(train), replacement=True, generator=generator)
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, generator=generator,
                        collate_fn=collate_features, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collate_features)
    baseline = evaluate(None, val_loader, device)
    out.mkdir(parents=True, exist_ok=False)
    parameter_count = sum(p.numel() for p in model.parameters())
    write_json(out / "config.json", {
        "model_kind": MODEL_KIND, "arguments": vars(args), "head_config": config.to_dict(),
        "feature_signature": train.signature, "data_sha256": hashes, "paired_initialization": paired,
        "gate_definition": "self + sigmoid(alpha) * (full - self), shared attention weights",
        "gate_init": args.gate_init, "definition_sha256": definition_hashes(),
        "selection_metric": "source macro_accuracy; tie-break macro_nll", "parameter_count": parameter_count,
        "baseline_source_validation": baseline,
        "environment": {"torch": str(torch.__version__), "python": platform.python_version(), "device": str(device)},
        "dropout_note": "Same initialization and sampler protocol; two attention evaluations consume different dropout draws than legacy single-branch heads.",
    })

    def save(filename, epoch, metrics):
        torch.save({"schema_version": 1, "model_kind": MODEL_KIND, "head_config": config.to_dict(),
                    "gate_init": args.gate_init, "definition_sha256": definition_hashes(),
                    "state_dict": model.state_dict(), "feature_signature": train.signature,
                    "seed": args.seed, "epoch": epoch, "source_validation": metrics,
                    "gate_value": model.gate_value()}, out / filename)

    save("initial.pt", 0, baseline)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, best_score, stale = [], (-float("inf"), -float("inf")), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        total_loss = n = 0
        gate_before, gradient_sum, gradient_max = model.gate_value(), 0.0, 0.0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch["hidden"], batch["base_logits"], batch["valid"]), batch["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            gradient = model.interaction_logit.grad
            if gradient is None or not torch.isfinite(gradient):
                raise FloatingPointError("Missing/non-finite interaction-gate gradient")
            value = abs(float(gradient.detach().cpu()))
            gradient_sum += value
            gradient_max = max(gradient_max, value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(batch["labels"])
            n += len(batch["labels"])
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_ce": total_loss / n, "seconds": time.perf_counter() - started,
               "gate_before": gate_before, "gate_after": model.gate_value(),
               "gate_gradient_abs_sum": gradient_sum, "gate_gradient_abs_max": gradient_max,
               "source_validation": metrics}
        history.append(row)
        write_json(out / "history.json", history)
        print(json.dumps(row), flush=True)
        score = (metrics["macro_accuracy"], -metrics["macro_nll"])
        if score > best_score:
            best_score, stale = score, 0
            save("best.pt", epoch, metrics)
        else:
            stale += 1
        if stale >= args.patience:
            break
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    write_json(out / "summary.json", {
        "model_kind": MODEL_KIND, "best_epoch": checkpoint["epoch"], "best_gate_value": checkpoint["gate_value"],
        "last_gate_value": history[-1]["gate_after"], "gate_init": args.gate_init,
        "gate_received_nonzero_gradient": any(row["gate_gradient_abs_max"] > 0 for row in history),
        "baseline_source_validation": baseline, "best_source_validation": checkpoint["source_validation"],
        "parameter_count": parameter_count, "synthetic_only": bool(train.signature.get("synthetic")),
        "paired_initialization": paired,
    })
    return out


if __name__ == "__main__":
    run(parser().parse_args())
