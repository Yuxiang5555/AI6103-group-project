"""Train only a residual head on source caches; never select on target tasks."""
import argparse
import json
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_io import (FeatureDataset, assert_disjoint, collate_features, evaluate,
                     file_sha256, move_batch, seed_everything, select_device, write_json)
from decision_head import HeadConfig, ResidualOptionHead


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--output", required=True, help="New run directory; existing directories are refused")
    p.add_argument("--mode", choices=["mlp", "self", "full"], required=True)
    p.add_argument("--gate", action="store_true")
    p.add_argument("--model-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling", choices=["balanced", "natural"], default="balanced")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=2)
    return p


def run(args):
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.lr <= 0:
        raise ValueError("epochs, patience, batch-size, threads and lr must be positive")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = select_device(args.device)
    train = FeatureDataset(args.train, required_role="source_train")
    val = FeatureDataset(args.val, required_role="source_val")
    if train.signature != val.signature:
        raise ValueError("Train/validation extraction signatures differ")
    assert_disjoint(train, val)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = HeadConfig(
        hidden_dim=train.signature["hidden_dim"], model_dim=args.model_dim,
        heads=args.heads, feedforward_dim=args.ffn_dim, dropout=args.dropout,
        mode=args.mode, gate=args.gate,
    )
    model = ResidualOptionHead(config).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if args.sampling == "balanced":
        sampler = WeightedRandomSampler(train.sampling_weights(), len(train), replacement=True, generator=generator)
    loader = DataLoader(
        train, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
        generator=generator, collate_fn=collate_features, num_workers=0,
    )
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collate_features)
    baseline = evaluate(None, val_loader, device)
    manifest = {
        "arguments": vars(args), "head_config": config.to_dict(),
        "feature_signature": train.signature, "selection_metric": "source macro_accuracy; tie-break macro_nll",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "data_sha256": {str(p): file_sha256(p) for p in args.train + args.val},
        "environment": {"torch": str(torch.__version__), "python": platform.python_version(), "device": str(device)},
        "baseline_source_validation": baseline,
    }
    write_json(out / "config.json", manifest)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_score = (-float("inf"), -float("inf"))
    stale = 0
    # Save the zero-correction baseline too, for an exact initialization reference.
    def save_checkpoint(filename, epoch, metrics):
        torch.save({
            "schema_version": 1, "head_config": config.to_dict(), "state_dict": model.state_dict(),
            "feature_signature": train.signature, "seed": args.seed, "epoch": epoch,
            "source_validation": metrics,
        }, out / filename)
    save_checkpoint("initial.pt", 0, baseline)
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss, n = 0.0, 0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["hidden"], batch["base_logits"], batch["valid"])
            loss = F.cross_entropy(logits, batch["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; check masks, features and learning rate")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(batch["labels"])
            n += len(batch["labels"])
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_ce": total_loss / n,
               "seconds": time.perf_counter() - started, "source_validation": metrics}
        history.append(row)
        write_json(out / "history.json", history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = (metrics["macro_accuracy"], -metrics["macro_nll"])
        if score > best_score:
            best_score, stale = score, 0
            save_checkpoint("best.pt", epoch, metrics)
        else:
            stale += 1
        if stale >= args.patience:
            break
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    write_json(out / "summary.json", {
        "best_epoch": checkpoint["epoch"], "baseline_source_validation": baseline,
        "best_source_validation": checkpoint["source_validation"],
        "parameter_count": manifest["parameter_count"], "synthetic_only": bool(train.signature.get("synthetic")),
    })
    return out


if __name__ == "__main__":
    run(parser().parse_args())
