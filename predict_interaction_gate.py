"""Export uncalibrated logits for new interaction-gate or unchanged legacy heads."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_io import FeatureDataset, collate_features, evaluate, file_sha256, move_batch, select_device, write_json
from decision_head import HeadConfig, ResidualOptionHead
from interaction_gate_head import MODEL_KIND, load_gate_checkpoint


def export_predictions(data, output, device, checkpoint_path=None, batch_size=128):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if len({meta["role"] for meta in data.metadata}) != 1:
        raise ValueError("Do not mix source/target roles in a prediction file")
    model, model_kind = None, "baseline"
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_kind") == MODEL_KIND:
            model, checkpoint = load_gate_checkpoint(checkpoint_path, data.signature, device)
            model_kind = MODEL_KIND
        else:
            if checkpoint.get("model_kind") is not None:
                raise ValueError("Unknown checkpoint model_kind")
            if checkpoint["feature_signature"] != data.signature:
                raise ValueError("Checkpoint/cache extraction signatures differ")
            model = ResidualOptionHead(HeadConfig(**checkpoint["head_config"])).to(device).eval()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model_kind = "legacy_" + checkpoint["head_config"]["mode"]
    loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_features)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            scores = batch["base_logits"].masked_fill(~batch["valid"], -torch.inf) if model is None else model(
                batch["hidden"], batch["base_logits"], batch["valid"])
            if not torch.isfinite(scores[batch["valid"]]).all():
                raise FloatingPointError("Non-finite valid logits")
            for i, sample in enumerate(batch["samples"]):
                k = len(sample["choices"])
                score = scores[i, :k].float().cpu()
                row = {"id": sample["id"], "dataset": sample["dataset"], "gold_label": sample.get("gold_label"),
                       "choices": sample["choices"], "num_options": k, "logits": score.tolist(),
                       "base_logits": sample["base_logits"].float().tolist(), "prediction": int(score.argmax()),
                       "probabilities": score.softmax(-1).tolist()}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    metrics = evaluate(model, loader, device)
    write_json(str(output) + ".metrics.json", metrics)
    write_json(str(output) + ".metadata.json", {
        "uncalibrated": True, "model_kind": model_kind, "feature_signature": data.signature,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": file_sha256(checkpoint_path) if checkpoint_path is not None else None,
        "interaction_gate_value": model.gate_value() if model_kind == MODEL_KIND else None,
        "cache_roles": [meta["role"] for meta in data.metadata],
        "cache_sha256": {str(path): file_sha256(path) for path in data.paths},
        "timing_note": "Cached-head inference; gated attention evaluates two masks with shared weights.",
    })
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", nargs="+", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--baseline", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if min(args.threads, args.batch_size) < 1:
        raise ValueError("Threads and batch size must be positive")
    torch.set_num_threads(args.threads)
    metrics = export_predictions(FeatureDataset(args.cache), args.output, select_device(args.device),
                                 args.checkpoint, args.batch_size)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
