"""Export uncalibrated, valid-option logits for the evaluation/calibration team."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_io import FeatureDataset, collate_features, evaluate, file_sha256, move_batch, select_device, write_json
from decision_head import HeadConfig, ResidualOptionHead


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", nargs="+", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", action="store_true")
    group.add_argument("--checkpoint")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    data = FeatureDataset(args.cache)
    loader = DataLoader(data, batch_size=args.batch_size, collate_fn=collate_features)
    model = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["feature_signature"] != data.signature:
            raise ValueError("Checkpoint/cache extraction signatures differ")
        model = ResidualOptionHead(HeadConfig(**checkpoint["head_config"])).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = batch["base_logits"].masked_fill(~batch["valid"], -torch.inf) if model is None else model(
                batch["hidden"], batch["base_logits"], batch["valid"]
            )
            if not torch.isfinite(logits[batch["valid"]]).all():
                raise FloatingPointError("Non-finite valid logits")
            for i, sample in enumerate(batch["samples"]):
                k = len(sample["choices"])
                scores = logits[i, :k].float().cpu()
                row = {"id": sample["id"], "dataset": sample["dataset"],
                       "gold_label": sample.get("gold_label"), "choices": sample["choices"],
                       "num_options": k, "logits": scores.tolist(),
                       "base_logits": sample["base_logits"].float().tolist(),
                       "prediction": int(scores.argmax()), "probabilities": scores.softmax(-1).tolist()}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    metrics = evaluate(model, loader, device)
    write_json(str(out) + ".metrics.json", metrics)
    write_json(str(out) + ".metadata.json", {
        "uncalibrated": True, "baseline": args.baseline, "feature_signature": data.signature,
        "checkpoint": args.checkpoint, "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
        "cache_roles": [m["role"] for m in data.metadata],
        "cache_sha256": {str(x): file_sha256(x) for x in args.cache},
        "timing_note": "Cached-feature inference; this is not an end-to-end latency benchmark.",
    })
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
