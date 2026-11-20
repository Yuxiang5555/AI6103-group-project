"""Extract frozen UniMC option features INSIDE an already working Fengshen environment.

This adapter reuses official encoding and model classes. It does not reimplement
the special position IDs / 3D attention masks or assume stock HF BERT parity.
"""
import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from data_io import file_sha256, read_json_records, select_device, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Canonical JSONL from prepare_data.py")
    p.add_argument("--output", required=True)
    p.add_argument("--fengshen-root", required=True, help="Checkout with a working official UniMC environment")
    p.add_argument("--model", default="IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English")
    p.add_argument("--revision", default="main", help="Prefer a fixed Hub commit; resolved snapshot is recorded")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, help="First N examples, for an initial pipeline check")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    p.add_argument("--verify-official", action="store_true", help="Verify labels/probabilities on the first batch; requires CUDA")
    p.add_argument("--allow-option-truncation", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or not 32 < args.max_length <= 512:
        raise ValueError("Need positive batch-size and 32 < max-length <= 512")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    root = Path(args.fengshen_root).resolve()
    if not (root / "fengshen").is_dir():
        raise ValueError("--fengshen-root must contain fengshen/")
    sys.path.insert(0, str(root))
    try:
        from transformers import AlbertTokenizer
        from fengshen.models.unimc.modeling_unimc import UniMCDataset, UniMCDataModel, UniMCModel, UniMCPredict
    except ImportError as exc:
        raise RuntimeError("Run this script in the environment where the OFFICIAL UniMC pipeline already works. "
                           "Head training can use a separate PyTorch-only environment.") from exc
    device = select_device(args.device)
    if args.verify_official and device.type != "cuda":
        raise ValueError("The official UniMCPredict hard-codes .cuda(); verify on your A40")
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("AMP extraction in this adapter requires CUDA")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is unsupported on this device; use --amp none")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    def amp_context():
        return contextlib.nullcontext() if amp_dtype is None else torch.autocast("cuda", dtype=amp_dtype)
    model_dir = Path(args.model)
    if not model_dir.is_dir():
        from huggingface_hub import snapshot_download
        model_dir = Path(snapshot_download(args.model, revision=args.revision))
    tokenizer = AlbertTokenizer.from_pretrained(str(model_dir))
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError("This adapter requires single-token English yes/no, as in the ALBERT checkpoint")
    yes_token, no_token = yes_ids[0], no_ids[0]
    legacy_args = SimpleNamespace(max_length=args.max_length, num_labels=2, batchsize=args.batch_size)
    model = UniMCModel(str(model_dir), yes_token).to(device)
    model.requires_grad_(False)
    model.eval()
    if model.config.model_type != "albert" or not hasattr(model.bert, "albert"):
        raise ValueError("This adapter targets the English ALBERT UniMC checkpoint")
    records = read_json_records(args.input)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        records = records[:args.limit]
    roles = {row.get("role") for row in records}
    if len(roles) != 1 or next(iter(roles)) not in {"source_train", "source_val", "source_cal", "target_eval"}:
        raise ValueError("Each input file must have one explicit source_train/source_val/source_cal/target_eval role")
    role = next(iter(roles))
    weight_files = sorted(set(model_dir.glob("*.safetensors")) | set(model_dir.glob("pytorch_model*.bin")))
    if not weight_files:
        raise FileNotFoundError("No PyTorch/safetensors weight files in the resolved model snapshot")
    weight_hashes = {f.name: file_sha256(f) for f in weight_files}
    tokenizer_hashes = {f.name: file_sha256(f) for f in sorted(model_dir.iterdir())
                        if f.is_file() and (f.suffix == ".model" or f.name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"})}
    signature = {
        "hidden_dim": int(model.config.hidden_size), "weights_sha256": weight_hashes,
        "config_sha256": file_sha256(model_dir / "config.json"), "tokenizer_sha256": tokenizer_hashes,
        "encoding_sha256": hashlib.sha256(inspect.getsource(UniMCDataset).encode()).hexdigest(),
        "max_length": args.max_length, "amp": args.amp, "cached_hidden_dtype": "float16",
        "yes_token": yes_token, "allow_option_truncation": args.allow_option_truncation,
    }
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    collator = UniMCDataModel([], [], yes_token, no_token, tokenizer, legacy_args).collate_fn
    encoder = UniMCDataset([], yes_token, no_token, tokenizer, legacy_args, used_mask=False)
    captured = {}
    def capture_last_hidden(module, inputs, output):
        captured["hidden"] = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    hook = model.bert.albert.register_forward_hook(capture_last_hidden)
    samples, truncated = [], 0
    reference_check = {"requested": args.verify_official, "passed": False, "n": 0}
    try:
        with torch.inference_mode():
            for start in range(0, len(records), args.batch_size):
                originals = records[start:start + args.batch_size]
                legacy, encoded = [], []
                for row in originals:
                    if len(set(row["choices"])) != len(row["choices"]):
                        raise ValueError(f"Duplicate option texts in {row['id']}; audit ambiguous examples")
                    if row.get("gold_label") is not None and not 0 <= row["gold_label"] < len(row["choices"]):
                        raise ValueError(f"Invalid gold label in {row['id']}")
                    item = {"id": row["id"], "texta": row["texta"], "textb": row.get("textb", ""),
                            "question": row.get("question", ""), "choice": copy.deepcopy(row["choices"]),
                            "label": row.get("gold_label") if row.get("gold_label") is not None else 0}
                    feature = encoder.encode(item, used_mask=False)
                    if item["choice"] != row["choices"]:
                        truncated += 1
                        if not args.allow_option_truncation:
                            raise ValueError(f"Official preprocessing shortened options in {row['id']}. Audit it or explicitly allow truncation.")
                    legacy.append(item)
                    encoded.append(feature)
                batch = {k: v.to(device) for k, v in collator(encoded).items()}
                kwargs = {k: batch[k] for k in ("input_ids", "attention_mask", "token_type_ids", "position_ids", "clslabels_mask")}
                with amp_context():
                    _, _, sequence_logits = model(**kwargs)
                hidden = captured.pop("hidden")
                positions = [(batch["clslabels_mask"][i] == 0).nonzero(as_tuple=True)[0] for i in range(len(originals))]
                probabilities = []
                for i, (row, item, pos) in enumerate(zip(originals, legacy, positions)):
                    if len(pos) != len(row["choices"]) or hidden.shape[-1] != signature["hidden_dim"]:
                        raise ValueError("Option markers/hidden dimensions do not match")
                    logits = sequence_logits[i, pos].float().cpu()
                    h = hidden[i, pos].to(dtype=torch.float16).cpu()
                    if not torch.isfinite(h).all() or not torch.isfinite(logits).all():
                        raise FloatingPointError("Non-finite extraction output; retry full precision")
                    probabilities.append(logits.softmax(-1))
                    samples.append({"id": row["id"], "dataset": row["dataset"],
                                    "gold_label": row.get("gold_label"), "choices": item["choice"],
                                    "content_hash": row.get("content_hash"), "hidden": h, "base_logits": logits})
                if start == 0 and args.verify_official:
                    official = UniMCPredict(yes_token, no_token, SimpleNamespace(model=model), tokenizer, legacy_args)
                    with amp_context():
                        reference = official.predict(copy.deepcopy(legacy))
                    for i, item in enumerate(reference):
                        expected = torch.tensor([item["score"][c] for c in legacy[i]["choice"]])
                        torch.testing.assert_close(probabilities[i], expected, atol=1e-5, rtol=1e-5)
                        if int(probabilities[i].argmax()) != item["label"]:
                            raise AssertionError("Official prediction and extracted scores disagree")
                    reference_check.update(passed=True, n=len(reference))
                    captured.clear()
                if (start // args.batch_size) % 20 == 0:
                    print(f"extracted {len(samples)}/{len(records)}", flush=True)
    finally:
        hook.remove()
    meta = {"role": role, "feature_signature": signature, "input_sha256": file_sha256(args.input),
            "model": args.model, "resolved_model_dir": str(model_dir), "fengshen_commit": commit,
            "encoder_eval": True, "official_prediction_check": reference_check, "limit": args.limit,
            "truncated_option_examples": truncated, "n": len(samples)}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "metadata": meta, "samples": samples}, out)
    write_json(str(out) + ".metadata.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
