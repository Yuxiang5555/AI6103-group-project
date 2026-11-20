"""Shared feature-cache contract, validation, batching and evaluation."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_json_records(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or not isinstance(rows, list):
        raise ValueError(f"No records in {path}")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


class FeatureDataset(Dataset):
    def __init__(self, paths, required_role=None):
        self.samples = []
        self.metadata = []
        self.paths = [str(p) for p in paths]
        self.signature = None
        seen = set()
        for path in paths:
            cache = torch.load(path, map_location="cpu", weights_only=True)
            if cache.get("schema_version") != 1:
                raise ValueError(f"Unsupported cache schema: {path}")
            meta = cache["metadata"]
            if required_role is not None and meta.get("role") != required_role:
                raise ValueError(f"{path}: expected role={required_role}, got {meta.get('role')}")
            signature = meta["feature_signature"]
            if self.signature is None:
                self.signature = signature
            elif self.signature != signature:
                raise ValueError("Feature extraction settings/backbone differ between caches")
            if not cache["samples"]:
                raise ValueError(f"Empty cache: {path}")
            self.metadata.append(meta)
            for sample in cache["samples"]:
                key = (sample["dataset"], str(sample["id"]))
                if key in seen:
                    raise ValueError(f"Duplicate sample ID: {key}")
                seen.add(key)
                h, logits = sample["hidden"], sample["base_logits"]
                if h.ndim != 2 or logits.ndim != 1 or len(h) != len(logits) or len(h) < 2:
                    raise ValueError(f"Invalid feature shapes: {key}")
                if h.shape[1] != signature["hidden_dim"]:
                    raise ValueError(f"Hidden dimension mismatch: {key}")
                if not torch.isfinite(h).all() or not torch.isfinite(logits).all():
                    raise ValueError(f"Non-finite cache values: {key}")
                label = sample.get("gold_label")
                if label is not None and (isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(logits)):
                    raise ValueError(f"Label must be an option index: {key}")
                if required_role is not None and label is None:
                    raise ValueError(f"Training/validation requires gold labels: {key}")
                if len(sample.get("choices", [])) != len(logits):
                    raise ValueError(f"Choices do not match features: {key}")
                self.samples.append(sample)
        if not self.samples:
            raise ValueError("No feature caches supplied")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def sampling_weights(self):
        counts = Counter(s["dataset"] for s in self.samples)
        return torch.tensor([1.0 / counts[s["dataset"]] for s in self.samples], dtype=torch.double)


def assert_disjoint(train, val):
    train_ids = {(s["dataset"], str(s["id"])) for s in train.samples}
    val_ids = {(s["dataset"], str(s["id"])) for s in val.samples}
    if train_ids & val_ids:
        raise ValueError("Train/validation sample IDs overlap")
    a = {s["content_hash"] for s in train.samples if s.get("content_hash")}
    b = {s["content_hash"] for s in val.samples if s.get("content_hash")}
    if a & b:
        raise ValueError("Train/validation question contents overlap")


def collate_features(samples):
    b = len(samples)
    k = max(len(s["base_logits"]) for s in samples)
    d = samples[0]["hidden"].shape[-1]
    hidden = torch.zeros(b, k, d)
    base = torch.zeros(b, k)
    valid = torch.zeros(b, k, dtype=torch.bool)
    labels = torch.full((b,), -100, dtype=torch.long)
    for i, sample in enumerate(samples):
        n = len(sample["base_logits"])
        hidden[i, :n] = sample["hidden"].float()
        base[i, :n] = sample["base_logits"].float()
        valid[i, :n] = True
        if sample.get("gold_label") is not None:
            labels[i] = sample["gold_label"]
    return {"hidden": hidden, "base_logits": base, "valid": valid, "labels": labels, "samples": samples}


def move_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.inference_mode()
def evaluate(model, loader, device):
    if model is not None:
        model.eval()
    totals = defaultdict(lambda: {"n": 0, "correct": 0, "nll_sum": 0.0})
    for batch in loader:
        batch = move_batch(batch, device)
        logits = batch["base_logits"].masked_fill(~batch["valid"], -torch.inf) if model is None else model(
            batch["hidden"], batch["base_logits"], batch["valid"]
        )
        if not torch.isfinite(logits[batch["valid"]]).all():
            raise FloatingPointError("Non-finite valid logits")
        logp = logits.log_softmax(-1)
        predictions = logits.argmax(-1)
        for i, sample in enumerate(batch["samples"]):
            gold = sample.get("gold_label")
            if gold is None:
                continue
            group = totals[sample["dataset"]]
            group["n"] += 1
            group["correct"] += int(predictions[i].item() == gold)
            group["nll_sum"] -= float(logp[i, gold])
    per_dataset = {task: {
        "n": v["n"], "correct": v["correct"], "accuracy": v["correct"] / v["n"],
        "nll": v["nll_sum"] / v["n"],
    } for task, v in sorted(totals.items())}
    if not per_dataset:
        return {"n": 0, "accuracy": None, "macro_accuracy": None, "macro_nll": None, "per_dataset": {}}
    n = sum(v["n"] for v in per_dataset.values())
    return {
        "n": n, "accuracy": sum(v["correct"] for v in per_dataset.values()) / n,
        "macro_accuracy": sum(v["accuracy"] for v in per_dataset.values()) / len(per_dataset),
        "macro_nll": sum(v["nll"] for v in per_dataset.values()) / len(per_dataset),
        "per_dataset": per_dataset,
    }
