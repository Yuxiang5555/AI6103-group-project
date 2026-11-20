"""Official UniMC online encoding, losses, evaluation and local model fingerprint."""
import contextlib
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, DataLoader

from data_io import file_sha256, read_json_records, write_json


def load_records(paths, role):
    records, seen, contents = [], set(), set()
    for path in paths:
        for row in read_json_records(path):
            if row.get("role") != role:
                raise ValueError(f"{path}: expected {role}")
            key = (row["dataset"], str(row["id"]))
            if key in seen or row["content_hash"] in contents:
                raise ValueError("Duplicate IDs/question contents")
            if len(row["choices"]) < 2 or len(set(row["choices"])) != len(row["choices"]):
                raise ValueError("Need at least two distinct options")
            gold = row.get("gold_label")
            if gold is not None and (isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(row["choices"])):
                raise ValueError("Invalid option label")
            if role in {"source_train", "source_val"} and gold is None:
                raise ValueError("Source training/selection requires labels")
            seen.add(key)
            contents.add(row["content_hash"])
            records.append(row)
    if not records:
        raise ValueError("No records")
    return records


def disjoint_records(train, val):
    if ({r["content_hash"] for r in train} & {r["content_hash"] for r in val} or
            {(r["dataset"], str(r["id"])) for r in train} & {(r["dataset"], str(r["id"])) for r in val}):
        raise ValueError("Source train/validation overlap")


class OnlineDataset(Dataset):
    def __init__(self, records, official_encoder, masked=False):
        self.records, self.encoder, self.masked = records, official_encoder, masked

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        item = {"id": row["id"], "texta": row["texta"], "textb": row.get("textb", ""),
                "question": row.get("question", ""), "choice": copy.deepcopy(row["choices"]),
                "label": row["gold_label"] if row.get("gold_label") is not None else 0}
        feature = self.encoder.encode(item, used_mask=self.masked)
        positions = (feature["clslabels_mask"] == 0).nonzero(as_tuple=True)[0]
        if len(positions) != len(row["choices"]):
            raise ValueError("Official option markers were truncated")
        return {"feature": feature, "record": row, "choices": item["choice"],
                "options_shortened": item["choice"] != row["choices"]}


class OnlineCollator:
    def __init__(self, official_collator):
        self.official = official_collator

    def __call__(self, items):
        features = [item["feature"] for item in items]
        batch = self.official(features)
        # Official collator pads mlmlabels with zero. Ignore every padding target;
        # otherwise padding would become an unintended language-model objective.
        labels = batch["mlmlabels"].clone()
        for i, feature in enumerate(features):
            labels[i, len(feature["input_ids"]):] = -100
        batch["mlmlabels"] = labels
        return {"features": batch, "items": items}


def forward_batch(model, packed, device, training=False):
    source = packed["features"]
    kwargs = {key: source[key].to(device) for key in
              ("input_ids", "attention_mask", "position_ids", "token_type_ids", "clslabels_mask")}
    if training:
        kwargs.update(mlmlabels=source["mlmlabels"].to(device), clslabels=source["clslabels"].to(device))
    loss, _, sequence_scores = model(**kwargs)
    scores = []
    for i, item in enumerate(packed["items"]):
        positions = (source["clslabels_mask"][i] == 0).nonzero(as_tuple=True)[0].to(device)
        option_scores = sequence_scores[i, positions].float()
        if len(option_scores) != len(item["choices"]) or not torch.isfinite(option_scores).all():
            raise FloatingPointError("Invalid option scores")
        scores.append(option_scores)
    return loss, scores


def amp_context(device, mode):
    if mode == "none":
        return contextlib.nullcontext()
    if device.type != "cuda" or mode != "bf16" or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 requires a supported CUDA GPU; use --amp none otherwise")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def metrics_from_predictions(rows):
    totals = {}
    for row in rows:
        gold = row.get("gold_label")
        if gold is None:
            continue
        score = torch.tensor(row["logits"], dtype=torch.float64)
        group = totals.setdefault(row["dataset"], {"n": 0, "correct": 0, "nll_sum": 0.})
        group["n"] += 1
        group["correct"] += int(int(score.argmax()) == gold)
        group["nll_sum"] -= float(score.log_softmax(0)[gold])
    per_task = {key: {"n": row["n"], "correct": row["correct"], "accuracy": row["correct"]/row["n"],
                      "nll": row["nll_sum"]/row["n"]} for key, row in sorted(totals.items())}
    n = sum(row["n"] for row in per_task.values())
    return {"n": n, "accuracy": sum(row["correct"] for row in per_task.values())/n if n else None,
            "macro_accuracy": sum(row["accuracy"] for row in per_task.values())/len(per_task) if n else None,
            "macro_nll": sum(row["nll"] for row in per_task.values())/len(per_task) if n else None,
            "per_dataset": per_task}


@torch.inference_mode()
def predict_online(model, dataset, collator, device, output=None, baseline=None, progress=False):
    model.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator)
    for step, packed in enumerate(loader):
        _, scores = forward_batch(model, packed, device, training=False)  # always FP32 evaluation
        for item, score in zip(packed["items"], scores):
            original = item["record"]
            values = score.cpu()
            row = {"id": original["id"], "dataset": original["dataset"], "gold_label": original.get("gold_label"),
                   "choices": item["choices"], "num_options": len(values), "logits": values.tolist(),
                   "prediction": int(values.argmax()), "probabilities": values.softmax(0).tolist(),
                   "options_shortened": item["options_shortened"]}
            if baseline is not None:
                previous = baseline[(row["dataset"], str(row["id"]))]
                if row["choices"] != previous["choices"] or row["gold_label"] != previous["gold_label"]:
                    raise ValueError("Online/reference labels or options differ")
                row["base_logits"] = previous.get("base_logits", previous["logits"])
            rows.append(row)
        if progress and (step+1) % 100 == 0:
            print(f"ONLINE EVAL {step+1}/{len(dataset)}", flush=True)
    metrics = metrics_from_predictions(rows)
    if output is not None:
        output = Path(output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        write_json(str(output)+".metrics.json", metrics)
    return rows, metrics


def build_runtime(model_dir, fengshen_root, device, max_length=512):
    model_dir, fengshen_root = Path(model_dir).resolve(), Path(fengshen_root).resolve()
    if not model_dir.is_dir() or not (fengshen_root / "fengshen").is_dir():
        raise ValueError("Use existing local model and Fengshen checkout")
    sys.path.insert(0, str(fengshen_root))
    from transformers import AlbertTokenizer
    from fengshen.models.unimc.modeling_unimc import UniMCModel, UniMCDataset, UniMCDataModel
    tokenizer = AlbertTokenizer.from_pretrained(str(model_dir))
    yes, no = [tokenizer.encode(word, add_special_tokens=False) for word in ("yes", "no")]
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("Need single-token yes/no")
    model = UniMCModel(str(model_dir), yes[0]).to(device).eval()
    if model.config.model_type != "albert":
        raise ValueError("This LoRA implementation targets the verified English ALBERT model")
    settings = SimpleNamespace(max_length=max_length, num_labels=2, batchsize=1)
    encoder = UniMCDataset([], yes[0], no[0], tokenizer, settings, used_mask=False)
    collator = OnlineCollator(UniMCDataModel([], [], yes[0], no[0], tokenizer, settings).collate_fn)
    weights = sorted(set(model_dir.glob("pytorch_model*.bin")) | set(model_dir.glob("*.safetensors")))
    signature = {"hidden_dim": int(model.config.hidden_size), "weights_sha256": {p.name: file_sha256(p) for p in weights},
                 "config_sha256": file_sha256(model_dir / "config.json"),
                 "tokenizer_sha256": {p.name: file_sha256(p) for p in sorted(model_dir.iterdir())
                                      if p.is_file() and (p.suffix == ".model" or p.name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"})},
                 "encoding_sha256": hashlib.sha256(inspect.getsource(UniMCDataset).encode()).hexdigest(),
                 "max_length": max_length, "amp": "none", "cached_hidden_dtype": "float16", "yes_token": yes[0],
                 "allow_option_truncation": True}
    return SimpleNamespace(model=model, encoder=encoder, collator=collator, signature=signature,
                           no_token=no[0], model_dir=str(model_dir), fengshen_root=str(fengshen_root))
