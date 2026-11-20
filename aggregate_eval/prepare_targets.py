"""Prepare frozen target validation inputs and paired candidate perturbations."""
import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "unimc_model_optimization"))
from prepare_data import convert

SUPER = ("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646")
SOURCES = {
    "sst2": ("nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "sst2"),
    "cb": (*SUPER, "cb"),
    "copa": (*SUPER, "copa"),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "data"),
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def dump_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def variants(item):
    k = len(item["choices"])
    original = list(range(k))
    gold = item["gold_label"]
    yield "original", original
    seed = int(digest(f'{item["dataset"]}/{item["id"]}/20261120'.encode()), 16)
    shift = 1 + seed % (k - 1)
    yield "rotate", original[shift:] + original[:shift]
    if k >= 3:
        distractors = [i for i in original if i != gold]
        drop = distractors[seed % len(distractors)]
        yield "drop_distractor", [i for i in original if i != drop]
        yield "drop_gold", distractors


def prepare(task, path, dest):
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    seen, ids, counts, exclusions = {}, set(), Counter(), []
    with (dest / f"{task}.jsonl").open("x", encoding="utf-8") as inputs:
        for i, row in enumerate(rows):
            item = convert(row, task, i)
            upstream_id = item["id"]
            # Several distinct continuations share HellaSwag's upstream ind.
            if task == "hellaswag":
                item["id"] = f"row-{i}"
            key = item["content_hash"]
            if key in seen and seen[key] != item["gold_label"]:
                raise ValueError(f"Conflicting duplicate in {task} row {i}")
            if key in seen or len(set(item["choices"])) != len(item["choices"]):
                exclusions.append({"row": i, "id": item["id"], "reason": "duplicate item" if key in seen else "duplicate choices"})
                continue
            seen[key] = item["gold_label"]
            if item["id"] in ids:
                raise ValueError(f"Duplicate ID in {task}: {item['id']}")
            ids.add(item["id"])
            for condition, order in variants(item):
                choices = [item["choices"][j] for j in order]
                gold = item["gold_label"]
                fields = dict(texta=item["texta"], textb=item["textb"], question=item["question"], choices=choices)
                content_hash = digest(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode())
                dump_line(inputs, dict(fields, id=f"{condition}/{item['id']}", dataset=task,
                                       original_id=item["id"], upstream_id=upstream_id, original_indices=order, condition=condition,
                                       role="target_eval", gold_label=order.index(gold) if gold in order else None,
                                       content_hash=content_hash))
                counts[condition] += 1
    return dict(raw_rows=len(rows), conditions=dict(counts), exclusions=exclusions,
                parquet_sha256=digest(path.read_bytes()), input_sha256=digest((dest / f"{task}.jsonl").read_bytes()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parquet-dir", type=Path, help="Use local pinned downloads named TASK.parquet")
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        dest = Path(tmp) / "inputs"
        dest.mkdir()
        manifest = {}
        for task, (repo, revision, folder) in SOURCES.items():
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{folder}/validation-00000-of-00001.parquet"
            path = (args.parquet_dir or Path(tmp)) / f"{task}.parquet"
            if not args.parquet_dir:
                with urlopen(url, timeout=90) as response, path.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            manifest[task] = dict(url=url, **prepare(task, path, dest))
            print(task, manifest[task]["conditions"], flush=True)
        expected = Path(__file__).with_name("target_manifest.json")
        if manifest != json.loads(expected.read_text()):
            raise ValueError("Target sources or prepared inputs differ from the pinned manifest")
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(dest, out)


if __name__ == "__main__":
    main()
