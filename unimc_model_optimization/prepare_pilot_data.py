"""Download public PIQA/CQA source data and prepare a reproducible head pilot.

Run beside prepare_data.py. Uses only Python's standard library; no GPU needed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import urllib.request
import zipfile

from prepare_data import convert


def download(url, path):
    if not path.is_file():
        for attempt in range(3):
            print(f"Downloading {url} (attempt {attempt + 1}/3)", flush=True)
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(path)
    payload = path.read_bytes()
    return payload, {"url": url, "bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()}


def read_jsonl(payload):
    return [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]


def clean(rows, task, origin, raw_dir):
    retained, excluded = [], []
    for index, row in enumerate(rows):
        row = dict(row)
        row.setdefault("id", f"{task}-{origin}-{index}")
        canonical = convert(row, task, index)
        choices = [choice.strip() for choice in canonical["choices"]]
        if len(set(choices)) != len(choices):
            excluded.append({"id": str(row["id"]), "reason": "duplicate candidate text"})
        else:
            retained.append(row)
    path = raw_dir / f"{task}_{origin}_clean.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained), encoding="utf-8")
    return path, {"original_n": len(rows), "retained_n": len(retained), "excluded": excluded}


def exclude_holdout_overlap(train_path, holdout_path, task):
    """Keep the evaluation pool fixed; exclude its exact content from training."""
    holdout_by_hash = {}
    for index, row in enumerate(read_jsonl(holdout_path.read_bytes())):
        item = convert(row, task, index)
        holdout_by_hash.setdefault(item["content_hash"], []).append(item)
    train = read_jsonl(train_path.read_bytes())
    retained, excluded, overlap_hashes = [], [], set()
    for index, row in enumerate(train):
        item = convert(row, task, index)
        matches = holdout_by_hash.get(item["content_hash"])
        if matches:
            overlap_hashes.add(item["content_hash"])
            excluded.append({
                "id": item["id"], "content_hash": item["content_hash"],
                "holdout_ids": [match["id"] for match in matches],
                "labels_match": all(item["gold_label"] == match["gold_label"] for match in matches),
            })
        else:
            retained.append(row)
    if not retained:
        raise ValueError(f"{task}: no source training rows remain after excluding holdout overlap")
    disjoint_path = train_path.with_name(train_path.stem + "_disjoint.jsonl")
    disjoint_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained), encoding="utf-8")
    report = {
        "policy": "Keep holdout unchanged; exclude exact content_hash matches from train before subsampling.",
        "original_train_n": len(train), "retained_train_n": len(retained),
        "excluded_train_n": len(excluded), "overlap_unique_contents": len(overlap_hashes),
        "conflicting_label_n": sum(not row["labels_match"] for row in excluded),
        "excluded_train": excluded,
    }
    return disjoint_path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="raw/pilot_seed42")
    parser.add_argument("--output", default="data/pilot_seed42")
    parser.add_argument("--limit-train", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit_train < 1:
        raise ValueError("--limit-train must be positive")
    raw_dir, output = Path(args.raw_dir).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a NEW --output directory: {output}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    provenance, cleaning, inputs = {}, {}, {}
    payload, provenance["piqa_zip"] = download(
        "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
        raw_dir / "piqa.zip")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = {}
        for name in ("train.jsonl", "train-labels.lst", "dev.jsonl", "dev-labels.lst"):
            matches = [member for member in archive.namelist() if PurePosixPath(member).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name} in PIQA archive")
            members[name] = archive.read(matches[0])
    for origin in ("train", "dev"):
        rows = read_jsonl(members[f"{origin}.jsonl"])
        labels = [int(line) for line in members[f"{origin}-labels.lst"].decode().splitlines() if line.strip()]
        if len(rows) != len(labels):
            raise ValueError(f"PIQA {origin}: example/label count mismatch")
        rows = [dict(row, label=label) for row, label in zip(rows, labels)]
        inputs[("piqa", origin)], cleaning[f"piqa_{origin}"] = clean(rows, "piqa", origin, raw_dir)
    for origin in ("train", "dev"):
        payload, provenance[f"cqa_{origin}"] = download(
            f"https://s3.amazonaws.com/commensenseqa/{origin}_rand_split.jsonl",
            raw_dir / f"cqa_{origin}.jsonl")
        inputs[("commonsenseqa", origin)], cleaning[f"cqa_{origin}"] = clean(
            read_jsonl(payload), "commonsenseqa", origin, raw_dir)
    converter = Path(__file__).resolve().with_name("prepare_data.py")
    for task, directory in (("piqa", "piqa"), ("commonsenseqa", "cqa")):
        inputs[(task, "train")], cleaning[f"{directory}_train_holdout_overlap"] = exclude_holdout_overlap(
            inputs[(task, "train")], inputs[(task, "dev")], task)
        overlap = cleaning[f"{directory}_train_holdout_overlap"]
        print(f"{task}: excluded {overlap['excluded_train_n']} training rows matching holdout; "
              f"{overlap['retained_train_n']} training rows remain", flush=True)
        print(f"Preparing {task}", flush=True)
        subprocess.run([
            sys.executable, str(converter), "--task", task, "--role", "source",
            "--input", str(inputs[(task, "train")]), "--holdout", str(inputs[(task, "dev")]),
            "--output", str(output / directory), "--limit-train", str(args.limit_train),
            "--seed", str(args.seed)], check=True)
    manifest = {
        "protocol_version": "source-pilot-v2-disjoint",
        "purpose": "Source-task pilot; not final target evaluation", "seed": args.seed,
        "limit_train_per_task": args.limit_train,
        "selection": "Exclude train contents overlapping official dev, then subsample train; official dev split 50/50 into source_val/source_cal.",
        "old_backbone_exposure": "Not established. Disjointness applies to training the NEW head.",
        "source_provenance": provenance, "cleaning": cleaning,
        "scripts_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (Path(__file__).resolve(), converter)},
        "files": {},
    }
    for directory in ("piqa", "cqa"):
        seen = set()
        for split, role in (("train", "source_train"), ("val", "source_val"), ("cal", "source_cal")):
            path = output / directory / f"{split}.jsonl"
            rows = read_jsonl(path.read_bytes())
            hashes = {row["content_hash"] for row in rows}
            if not rows or len(hashes) != len(rows) or hashes & seen:
                raise ValueError(f"Empty/duplicate/overlapping split: {path}")
            if any(row["role"] != role for row in rows):
                raise ValueError(f"Incorrect data role: {path}")
            seen.update(hashes)
            manifest["files"][path.relative_to(output).as_posix()] = {
                "n": len(rows), "role": role,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["files"], indent=2), flush=True)
    print(f"PILOT DATA READY: {output}", flush=True)


if __name__ == "__main__":
    main()
