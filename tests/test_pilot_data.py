"""Regression checks for leakage-free public source pilot preparation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from prepare_pilot_data import exclude_holdout_overlap


def jsonl(rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


class PilotDataTests(unittest.TestCase):
    def test_overlap_removed_even_when_labels_disagree_and_holdout_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shared = {"goal": "Shared question", "sol1": "First", "sol2": "Second"}
            train = [dict(shared, id="overlap-1", label=0), dict(shared, id="overlap-2", label=1),
                     {"id": "unique", "goal": "Unique question", "sol1": "A", "sol2": "B", "label": 0}]
            holdout = [dict(shared, id="dev-shared", label=1)]
            train_path, dev_path = root / "train.jsonl", root / "dev.jsonl"
            train_path.write_text(jsonl(train), encoding="utf-8")
            dev_path.write_text(jsonl(holdout), encoding="utf-8")
            original_dev = dev_path.read_bytes()
            filtered, report = exclude_holdout_overlap(train_path, dev_path, "piqa")
            self.assertEqual([row["id"] for row in map(json.loads, filtered.read_text().splitlines())], ["unique"])
            self.assertEqual(report["excluded_train_n"], 2)
            self.assertEqual(report["overlap_unique_contents"], 1)
            self.assertEqual(report["conflicting_label_n"], 1)
            self.assertEqual(dev_path.read_bytes(), original_dev)
            self.assertEqual(train_path.read_text(encoding="utf-8"), jsonl(train))

    def test_cached_downloads_prepare_both_tasks_without_split_leakage(self):
        code = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            with zipfile.ZipFile(raw / "piqa.zip", "w") as archive:
                shared = {"goal": "Shared PIQA", "sol1": "A", "sol2": "B"}
                for split, count in (("train", 8), ("dev", 6)):
                    rows = [dict(goal=f"{split} PIQA {i}", sol1="A", sol2="B") for i in range(count)]
                    rows.append(shared)
                    archive.writestr(f"data/{split}.jsonl", jsonl(rows))
                    archive.writestr(f"data/{split}-labels.lst", "0\n" * len(rows))
            for split, count in (("train", 8), ("dev", 6)):
                def row(identifier, stem, choices=("A text", "B text")):
                    return {"id": identifier, "answerKey": "B", "question": {"stem": stem,
                            "choices": [{"label": key, "text": text} for key, text in zip(("A", "B"), choices)]}}
                rows = [row(f"{split}-{i}", f"{split} CQA {i}") for i in range(count)]
                rows.append(row(f"{split}-shared", "Shared CQA"))
                if split == "train":
                    rows.append(row("ambiguous", "Duplicate options", ("same", "same")))
                (raw / f"cqa_{split}.jsonl").write_text(jsonl(rows), encoding="utf-8")
            prepared = root / "prepared"
            result = subprocess.run([sys.executable, str(code / "prepare_pilot_data.py"),
                                     "--raw-dir", str(raw), "--output", str(prepared),
                                     "--limit-train", "4"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((prepared / "provenance.json").read_text(encoding="utf-8"))
            for task in ("piqa", "cqa"):
                self.assertEqual(manifest["cleaning"][f"{task}_train_holdout_overlap"]["excluded_train_n"], 1)
                sets = []
                for split, role in (("train", "source_train"), ("val", "source_val"), ("cal", "source_cal")):
                    rows = [json.loads(line) for line in (prepared / task / f"{split}.jsonl").read_text().splitlines()]
                    self.assertEqual({row["role"] for row in rows}, {role})
                    sets.append({row["content_hash"] for row in rows})
                self.assertEqual([len(items) for items in sets], [4, 4, 3])
                self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertEqual(len(manifest["cleaning"]["cqa_train"]["excluded"]), 1)
            # The strict converter still rejects unfiltered train/dev overlap.
            guard = subprocess.run([sys.executable, str(code / "prepare_data.py"), "--task", "piqa",
                                   "--role", "source", "--input", str(raw / "piqa_train_clean.jsonl"),
                                   "--holdout", str(raw / "piqa_dev_clean.jsonl"),
                                   "--output", str(root / "must_not_exist")], text=True, capture_output=True)
            self.assertNotEqual(guard.returncode, 0)
            self.assertIn("question contents overlap", guard.stderr)
            self.assertFalse((root / "must_not_exist").exists())


if __name__ == "__main__":
    unittest.main()
