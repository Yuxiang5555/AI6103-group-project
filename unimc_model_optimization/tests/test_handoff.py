import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_model_handoff import audit_calibration_exports, python_stage, validate_calibration_cache, validate_source_splits


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class HandoffContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def prepare_source(self):
        for task, numbers in (("piqa", (2000, 919, 919)), ("cqa", (2000, 598, 599))):
            for split, count in zip(("train", "val", "cal"), numbers):
                write_rows(self.root / task / f"{split}.jsonl", [
                    {"role": f"source_{split}", "content_hash": f"{task}:{split}:{i}"} for i in range(count)])

    def prepare_exports(self):
        rows = [{"id": str(i), "dataset": "piqa" if i < 919 else "commonsenseqa",
                 "choices": ["a", "b"], "gold_label": i % 2, "base_logits": [0.25, -0.75]}
                for i in range(1518)]
        for name in ("baseline", "centered_full_seed42", "centered_full_seed43", "centered_full_seed44"):
            write_rows(self.root / f"{name}.jsonl", rows)
            (self.root / f"{name}.jsonl.metadata.json").write_text(json.dumps({
                "uncalibrated": True, "cache_roles": ["source_cal", "source_cal"]}), encoding="utf-8")
        return rows

    def test_source_split_overlap_is_rejected_without_relaxing_counts(self):
        self.prepare_source()
        self.assertEqual(sum(validate_source_splits(self.root).values()), 7035)
        path = self.root / "piqa/cal.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["content_hash"] = "piqa:train:0"
        write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_source_splits(self.root)

    def test_wrong_role_is_rejected_before_packaging(self):
        self.prepare_source()
        path = self.root / "piqa/cal.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["role"] = "source_val"
        write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "role"):
            validate_source_splits(self.root)

    def test_all_calibration_model_views_must_have_matching_labels_and_ids(self):
        rows = self.prepare_exports()
        self.assertEqual(audit_calibration_exports(self.root)["matched_model_views"], 4)
        rows[0]["gold_label"] = 1
        write_rows(self.root / "centered_full_seed44.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "labels"):
            audit_calibration_exports(self.root)

    def test_baseline_also_requires_uncalibrated_source_cal_metadata(self):
        self.prepare_exports()
        path = self.root / "baseline.jsonl.metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["cache_roles"] = ["source_val", "source_val"]
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source-calibration"):
            audit_calibration_exports(self.root)
        metadata.update(cache_roles=["source_cal", "source_cal"], uncalibrated=False)
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(ValueError):
            audit_calibration_exports(self.root)

    def test_calibration_cache_is_joined_to_prepared_inputs_by_id(self):
        rows = [{"dataset": task, "id": "0", "content_hash": task,
                 "choices": ["a", "b"], "gold_label": 1} for task in ("piqa", "commonsenseqa")]
        for task, row in zip(("piqa", "cqa"), rows):
            write_rows(self.root / task / "cal.jsonl", [row])
        data = SimpleNamespace(samples=list(reversed(rows)))
        validate_calibration_cache(data, self.root)
        data.samples = [dict(row) for row in data.samples]
        data.samples[0]["content_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "contents differ"):
            validate_calibration_cache(data, self.root)

    def test_linux_child_uses_gnu_without_mutating_parent_environment(self):
        with patch.dict(os.environ, {"MKL_THREADING_LAYER": "INTEL"}), \
                patch("build_model_handoff.sys.platform", "linux"), \
                patch("build_model_handoff.subprocess.run") as run:
            python_stage(["predict_head.py", "--baseline"], self.root / "stage.log")
            self.assertEqual(run.call_args.kwargs["env"]["MKL_THREADING_LAYER"], "GNU")
            self.assertEqual(os.environ["MKL_THREADING_LAYER"], "INTEL")
            self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
