import math
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirm_centering import (plan_runs, summarize, training_args, check_initial_pair,
                               export, prediction_environment, ensure_training_completed,
                               prediction_directory, FIXED)
from train_head import parser


class ConfirmationTests(unittest.TestCase):
    def test_linux_export_replaces_inherited_intel_layer_without_changing_parent(self):
        with mock.patch.dict(os.environ, {"MKL_THREADING_LAYER": "INTEL"}):
            with mock.patch("confirm_centering.sys.platform", "linux"), mock.patch("confirm_centering.subprocess.run") as child:
                export(Path("cache"), Path("output.jsonl"), "cpu", 2)
                self.assertEqual(child.call_args.kwargs["env"]["MKL_THREADING_LAYER"], "GNU")
                self.assertTrue(child.call_args.kwargs["check"])
                self.assertEqual(os.environ["MKL_THREADING_LAYER"], "INTEL")

    def test_windows_threading_environment_is_not_forced_to_gnu(self):
        with mock.patch.dict(os.environ, {"MKL_THREADING_LAYER": "INTEL"}):
            with mock.patch("confirm_centering.sys.platform", "win32"):
                self.assertEqual(prediction_environment()["MKL_THREADING_LAYER"], "INTEL")

    def test_report_only_verifies_existing_runs_without_calling_trainer(self):
        from data_io import file_sha256
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "cache"
            cache.mkdir()
            hashes = {}
            for split in ("train", "val"):
                for task in ("piqa", "cqa"):
                    path = cache / f"{task}_{split}.pt"
                    path.write_bytes(b"cache-fingerprint-fixture")
                    hashes[str(path)] = file_sha256(path)
            run_path = root / "completed"
            run_path.mkdir()
            config = {"arguments": dict(FIXED, seed=43, mode="full"), "data_sha256": hashes}
            (run_path / "config.json").write_text(json.dumps(config))
            (run_path / "summary.json").write_text(json.dumps({"best_epoch": 2, "synthetic_only": False}))
            (run_path / "history.json").write_text("[]")
            for filename in ("initial.pt", "best.pt"):
                torch.save({"weight": torch.ones(1)}, run_path / filename)
            plan = [{"seed": 43, "variant": "centered", "mode": "full", "path": run_path, "train_needed": True}]
            with mock.patch("train_head.run", side_effect=AssertionError("Recovery must never train")) as trainer:
                with contextlib.redirect_stdout(io.StringIO()):
                    ensure_training_completed(plan, {"centered": cache}, "cpu", 2, report_only=True)
                trainer.assert_not_called()
                (run_path / "best.pt").unlink()
                with self.assertRaises(FileNotFoundError):
                    ensure_training_completed(plan, {"centered": cache}, "cpu", 2, report_only=True)
                trainer.assert_not_called()

    def test_recovery_exports_to_fresh_directory_preserving_partial_files(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            original = output / "predictions"
            original.mkdir()
            partial = original / "baseline_val.jsonl"
            partial.write_text("partial old export")
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "55800"}):
                recovered = prediction_directory(output, report_only=True)
            self.assertEqual(recovered.name, "predictions_resume_55800")
            self.assertEqual(list(recovered.iterdir()), [])
            self.assertEqual(partial.read_text(), "partial old export")

    def test_plan_reuses_seed42_and_keeps_paired_training_settings(self):
        refs = {"uncentered": Path("old_uncentered"), "centered": Path("old_centered")}
        plan = plan_runs(Path("new_run"), refs)
        self.assertEqual(len(plan), 18)
        self.assertEqual(sum(row["train_needed"] for row in plan), 12)
        self.assertEqual(len({row["path"] for row in plan}), 18)
        for row in plan:
            args = parser().parse_args(training_args(row, Path("cache") / row["variant"], "cpu", 2))
            self.assertEqual(args.seed, row["seed"])
            self.assertEqual(args.mode, row["mode"])
            self.assertEqual((args.lr, args.model_dim, args.ffn_dim, args.patience), (0.0001, 64, 256, 2))
            self.assertFalse(args.gate)
            if row["seed"] == 42:
                self.assertEqual(row["path"].parent, refs[row["variant"]])

    def records(self):
        rows = []
        for seed, shift, full_delta in ((42, -0.01, -0.001), (43, 0, 0.002), (44, 0.01, 0.005)):
            for variant in ("uncentered", "centered"):
                for mode in ("mlp", "self", "full"):
                    delta = 0 if variant == "uncentered" else {"mlp": 0.002, "self": -0.001, "full": full_delta}[mode]
                    macro = 0.76 + shift + delta
                    rows.append({"variant": variant, "mode": mode, "seed": seed,
                                 "macro_accuracy": macro, "piqa_accuracy": macro - 0.01,
                                 "cqa_accuracy": macro + 0.01, "macro_nll": 0.5})
        return rows

    def test_seed_sd_and_paired_deltas(self):
        result = summarize(self.records(), {"macro_accuracy": 0.76})
        self.assertEqual(len(result["groups"]), 6)
        group = next(row for row in result["groups"] if row["variant"] == "uncentered" and row["mode"] == "full")
        self.assertAlmostEqual(group["macro_accuracy"]["mean"], 0.76)
        self.assertAlmostEqual(group["macro_accuracy"]["sd"], 0.01)
        paired = next(row for row in result["paired_centering_delta_pp"] if row["mode"] == "full")
        self.assertAlmostEqual(paired["mean"], 0.2)
        self.assertAlmostEqual(paired["sd"], 0.3)
        self.assertEqual(paired["positive_seeds"], 2)
        self.assertAlmostEqual(paired["delta_by_seed_pp"]["42"], -0.1)
        contrast = next(row for row in result["attention_delta_pp"] if row["comparison"].endswith("mlp"))
        self.assertAlmostEqual(contrast["mean"], 0)

    def test_missing_duplicate_or_incorrect_seed_rejected(self):
        for rows in (self.records()[:-1], self.records() + self.records()[:1]):
            with self.assertRaises(ValueError):
                summarize(rows, {"macro_accuracy": 0.76})
        rows = self.records()
        rows[0]["seed"] = 99
        with self.assertRaises(ValueError):
            summarize(rows, {"macro_accuracy": 0.76})

    def test_paired_initialization_check_ignores_only_feature_signature(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, second = root / "a", root / "b"
            first.mkdir()
            second.mkdir()
            base = {"head_config": {"mode": "full"}, "state_dict": {"weight": torch.arange(4.)},
                    "feature_signature": {"centered": False}}
            torch.save(base, first / "initial.pt")
            other = dict(base, feature_signature={"centered": True})
            torch.save(other, second / "initial.pt")
            check_initial_pair(first, second)
            other["state_dict"] = {"weight": torch.ones(4)}
            torch.save(other, second / "initial.pt")
            with self.assertRaisesRegex(ValueError, "initializations differ"):
                check_initial_pair(first, second)


if __name__ == "__main__":
    unittest.main()
