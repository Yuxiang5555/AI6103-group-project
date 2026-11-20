import copy
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_io import FeatureDataset
from decision_head import HeadConfig, ResidualOptionHead
from interaction_gate_head import InteractionGatedOptionHead, load_gate_checkpoint
from predict_interaction_gate import export_predictions
from run_interaction_gate import paired_diagnostics, summarize
import run_interaction_gate as workflow
import train_head
import train_interaction_gate


class InteractionGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(19)
        self.hidden = torch.randn(3, 5, 16)
        self.base = torch.randn(3, 5)
        self.valid = torch.tensor([[1]*5, [1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        self.config = HeadConfig(16, 16, 4, 32, 0.0, "full", False)

    def head(self, active=True):
        model = InteractionGatedOptionHead(self.config)
        if active:
            with torch.no_grad():
                model.scorer.weight.copy_(torch.linspace(-0.3, 0.3, 16)[None])
        return model

    def test_shared_initialization_and_only_one_added_parameter(self):
        torch.manual_seed(42)
        parent = ResidualOptionHead(self.config)
        torch.manual_seed(42)
        gated = self.head(active=False)
        for key, value in parent.state_dict().items():
            torch.testing.assert_close(gated.state_dict()[key], value, atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in gated.parameters())-sum(p.numel() for p in parent.parameters()), 1)
        self.assertEqual(gated.gate_value(), 0.5)

    def test_zero_scorer_exactly_preserves_baseline(self):
        model = self.head(active=False)
        for gate in (None, 0.0, 0.35, 1.0):
            actual = model(self.hidden, self.base, self.valid, gate_override=gate)
            torch.testing.assert_close(actual, self.base.masked_fill(~self.valid, -torch.inf), atol=0, rtol=0)

    def test_nonzero_scorer_endpoints_equal_trained_self_and_full_architectures(self):
        gated = self.head().eval()
        common = {k: v for k, v in gated.state_dict().items() if k != "interaction_logit"}
        for mode, gate in (("self", 0.0), ("full", 1.0)):
            parent = ResidualOptionHead(HeadConfig(**{**self.config.to_dict(), "mode": mode})).eval()
            parent.load_state_dict(common, strict=True)
            with torch.inference_mode():
                expected = parent(self.hidden, self.base, self.valid)
                actual = gated(self.hidden, self.base, self.valid, gate_override=gate)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertGreater((gated(self.hidden, self.base, self.valid, gate_override=0)[self.valid] -
                            gated(self.hidden, self.base, self.valid, gate_override=1)[self.valid]).abs().max().item(), 1e-5)

    def test_gate_closes_cross_option_dependency_at_head_input(self):
        model = self.head().eval()
        h = self.hidden[:1].clone().requires_grad_()
        grad0 = torch.autograd.grad(model(h, self.base[:1], self.valid[:1], gate_override=0)[0, 0], h)[0]
        torch.testing.assert_close(grad0[0, 1:], torch.zeros_like(grad0[0, 1:]), atol=0, rtol=0)
        grad1 = torch.autograd.grad(model(h, self.base[:1], self.valid[:1], gate_override=1)[0, 0], h)[0]
        self.assertGreater(grad1[0, 1:].abs().sum().item(), 1e-6)

    def test_padding_and_permutation_do_not_change_valid_decisions(self):
        model = self.head().eval()
        permutation = torch.tensor([4, 2, 0, 3, 1])
        with torch.inference_mode():
            before = model(self.hidden, self.base, self.valid)
            after = model(self.hidden[:, permutation], self.base[:, permutation], self.valid[:, permutation])[:, permutation.argsort()]
            torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)
            a = model(self.hidden[1:2, :2], self.base[1:2, :2], self.valid[1:2, :2])
            h, b = self.hidden[1:2].clone(), self.base[1:2].clone()
            h[:, 2:], b[:, 2:] = 1e5, -1e5
            padded = model(h, b, self.valid[1:2])[:, :2]
            torch.testing.assert_close(a, padded, atol=1e-5, rtol=1e-5)

    def test_gate_receives_finite_gradients_and_updates_with_variable_options(self):
        model = self.head(active=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        seen_nonzero = False
        for _ in range(6):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(self.hidden, self.base, self.valid), torch.tensor([4, 1, 2]))
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            seen_nonzero |= abs(model.interaction_logit.grad.item()) > 0
            optimizer.step()
        self.assertTrue(seen_nonzero)
        self.assertNotEqual(model.gate_value(), 0.5)
        self.assertTrue(0 < model.gate_value() < 1)

    def test_invalid_settings_and_all_padding_are_rejected(self):
        with self.assertRaises(ValueError):
            InteractionGatedOptionHead(HeadConfig(mode="self"))
        with self.assertRaises(ValueError):
            InteractionGatedOptionHead(HeadConfig(gate=True))
        for value in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                InteractionGatedOptionHead(self.config, value)
        with self.assertRaises(ValueError):
            self.head()(self.hidden, self.base, torch.zeros_like(self.valid))


class InteractionGatePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_training_save_load_export_and_role_guards_on_synthetic_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = torch.Generator().manual_seed(121)
            for split, n in (("train", 20), ("val", 12)):
                samples = []
                for i in range(n):
                    k = 2 if i % 2 else 5
                    samples.append({"id": f"{split}-{i}", "dataset": "a" if k == 2 else "b", "choices": [str(j) for j in range(k)],
                                    "gold_label": i % k, "content_hash": f"{split}-{i}",
                                    "hidden": torch.randn(k, 16, generator=generator), "base_logits": torch.zeros(k)})
                torch.save({"schema_version": 1, "metadata": {"role": f"source_{split}",
                            "feature_signature": {"hidden_dim": 16, "synthetic": True}}, "samples": samples}, root / f"{split}.pt")
            common = ["--train", str(root / "train.pt"), "--val", str(root / "val.pt"), "--mode", "full",
                      "--model-dim", "16", "--heads", "4", "--ffn-dim", "32", "--dropout", "0",
                      "--epochs", "3", "--patience", "2", "--batch-size", "8", "--lr", "0.001",
                      "--threads", "2", "--seed", "42", "--device", "cpu"]
            reference = root / "reference"
            with contextlib.redirect_stdout(io.StringIO()):
                train_head.run(train_head.parser().parse_args(common + ["--output", str(reference)]))
            args = train_interaction_gate.parser().parse_args(common + ["--output", str(root / "gated"), "--reference-dir", str(reference)])
            with contextlib.redirect_stdout(io.StringIO()):
                folder = train_interaction_gate.run(args)
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["paired_initialization"]["shared_initial_weights_equal"])
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["gate_received_nonzero_gradient"])
            self.assertTrue(summary["synthetic_only"])
            data = FeatureDataset([root / "val.pt"], "source_val")
            model, checkpoint = load_gate_checkpoint(folder / "best.pt", data.signature, torch.device("cpu"))
            self.assertEqual(model.gate_value(), checkpoint["gate_value"])
            path = root / "gated.jsonl"
            metrics = export_predictions(data, path, torch.device("cpu"), folder / "best.pt")
            self.assertEqual(metrics["n"], 12)
            self.assertEqual(metrics["macro_accuracy"], summary["best_source_validation"]["macro_accuracy"])
            meta = json.loads(Path(str(path)+".metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["uncalibrated"])
            with self.assertRaisesRegex(ValueError, "signatures differ"):
                load_gate_checkpoint(folder / "best.pt", {"hidden_dim": 15}, torch.device("cpu"))
            broken = torch.load(root / "train.pt", weights_only=True)
            broken["metadata"]["role"] = "source_cal"
            torch.save(broken, root / "cal.pt")
            args.train, args.output = [str(root / "cal.pt")], str(root / "must_not_exist")
            with self.assertRaisesRegex(ValueError, "source_train"):
                train_interaction_gate.run(args)
            self.assertFalse((root / "must_not_exist").exists())

    def test_paired_analysis_uses_question_pairs_and_rejects_changed_choices(self):
        baseline = {}
        for task in ("a", "b"):
            for i in range(4):
                baseline[(task, str(i))] = {"dataset": task, "id": str(i), "choices": ["x", "y"],
                                          "gold_label": 0, "prediction": 1, "logits": [0., 1.], "base_logits": [0., 1.]}
        current = copy.deepcopy(baseline)
        current[("a", "0")]["prediction"] = 0
        self.assertEqual(paired_diagnostics(baseline, current)["corrected"], 1)
        predictions, records = {}, []
        for mode in ("mlp", "self", "full", "gated"):
            for seed in (42, 43, 44):
                predictions[(mode, seed)] = copy.deepcopy(current if mode == "gated" else baseline)
                records.append({"mode": mode, "seed": seed, "macro_accuracy": 0.125 if mode == "gated" else 0., "macro_nll": 1.})
        result = summarize(records, predictions, {"macro_accuracy": 0.}, draws=100)
        self.assertEqual(result["paired_gate_comparisons"][0]["mean_delta_pp"], 12.5)
        self.assertEqual(result["paired_gate_comparisons"][0]["positive_seeds"], 3)
        current[("a", "0")]["choices"] = ["y", "x"]
        with self.assertRaisesRegex(ValueError, "options"):
            paired_diagnostics(baseline, current)

    def test_runner_exports_packages_and_recovers_without_retraining(self):
        source = Path(__file__).resolve().parents[1]
        original_cwd = Path.cwd()
        original_verifier = workflow.verify_completed
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            project.mkdir()
            for name in workflow.NEW_FILES + workflow.LEGACY_FILES + workflow.INFERENCE_FILES + ("README_交互Gate实施.md", "tests/test_interaction_gate.py"):
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
            try:
                os.chdir(project)
                cache = Path("cache/fixture")
                cache.mkdir(parents=True)
                generator = torch.Generator().manual_seed(77)
                signature = {"hidden_dim": 128, "synthetic": True, "option_centering": {"fixture_only": True}}
                for split, count in (("train", 10), ("val", 6)):
                    for task, dataset, k in (("piqa", "piqa", 2), ("cqa", "commonsenseqa", 5)):
                        samples = [{"id": f"{split}-{task}-{i}", "dataset": dataset, "choices": [str(j) for j in range(k)],
                                    "gold_label": i % k, "content_hash": f"{split}-{task}-{i}",
                                    "hidden": torch.randn(k, 128, generator=generator), "base_logits": torch.zeros(k)} for i in range(count)]
                        torch.save({"schema_version": 1, "metadata": {"role": f"source_{split}", "feature_signature": signature},
                                    "samples": samples}, cache / f"{task}_{split}.pt")
                train_paths = [str(cache / f"{task}_train.pt") for task in ("piqa", "cqa")]
                val_paths = [str(cache / f"{task}_val.pt") for task in ("piqa", "cqa")]
                train, val = FeatureDataset(train_paths, "source_train"), FeatureDataset(val_paths, "source_val")
                hashes = {p: workflow.file_sha256(p) for p in train_paths+val_paths}
                references, baseline = {}, None
                with contextlib.redirect_stdout(io.StringIO()):
                    for mode in ("mlp", "self", "full"):
                        for seed in (42, 43, 44):
                            folder = Path("runs/controls") / f"{mode}_seed{seed}"
                            args = train_head.parser().parse_args([
                                "--train", *train_paths, "--val", *val_paths, "--output", str(folder), "--mode", mode,
                                "--model-dim", "64", "--ffn-dim", "256", "--seed", str(seed), "--threads", "4", "--device", "cpu"])
                            train_head.run(args)
                            references[(mode, seed)] = folder
                            baseline = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["baseline_source_validation"]
                report_path = Path("fixture_report.json")
                report = {"baseline": baseline}
                report_path.write_text(json.dumps(report), encoding="utf-8")
                fixture = (report, train, val, train_paths, val_paths, hashes, references)
                output = Path("runs/gate_fixture")
                with patch.object(workflow, "preflight", return_value=fixture), \
                        patch.object(workflow, "verify_completed", side_effect=lambda *a: original_verifier(*a, require_real=False)), \
                        patch.object(sys, "argv", ["run_interaction_gate.py", "--output", str(output), "--reference-report", str(report_path), "--device", "cpu"]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    workflow.main()
                first = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
                self.assertEqual(len(first["seed_results"]), 12)
                self.assertFalse(first["source_cal_used"])
                with self.assertRaisesRegex(ValueError, "real-data"):
                    original_verifier(output / "gated_seed42", 42, signature, hashes)
                with patch.object(workflow, "preflight", return_value=fixture), \
                        patch.object(workflow, "verify_completed", side_effect=lambda *a: original_verifier(*a, require_real=False)), \
                        patch.object(workflow.train_interaction_gate, "run", side_effect=AssertionError("Recovery must not train")), \
                        patch.object(sys, "argv", ["run_interaction_gate.py", "--output", str(output), "--reference-report", str(report_path), "--device", "cpu", "--report-only"]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    workflow.main()
                second = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
                self.assertEqual(first["seed_results"], second["seed_results"])
                self.assertEqual(first["paired_gate_comparisons"], second["paired_gate_comparisons"])
                with zipfile.ZipFile(project.parent / "UniMC_interaction_gate_results.zip") as bundle:
                    self.assertIsNone(bundle.testzip())
                    self.assertEqual(len([n for n in bundle.namelist() if n.startswith("models/")]), 12)
                    self.assertEqual(len([n for n in bundle.namelist() if n.startswith("predictions/") and n.endswith(".jsonl")]), 13)
                for name in workflow.LEGACY_FILES:
                    self.assertEqual((source / name).read_bytes(), Path(name).read_bytes())
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
