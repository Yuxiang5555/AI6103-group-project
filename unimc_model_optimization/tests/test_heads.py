import json
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_head import HeadConfig, ResidualOptionHead
from data_io import FeatureDataset, assert_disjoint
from prepare_data import convert


class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(13)
        self.h = torch.randn(3, 5, 16)
        self.s = torch.randn(3, 5)
        self.valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)

    def model(self, mode, active=False, gate=False):
        model = ResidualOptionHead(HeadConfig(16, 16, 4, 32, 0.0, mode, gate))
        if active:
            torch.nn.init.normal_(model.scorer.weight, std=0.2)
        return model

    def test_exact_initial_baseline_for_all_variants(self):
        expected = self.s.masked_fill(~self.valid, -torch.inf)
        for mode in ("mlp", "self", "full"):
            for gate in (False, True):
                actual = self.model(mode, gate=gate)(self.h, self.s, self.valid)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_self_and_full_have_same_parameters_and_initial_weights(self):
        torch.manual_seed(9)
        a = self.model("self")
        torch.manual_seed(9)
        b = self.model("full")
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        for key in a.state_dict():
            torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], atol=0, rtol=0)

    def test_self_has_no_cross_option_dependence_and_full_does(self):
        h = self.h[:1].clone().requires_grad_()
        s, valid = self.s[:1], self.valid[:1]
        isolated = self.model("self", active=True).eval()
        grad = torch.autograd.grad(isolated(h, s, valid)[0, 0], h)[0]
        torch.testing.assert_close(grad[0, 1:], torch.zeros_like(grad[0, 1:]), atol=0, rtol=0)
        interacting = self.model("full", active=True).eval()
        grad = torch.autograd.grad(interacting(h, s, valid)[0, 0], h)[0]
        self.assertGreater(grad[0, 1:].abs().sum().item(), 1e-6)

    def test_permutation_equivariance_after_nonzero_correction(self):
        permutation = torch.tensor([3, 0, 4, 1, 2])
        inverse = permutation.argsort()
        for mode in ("mlp", "self", "full"):
            model = self.model(mode, active=True).eval()
            with torch.inference_mode():
                before = model(self.h, self.s, self.valid)
                after = model(self.h[:, permutation], self.s[:, permutation], self.valid[:, permutation])[:, inverse]
            torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)

    def test_padding_has_no_effect_on_valid_scores(self):
        for mode in ("mlp", "self", "full"):
            model = self.model(mode, active=True).eval()
            h = self.h[1:2, :2]
            s = self.s[1:2, :2]
            valid = torch.ones(1, 2, dtype=torch.bool)
            padded_h = torch.cat([h, torch.randn(1, 4, 16) * 10000], dim=1)
            padded_s = torch.cat([s, torch.ones(1, 4) * 10000], dim=1)
            padded_valid = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool)
            with torch.inference_mode():
                before = model(h, s, valid)
                after = model(padded_h, padded_s, padded_valid)[:, :2]
            torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)

    def test_variable_option_padding_has_finite_backward_and_updates(self):
        for mode in ("mlp", "self", "full"):
            model = self.model(mode)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            for _ in range(3):
                optimizer.zero_grad()
                loss = F.cross_entropy(model(self.h, self.s, self.valid), torch.tensor([4, 1, 2]))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                optimizer.step()
            self.assertGreater(model.scorer.weight.abs().sum().item(), 0)

    def test_all_padding_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model("full")(self.h, self.s, torch.zeros_like(self.valid))

    def test_actual_4096_dimensional_head_shape_and_backward(self):
        model = ResidualOptionHead(HeadConfig())
        h = torch.randn(2, 5, 4096)
        valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
        logits = model(h, torch.zeros(2, 5), valid)
        self.assertEqual(tuple(logits.shape), (2, 5))
        F.cross_entropy(logits, torch.tensor([3, 1])).backward()
        self.assertTrue(torch.isfinite(model.scorer.weight.grad).all())


class DataContractTests(unittest.TestCase):
    def test_sst2_semantic_labels_are_mapped_to_correct_option(self):
        negative = convert({"sentence": "A dull movie.", "label": 0}, "sst2", 0)
        positive = convert({"sentence": "A great movie.", "label": 1}, "sst2", 1)
        self.assertEqual(negative["gold_label"], 1)
        self.assertEqual(positive["gold_label"], 0)

    def test_answer_key_maps_via_actual_choice_keys(self):
        row = {"question": "A question", "choices": {"text": ["one", "two"], "label": ["1", "2"]}, "answerKey": "2"}
        self.assertEqual(convert(row, "arc", 0)["gold_label"], 1)

    def test_unlabeled_official_test_data_are_rejected_by_converter(self):
        with self.assertRaises(ValueError):
            convert({"sentence": "Unknown label", "label": -1}, "sst2", 0)

    def test_roles_and_duplicate_contents_are_checked(self):
        sample = {"id": "a", "dataset": "toy", "choices": ["x", "y"], "gold_label": 0,
                  "hidden": torch.randn(2, 4), "base_logits": torch.zeros(2), "content_hash": "same-question"}
        with tempfile.TemporaryDirectory() as temp:
            train_path, val_path = Path(temp) / "train.pt", Path(temp) / "val.pt"
            def save(path, role, sample_id):
                torch.save({"schema_version": 1, "metadata": {"role": role, "feature_signature": {"hidden_dim": 4}},
                            "samples": [dict(sample, id=sample_id)]}, path)
            save(train_path, "source_train", "train:a")
            save(val_path, "source_val", "val:a")
            train, val = FeatureDataset([train_path], "source_train"), FeatureDataset([val_path], "source_val")
            with self.assertRaises(ValueError):
                assert_disjoint(train, val)
            save(val_path, "target_eval", "eval:a")
            with self.assertRaises(ValueError):
                FeatureDataset([val_path], "source_val")

    def test_prepare_cli_keeps_validation_and_calibration_separate(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            train = [{"id": f"train-{i}", "goal": f"train goal {i}", "sol1": "one", "sol2": "two", "label": i % 2} for i in range(8)]
            holdout = [{"id": f"dev-{i}", "goal": f"dev goal {i}", "sol1": "one", "sol2": "two", "label": i % 2} for i in range(6)]
            (temp / "train.json").write_text(json.dumps(train))
            (temp / "dev.json").write_text(json.dumps(holdout))
            subprocess.run([sys.executable, str(root / "prepare_data.py"), "--task", "piqa", "--role", "source",
                            "--input", str(temp / "train.json"), "--holdout", str(temp / "dev.json"),
                            "--output", str(temp / "prepared")], check=True, capture_output=True)
            groups = {}
            for name, role in (("train", "source_train"), ("val", "source_val"), ("cal", "source_cal")):
                rows = [json.loads(line) for line in (temp / "prepared" / f"{name}.jsonl").read_text().splitlines()]
                self.assertTrue(all(x["role"] == role for x in rows))
                groups[name] = {x["content_hash"] for x in rows}
            self.assertFalse(groups["train"] & groups["val"])
            self.assertFalse(groups["train"] & groups["cal"])
            self.assertFalse(groups["val"] & groups["cal"])


if __name__ == "__main__":
    unittest.main()
