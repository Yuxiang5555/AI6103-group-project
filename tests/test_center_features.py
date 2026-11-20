import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from center_option_features import center_cache
from data_io import FeatureDataset, collate_features, evaluate
from train_head import parser, run


class CenteringTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def cache(self, split="val", n=9):
        rows = []
        for i in range(n):
            k = (2, 3, 5)[i % 3]
            h = torch.randn(k, 16)
            rows.append(dict(id=f"{split}:{i}", dataset="synthetic", choices=[str(j) for j in range(k)],
                             gold_label=int(h[:, 0].argmax()), content_hash=f"{split}:{i}",
                             hidden=h, base_logits=torch.randn(k) * 0.05))
        return {"schema_version": 1, "metadata": {"role": f"source_{split}", "n": n,
                "feature_signature": {"hidden_dim": 16, "synthetic": True, "cached_hidden_dtype": "float32"}},
                "samples": rows}

    def test_zero_mean_preserved_rows_and_no_source_mutation(self):
        original = self.cache()
        before = copy.deepcopy(original)
        centered = center_cache(original, "test")
        for a, b, unchanged in zip(before["samples"], centered["samples"], original["samples"]):
            torch.testing.assert_close(b["hidden"].mean(0), torch.zeros(16), atol=2e-7, rtol=0)
            torch.testing.assert_close(a["base_logits"], b["base_logits"], atol=0, rtol=0)
            torch.testing.assert_close(a["hidden"], unchanged["hidden"], atol=0, rtol=0)
            for key in ("id", "dataset", "choices", "gold_label", "content_hash"):
                self.assertEqual(a[key], b[key])
        self.assertEqual(original["metadata"], before["metadata"])
        with self.assertRaisesRegex(ValueError, "already centered"):
            center_cache(centered, "test")

    def test_common_shift_and_option_permutation(self):
        original = self.cache()
        changed = copy.deepcopy(original)
        permutations = []
        for row in changed["samples"]:
            permutation = torch.randperm(len(row["choices"]))
            permutations.append(permutation)
            row["hidden"] = (row["hidden"] + torch.randn(16))[permutation]
            row["choices"] = [row["choices"][j] for j in permutation]
        first, second = center_cache(original, "test"), center_cache(changed, "test")
        for a, b, permutation in zip(first["samples"], second["samples"], permutations):
            torch.testing.assert_close(a["hidden"][permutation], b["hidden"], atol=5e-7, rtol=1e-5)

    def test_label_role_independence_and_exact_baseline_metrics(self):
        original = self.cache()
        changed = copy.deepcopy(original)
        changed["metadata"]["role"] = "target_eval"
        for row in changed["samples"]:
            row["gold_label"] = None
        first, second = center_cache(original, "test"), center_cache(changed, "test")
        self.assertEqual(first["metadata"]["feature_signature"], second["metadata"]["feature_signature"])
        for a, b in zip(first["samples"], second["samples"]):
            torch.testing.assert_close(a["hidden"], b["hidden"], atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            values = []
            for name, artifact in (("raw", original), ("center", first)):
                path = folder / f"{name}.pt"
                torch.save(artifact, path)
                data = FeatureDataset([path], "source_val")
                values.append(evaluate(None, DataLoader(data, batch_size=4, collate_fn=collate_features), torch.device("cpu")))
            self.assertEqual(values[0], values[1])

    def test_real_training_interface_and_rejecting_unprocessed_inference(self):
        code = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            paths = {}
            for split, n in (("train", 48), ("val", 24)):
                original = self.cache(split, n)
                torch.save(original, folder / f"raw_{split}.pt")
                paths[split] = folder / f"center_{split}.pt"
                torch.save(center_cache(original, "test"), paths[split])
            args = parser().parse_args(["--train", str(paths["train"]), "--val", str(paths["val"]),
                                        "--output", str(folder / "run"), "--mode", "full",
                                        "--model-dim", "16", "--ffn-dim", "32", "--epochs", "3",
                                        "--batch-size", "16", "--device", "cpu"])
            run(args)
            checkpoint = folder / "run/best.pt"
            common = [sys.executable, str(code / "predict_head.py"), "--checkpoint", str(checkpoint), "--device", "cpu"]
            result = subprocess.run(common + ["--cache", str(paths["val"]), "--output", str(folder / "valid.jsonl")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len((folder / "valid.jsonl").read_text().splitlines()), 24)
            mismatch = subprocess.run(common + ["--cache", str(folder / "raw_val.pt"), "--output", str(folder / "wrong.jsonl")], capture_output=True, text=True)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("signatures differ", mismatch.stderr)
            self.assertFalse((folder / "wrong.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
