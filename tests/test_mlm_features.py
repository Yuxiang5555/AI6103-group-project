"""Verify frozen predecoder extraction without downloading a real checkpoint."""
import copy
import sys
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_io import FeatureDataset, collate_features, evaluate
from project_mlm_features import FrozenMlmPredecoder, project_cache, verify_predecoder


class TinyMlmHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(16, 4)
        self.activation = nn.GELU()
        self.LayerNorm = nn.LayerNorm(4)
        self.decoder = nn.Linear(4, 12)

    def forward(self, hidden):
        return self.decoder(self.LayerNorm(self.activation(self.dense(hidden))))


class MlmFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.head = TinyMlmHead().eval()
        self.projector = FrozenMlmPredecoder(self.head)
        self.spec = {"kind": "SYNTHETIC_TEST_ONLY", "converter_sha256": "test"}
        samples = []
        for i, k in enumerate((2, 5, 3)):
            hidden = torch.randn(k, 16).half()
            with torch.inference_mode():
                scores = self.head(hidden.float())[:, 3].clone()
            samples.append({"id": str(i), "dataset": "toy", "content_hash": str(i),
                            "choices": [str(j) for j in range(k)], "gold_label": i % k,
                            "hidden": hidden, "base_logits": scores})
        self.cache = {"schema_version": 1, "metadata": {
            "role": "source_val", "n": 3, "feature_signature": {
                "hidden_dim": 16, "yes_token": 3, "synthetic": True},
            "official_prediction_check": {"passed": True}}, "samples": samples}

    def project(self, cache=None):
        return project_cache(self.cache if cache is None else cache, self.projector,
                             self.head.decoder, self.spec, torch.device("cpu"), batch_size=2)

    def test_matches_actual_decoder_input_and_preserves_rows_and_base(self):
        hidden = torch.cat([row["hidden"] for row in self.cache["samples"]]).float()
        result = verify_predecoder(self.head, self.projector, hidden, 3)
        self.assertTrue(result["passed"])
        original = copy.deepcopy(self.cache)
        projected = self.project()
        expected = self.projector(hidden).half()
        torch.testing.assert_close(torch.cat([row["hidden"] for row in projected["samples"]]),
                                   expected, atol=1e-3, rtol=1e-3)
        for before, after, unchanged in zip(original["samples"], projected["samples"], self.cache["samples"]):
            for key in ("id", "dataset", "content_hash", "choices", "gold_label"):
                self.assertEqual(before[key], after[key])
            torch.testing.assert_close(before["base_logits"], after["base_logits"], atol=0, rtol=0)
            torch.testing.assert_close(before["hidden"], unchanged["hidden"], atol=0, rtol=0)
            self.assertEqual(after["hidden"].shape[-1], 4)
        self.assertEqual(self.cache["metadata"], original["metadata"])
        self.assertEqual(projected["metadata"]["feature_signature"]["hidden_dim"], 4)
        self.assertTrue(all(not p.requires_grad for p in self.projector.parameters()))

    def test_projection_is_label_independent_and_signature_is_role_independent(self):
        first = self.project()
        changed = copy.deepcopy(self.cache)
        changed["metadata"]["role"] = "target_eval"
        for row in changed["samples"]:
            row["gold_label"] = None
        second = self.project(changed)
        self.assertEqual(first["metadata"]["feature_signature"], second["metadata"]["feature_signature"])
        for a, b in zip(first["samples"], second["samples"]):
            torch.testing.assert_close(a["hidden"], b["hidden"], atol=0, rtol=0)

    def test_roundtrip_keeps_exact_baseline_metrics(self):
        projected = self.project()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.save(self.cache, root / "raw.pt")
            torch.save(projected, root / "mlm.pt")
            metrics = []
            for name in ("raw", "mlm"):
                data = FeatureDataset([root / f"{name}.pt"], "source_val")
                loader = DataLoader(data, batch_size=2, collate_fn=collate_features)
                metrics.append(evaluate(None, loader, torch.device("cpu")))
            self.assertEqual(metrics[0], metrics[1])

    def test_rejects_wrong_dimensions_and_double_projection(self):
        changed = copy.deepcopy(self.cache)
        changed["metadata"]["feature_signature"]["hidden_dim"] = 99
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            self.project(changed)
        with self.assertRaisesRegex(ValueError, "already projected"):
            self.project(self.project())


if __name__ == "__main__":
    unittest.main()
