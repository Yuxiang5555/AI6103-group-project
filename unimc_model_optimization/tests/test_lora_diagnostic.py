import sys
from pathlib import Path
import unittest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_unimc_lora import loss_components, cosine, gradient_vector, audit_loss_precision
from unimc_lora import inject_lora, checkpoint_shared_groups, frozen_parameter_hash, adapter_state


class TinySharedGroup(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.query = nn.Linear(8, 8)
        self.attention.value = nn.Linear(8, 8)

    def forward(self, x):
        return x + 0.1*(self.attention.query(x).tanh()+self.attention.value(x))


class TinySharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7, 8)
        self.albert_layer_groups = nn.ModuleList([TinySharedGroup()])
        self.decoder = nn.Linear(8, 7)

    def forward(self, ids):
        x = self.embedding(ids)
        for _ in range(3):
            x = self.albert_layer_groups[0](x)
        return self.decoder(x)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_weighted_partition_reconstructs_mlm_loss_and_gradient(self):
        logits = torch.randn(2, 5, 7, requires_grad=True)
        labels = torch.tensor([[1, 2, -100, 4, -100], [2, 1, 5, -100, -100]])
        mask = torch.tensor([[0., 0., -10000, -10000, -10000], [0., 0., -10000, -10000, -10000]])
        components = loss_components(logits, labels, mask, torch.tensor([0, 1]), 1)
        original = F.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1), ignore_index=-100)
        torch.testing.assert_close(components["mlm"], original)
        a = torch.autograd.grad(components["mlm"], logits, retain_graph=True)[0]
        b = torch.autograd.grad(original, logits)[0]
        torch.testing.assert_close(a, b)
        self.assertEqual(components["option_targets"], 4)
        self.assertEqual(components["passage_targets"], 2)

    def test_no_passage_targets_and_zero_gradient_cosine(self):
        logits = torch.randn(1, 4, 7, requires_grad=True)
        labels = torch.tensor([[1, 2, -100, -100]])
        mask = torch.tensor([[0., 0., -10000, -10000]])
        result = loss_components(logits, labels, mask, torch.tensor([0]), 1)
        self.assertEqual(result["passage_targets"], 0)
        torch.testing.assert_close(result["mlm"], result["option_mlm"])
        self.assertIsNone(cosine(torch.zeros(3), torch.ones(3)))
        self.assertAlmostEqual(cosine(torch.ones(3), -torch.ones(3)), -1., places=6)

    def test_low_precision_scalar_is_reported_without_relaxing_fp32_validation(self):
        # Same BF16 logits, full low-precision CE versus explicit FP32 CE.
        logits = (torch.randn(1, 8, 32000)*2).bfloat16().requires_grad_()
        labels = torch.tensor([[1, 2, 3, -100, -100, -100, -100, -100]])
        mask = torch.tensor([[0., 0., -10000, -10000, -10000, -10000, -10000, -10000]])
        parts = loss_components(logits, labels, mask, torch.tensor([0]), 1)
        reduced = F.cross_entropy(logits.reshape(-1, 32000), labels.reshape(-1))
        probe = audit_loss_precision(logits, labels, parts, reduced, strict_official=False)
        self.assertEqual(probe['official_loss_dtype'], 'torch.bfloat16')
        self.assertFalse(probe['official_matches_fp32_tolerance'])
        self.assertGreater(probe['official_vs_fp32_absolute_error'], 2e-5)
        with self.assertRaisesRegex(AssertionError, 'Official loss differs'):
            audit_loss_precision(logits, labels, parts, reduced)
        reference = F.cross_entropy(logits.float().reshape(-1, 32000), labels.reshape(-1))
        precise = audit_loss_precision(logits, labels, parts, reference)
        self.assertTrue(precise['official_matches_fp32_tolerance'])
        self.assertLess(precise['partition_fp32_absolute_error'], 2e-5)

    def test_wrong_partition_still_fails_even_in_probe_mode(self):
        logits = torch.randn(1, 4, 7, requires_grad=True)
        labels = torch.tensor([[1, 2, -100, -100]])
        parts = loss_components(logits, labels, torch.tensor([[0., 0., -10000, -10000]]), torch.tensor([0]), 1)
        parts['mlm'] = parts['mlm']+0.1
        with self.assertRaises(AssertionError):
            audit_loss_precision(logits, labels, parts, parts['mlm'], strict_official=False)

    def test_repeated_component_backwards_through_shared_checkpoint_preserve_weights(self):
        model = TinySharedModel().train()
        inject_lora(model, rank=2, alpha=4, dropout=0.05)
        checkpoint_shared_groups(model)
        # Exercise gradients through both A and B, beyond the zero-B initialization.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith('lora_B'):
                    param.normal_(std=0.02)
        base_before, state_before = frozen_parameter_hash(model), adapter_state(model)
        parameters = [p for p in model.parameters() if p.requires_grad]
        for has_passage in (True, False):
            logits = model(torch.tensor([[1, 2, 3, 4, 5]]))
            labels = torch.tensor([[1, 2, 4 if has_passage else -100, -100, -100]])
            mask = torch.tensor([[0., 0., -10000, -10000, -10000]])
            terms = loss_components(logits, labels, mask, torch.tensor([1]), 1)
            original = F.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1), ignore_index=-100)+terms['op']
            g_op = gradient_vector(terms['op'], parameters, retain_graph=True)
            g_option = gradient_vector(terms['weighted_option_mlm'], parameters, retain_graph=True)
            g_passage = (gradient_vector(terms['weighted_passage_mlm'], parameters, retain_graph=True)
                         if has_passage else torch.zeros_like(g_op))
            expected = gradient_vector(original, parameters, retain_graph=False)
            torch.testing.assert_close(g_op+g_option+g_passage, expected, atol=2e-6, rtol=1e-5)
            self.assertGreater(float(g_op.norm()), 0.)
        self.assertEqual(base_before, frozen_parameter_hash(model))
        for name, tensor in adapter_state(model).items():
            self.assertTrue(torch.equal(tensor, state_before[name]))
        self.assertTrue(all(p.grad is None for p in parameters))


if __name__ == "__main__":
    unittest.main()
