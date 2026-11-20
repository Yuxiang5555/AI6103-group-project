"""Synthetic contract tests; optional tiny official ALBERT test in the server env."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unimc_lora import (LoRALinear, adapter_state, adapters_disabled, checkpoint_shared_groups,
                        frozen_parameter_hash, inject_lora, restore_adapter)
from unimc_online import OnlineCollator, OnlineDataset, disjoint_records, load_records, predict_online
from run_unimc_lora import pad_memory_probe, train_online
from predict_unimc_lora import load_adapter_for_runtime


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query, self.key, self.value = [nn.Linear(16, 16) for _ in range(3)]
        self.norm = nn.LayerNorm(16)

    def forward(self, x, attention_mask):
        score = self.query(x) @ self.key(x).transpose(-1, -2) / 4
        score = score.masked_fill(attention_mask == 0, -10000)
        return self.norm(x+score.softmax(-1) @ self.value(x))


class TinyGroup(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = TinyAttention()

    def forward(self, x, attention_mask):
        return (self.attention(x, attention_mask),)


class TinyNativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_groups=1, inner_group_num=1)
        self.embedding = nn.Embedding(32, 16)
        self.albert_layer_groups = nn.ModuleList([TinyGroup()])
        self.decoder = nn.Linear(16, 32)

    def forward(self, input_ids, attention_mask, position_ids, token_type_ids, clslabels_mask, mlmlabels=None, clslabels=None):
        x = self.embedding(input_ids)
        for _ in range(3):
            x = self.albert_layer_groups[0](x, attention_mask)[0]
        logits = self.decoder(x)
        scores = logits[:, :, 1]+clslabels_mask
        loss = 0 if mlmlabels is None else F.cross_entropy(logits.reshape(-1, 32), mlmlabels.reshape(-1), ignore_index=-100)+F.cross_entropy(scores, clslabels)
        return loss, logits, scores


class TinyEncoder:
    def encode(self, item, used_mask=False):
        options = item["choice"]
        ids = [3]
        positions = []
        for choice in options:
            positions.append(len(ids))
            ids.extend([4, 6+sum(map(ord, choice)) % 20])
        ids.extend([5, 10, 11])
        n = len(ids)
        target = [-100]*n
        for j, pos in enumerate(positions):
            target[pos] = 1 if j == item["label"] else 2
        if used_mask:
            ids[-1], target[-1] = 4, 11
        mask = torch.full((n,), -10000.)
        mask[positions] = 0
        return {"input_ids": torch.tensor(ids), "token_type_ids": torch.zeros(n, dtype=torch.long),
                "position_ids": torch.arange(n), "attention_mask": torch.ones(n, n),
                "clslabels_mask": mask, "clslabels": positions[item["label"]],
                "mlmlabels": torch.tensor(target), "mlmlabels_mask": torch.zeros(n)}


def tiny_collator(features):
    result = {}
    for key in features[0]:
        if key == "clslabels":
            result[key] = torch.tensor([r[key] for r in features])
        elif key == "attention_mask":
            length = max(len(r["input_ids"]) for r in features)
            value = torch.zeros(len(features), length, length)
            for i, row in enumerate(features):
                n = len(row["input_ids"])
                value[i, :n, :n] = row[key]
            result[key] = value
        else:
            result[key] = nn.utils.rnn.pad_sequence([r[key] for r in features], batch_first=True,
                                                    padding_value=-10000 if key == "clslabels_mask" else 0)
    return result


def records(split, n=8):
    result = []
    for i in range(n):
        k = 2 if i % 2 else 3
        result.append({"id": f"{split}-{i}", "dataset": "piqa" if k == 2 else "commonsenseqa",
                       "role": f"source_{split}", "content_hash": f"{split}-{i}", "texta": "toy", "question": "",
                       "choices": [chr(97+j+i % 3) for j in range(k)], "gold_label": i % k})
    return result


class LoRAContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_zero_initialization_exactly_preserves_original_output(self):
        base = nn.Linear(16, 16)
        x = torch.randn(3, 4, 16)
        expected = base(x)
        layer = LoRALinear(base, rank=2, alpha=4, dropout=0.05).eval()
        torch.testing.assert_close(layer(x), expected, atol=0, rtol=0)

    def test_shared_group_gets_two_physical_adapters_and_only_ab_is_trainable(self):
        model = TinyNativeModel()
        spec = inject_lora(model, rank=2, alpha=4, dropout=0)
        self.assertEqual(spec["physical_adapters"], 2)
        self.assertEqual(spec["trainable_parameters"], 128)
        self.assertTrue(all(name.endswith(("lora_A", "lora_B")) for name, p in model.named_parameters() if p.requires_grad))
        with self.assertRaisesRegex(ValueError, "already"):
            inject_lora(model)

    def test_checkpointed_shared_group_gets_gradients_without_grad_inputs(self):
        model = TinyNativeModel()
        spec = inject_lora(model, rank=2, alpha=4, dropout=0)
        before = frozen_parameter_hash(model)
        groups = checkpoint_shared_groups(model)
        self.assertEqual(groups, ["albert_layer_groups.0"])
        dataset = OnlineDataset(records("train"), TinyEncoder(), masked=True)
        packed = OnlineCollator(tiny_collator)([dataset[0], dataset[1]])
        from unimc_online import forward_batch
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        for _ in range(2):
            model.train()
            optimizer.zero_grad()
            loss, _ = forward_batch(model, packed, torch.device("cpu"), training=True)
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad))
            optimizer.step()
        self.assertEqual(before, frozen_parameter_hash(model))
        self.assertTrue(any(p.abs().sum() > 0 for name, p in model.named_parameters() if name.endswith("lora_B")))

    def test_padding_labels_ignored_and_probe_width_is_exact(self):
        dataset = OnlineDataset(records("train"), TinyEncoder(), masked=True)
        packed = OnlineCollator(tiny_collator)([dataset[0], dataset[1]])
        small = len(packed["items"][1]["feature"]["input_ids"])
        self.assertTrue((packed["features"]["mlmlabels"][1, small:] == -100).all())
        padded = pad_memory_probe(packed, 32)
        self.assertEqual(padded["features"]["attention_mask"].shape, (2, 32, 32))
        self.assertTrue((padded["features"]["mlmlabels"][:, 20:] == -100).all())
        self.assertTrue((padded["features"]["clslabels_mask"][:, 20:] == -10000).all())

    def test_adapter_restore_and_disable_preserve_base(self):
        model = TinyNativeModel()
        original = copy.deepcopy(model.state_dict())
        inject_lora(model, rank=2, alpha=4, dropout=0)
        state = adapter_state(model)
        for name in state:
            if name.endswith("lora_B"):
                state[name].fill_(0.01)
        restore_adapter(model, state)
        dataset = OnlineDataset(records("val"), TinyEncoder())
        with adapters_disabled(model):
            disabled, _ = predict_online(model, dataset, OnlineCollator(tiny_collator), torch.device("cpu"))
        parent = TinyNativeModel()
        parent.load_state_dict(original)
        reference, _ = predict_online(parent, dataset, OnlineCollator(tiny_collator), torch.device("cpu"))
        self.assertEqual(disabled, reference)
        with self.assertRaises(ValueError):
            restore_adapter(model, {})

    def test_role_and_overlap_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cal.jsonl"
            path.write_text("\n".join(json.dumps(dict(r, role="source_cal")) for r in records("train")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_train"):
                load_records([path], "source_train")
        with self.assertRaisesRegex(ValueError, "overlap"):
            disjoint_records(records("train"), records("train"))

    def test_online_encoding_does_not_mutate_canonical_options(self):
        class MutatingEncoder(TinyEncoder):
            def encode(self, item, used_mask=False):
                item["choice"][0] = "shortened"
                return super().encode(item, used_mask)
        original = records("train")
        snapshot = copy.deepcopy(original)
        dataset = OnlineDataset(original, MutatingEncoder())
        item = dataset[0]
        self.assertTrue(item["options_shortened"])
        self.assertEqual(original, snapshot)

    def test_native_training_export_adapter_reload_and_partial_accumulation(self):
        model = TinyNativeModel()
        base_state = copy.deepcopy(model.state_dict())
        collator, encoder = OnlineCollator(tiny_collator), TinyEncoder()
        train, val = records("train", 10), records("val", 8)
        signature = {"synthetic": True, "hidden_dim": 16}
        runtime = SimpleNamespace(model=model, encoder=encoder, collator=collator, signature=signature)
        reference_rows, reference_metrics = predict_online(model, OnlineDataset(val, encoder), collator, torch.device("cpu"))
        baseline = {(r["dataset"], str(r["id"])): r for r in reference_rows}
        args = SimpleNamespace(seed=42, rank=2, alpha=4., lora_dropout=0., no_gradient_checkpointing=False,
                               micro_batch=1, grad_accum=3, lr=0.01, weight_decay=0.01, epochs=2, patience=1, amp="none")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = train_online(runtime, args, train, val, baseline, reference_metrics, output, {}, torch.device("cpu"))
            self.assertTrue(summary["synthetic_only"])
            history = json.loads((output / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(history[0]["optimizer_updates_total"], 4)  # 10 microbatches, accumulate 3
            restored = TinyNativeModel()
            restored.load_state_dict(base_state)
            other = SimpleNamespace(model=restored, signature=signature)
            load_adapter_for_runtime(other, output / "best_adapter.pt")
            predictions, metrics = predict_online(restored, OnlineDataset(val, encoder), collator, torch.device("cpu"), baseline=baseline)
            saved = [json.loads(r) for r in (output / "best_val.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(predictions, saved)
            self.assertEqual(metrics, summary["best_source_validation"])
            self.assertTrue(all("base_logits" in row for row in predictions))

    def test_optional_official_tiny_albert_native_backward(self):
        try:
            from transformers import AlbertConfig
        except ImportError:
            self.skipTest("Local CPU environment has no transformers; server check runs this with official dependencies")
        root = Path(os.environ.get("FENGSHEN_ROOT", "../Fengshenbang-LM-main")).resolve()
        if not (root / "fengshen").is_dir():
            self.skipTest("Official checkout unavailable")
        sys.path.insert(0, str(root))
        from fengshen.models.albert.modeling_albert import AlbertForMaskedLM
        from fengshen.models.unimc.modeling_unimc import UniMCModel
        config = AlbertConfig(vocab_size=32, embedding_size=8, hidden_size=16, num_hidden_layers=2,
                              num_hidden_groups=1, inner_group_num=1, num_attention_heads=4, intermediate_size=32)
        model = UniMCModel.__new__(UniMCModel)
        nn.Module.__init__(model)
        model.config, model.bert, model.yes_token, model.loss_func = config, AlbertForMaskedLM(config), 1, nn.CrossEntropyLoss()
        inject_lora(model, rank=2, alpha=4, dropout=0)
        checkpoint_shared_groups(model)
        data = OnlineDataset(records("train"), TinyEncoder(), masked=True)
        from unimc_online import forward_batch
        loss, scores = forward_batch(model, OnlineCollator(tiny_collator)([data[0], data[1]]), torch.device("cpu"), training=True)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad))


if __name__ == "__main__":
    unittest.main()
