import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import prepare_targets as prep
import summarize as ev


class AggregateTests(unittest.TestCase):
    def test_frozen_target_pairs(self):
        inputs = prep.ROOT / "data/aggregate_targets"
        if not inputs.is_dir():
            self.skipTest("prepare the public targets first")
        expected = ev.read(Path(__file__).with_name("target_manifest.json"))
        for task, manifest in expected.items():
            path = inputs / (task + ".jsonl")
            self.assertEqual(ev.sha(path), manifest["input_sha256"])
            data = ev.records(path, task)
            self.assertEqual(len(data), sum(manifest["conditions"].values()))
            for r in data.values():
                parent = data["original/" + r["original_id"]]
                order = r["original_indices"]
                self.assertGreaterEqual(len(order), 2)
                self.assertEqual(len(order), len(set(order)))
                self.assertEqual(r["choices"], [parent["choices"][i] for i in order])
                self.assertEqual(r["texta"], parent["texta"])
                self.assertEqual(r["question"], parent["question"])
                if r["condition"] == "rotate":
                    self.assertNotEqual(order, parent["original_indices"])
                if r["condition"] == "drop_gold":
                    self.assertNotIn(parent["gold_label"], order)
                    self.assertIsNone(r["gold_label"])
                else:
                    self.assertEqual(order[r["gold_label"]], parent["gold_label"])

    def test_tied_and_separated_rejection(self):
        for pos, neg, auc, ap in [([1, 1], [1, 1], 0.5, 0.5), ([2], [1], 1, 1),
                                  ([0], [1], 0, 0.5)]:
            got = ev.rejection_metrics(pos, neg)
            self.assertEqual((got["auroc"], got["average_precision"]), (auc, ap))

    def test_seed_pairing(self):
        got = ev.paired_interval([[1 / 3, 1 / 3], [0, 0]], np.random.default_rng(7))
        np.testing.assert_allclose(got, [100 / 6, 100 / 6])

    def test_full_summary_with_shuffled_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {}
            for task in prep.SOURCES:
                item = dict(dataset=task, id="1", choices=["a", "b", "c"], gold_label=1)
                targets, preds = [], []
                for condition, order in prep.variants(item):
                    key = condition + "/1"
                    label = order.index(1) if 1 in order else None
                    scores = [float(i == 1) for i in order]
                    choices = [item["choices"][i] for i in order]
                    targets.append(dict(id=key, original_id="1", dataset=task, condition=condition,
                                        original_indices=order, choices=choices, gold_label=label))
                    preds.append(dict(id=key, dataset=task, prediction=int(np.argmax(scores)), logits=scores,
                                      base_logits=scores, choices=choices, gold_label=label))
                target = root / (task + ".jsonl")
                target.write_text("".join(json.dumps(r) + "\n" for r in targets))
                manifest[task] = {"input_sha256": ev.sha(target)}
                folder = root / task
                folder.mkdir()
                for name in ev.MODELS:
                    (folder / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in reversed(preds)))
            (root / "manifest.json").write_text(json.dumps(manifest))
            ev.summarize(root, root, root / "summary")
            result = ev.read(root / "summary/metrics.json")
            self.assertEqual(result["macro_original"]["baseline"]["accuracy"], 1)
            self.assertEqual(result["centered_full_contrasts"]["baseline"]["paired_95pct_pp"], [0, 0])
            for task in prep.SOURCES:
                metric = result["per_task"][task]["baseline"]
                self.assertEqual(metric["rotate"]["same_original_choice"], 1)
                self.assertEqual(metric["rejection"]["auroc"], 1)
            rows = ev.records(root / "cb.jsonl", "cb")
            baseline = ev.records(root / "cb/baseline.jsonl", "cb")
            broken = copy.deepcopy(baseline)
            broken["rotate/1"]["choices"].reverse()
            with self.assertRaises(ValueError):
                ev.task_scores(rows, broken, baseline)
            del broken["rotate/1"]
            with self.assertRaises(ValueError):
                ev.task_scores(rows, broken, baseline)


if __name__ == "__main__":
    unittest.main()
