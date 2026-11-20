"""CPU-only synthetic integration test. It is NOT a UniMC performance experiment."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parent
    generator = torch.Generator().manual_seed(7)
    with tempfile.TemporaryDirectory(prefix="unimc-head-smoke-") as temp:
        temp = Path(temp)
        paths = {}
        for name, role, n in (("train", "source_train", 64), ("val", "source_val", 24), ("eval", "target_eval", 24)):
            samples = []
            for i in range(n):
                k = (2, 3, 5)[i % 3]
                h = torch.randn(k, 16, generator=generator)
                samples.append({"id": f"{name}:{i}", "dataset": f"toy-{i % 2}", "choices": [str(j) for j in range(k)],
                                "gold_label": int(h[:, 0].argmax()), "content_hash": f"{name}:{i}",
                                "hidden": h.half(), "base_logits": torch.randn(k, generator=generator) * 0.05})
            path = temp / f"{name}.pt"
            torch.save({"schema_version": 1, "metadata": {"role": role, "feature_signature": {"hidden_dim": 16, "synthetic": True}},
                        "samples": samples}, path)
            paths[name] = path
        checked = []
        for mode in ("mlp", "self", "full"):
            run = temp / mode
            command = [sys.executable, str(root / "train_head.py"), "--train", str(paths["train"]),
                       "--val", str(paths["val"]), "--mode", mode, "--output", str(run),
                       "--model-dim", "16", "--heads", "4", "--ffn-dim", "32", "--dropout", "0",
                       "--epochs", "3", "--patience", "3", "--batch-size", "16", "--device", "cpu"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            output = temp / f"{mode}.jsonl"
            subprocess.run([sys.executable, str(root / "predict_head.py"), "--cache", str(paths["eval"]),
                            "--checkpoint", str(run / "best.pt"), "--output", str(output), "--device", "cpu"],
                           check=True, capture_output=True, text=True)
            rows = [json.loads(x) for x in output.read_text().splitlines()]
            assert len(rows) == 24
            assert all(len(row["logits"]) == row["num_options"] for row in rows)
            assert all(0 <= row["prediction"] < row["num_options"] for row in rows)
            initial = torch.load(run / "initial.pt", weights_only=True)
            best = torch.load(run / "best.pt", weights_only=True)
            assert not torch.equal(initial["state_dict"]["scorer.weight"], best["state_dict"]["scorer.weight"])
            checked.append(mode)
        subprocess.run([sys.executable, str(root / "predict_head.py"), "--cache", str(paths["eval"]),
                        "--baseline", "--output", str(temp / "baseline.jsonl"), "--device", "cpu"],
                       check=True, capture_output=True, text=True)
        print(json.dumps({"passed": True, "synthetic_only": True, "trained_exported_modes": checked,
                          "checks": ["train", "source selection", "save/load", "target logits export", "baseline export"]}, indent=2))


if __name__ == "__main__":
    main()
