"""Reuse the frozen official MLM predecoder on cached UniMC option vectors.

The original base logits, sample order, choices, labels, and roles are preserved.
This runs the small MLM transform, never the ALBERT encoder. No labels are fitted.
"""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from data_io import FeatureDataset, file_sha256, select_device, write_json


class FrozenMlmPredecoder(nn.Module):
    def __init__(self, mlm_head):
        super().__init__()
        self.dense = mlm_head.dense
        self.activation = mlm_head.activation
        self.norm = mlm_head.LayerNorm
        self.input_dim = self.dense.in_features
        self.output_dim = self.dense.out_features
        self.requires_grad_(False)
        self.eval()

    def forward(self, hidden):
        return self.norm(self.activation(self.dense(hidden.float())))


def verify_predecoder(mlm_head, projector, hidden, yes_token):
    """Compare against the decoder input and output of the actual official head."""
    captured = {}
    def capture(module, args):
        captured["input"] = args[0]
    hook = mlm_head.decoder.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            reference_yes = mlm_head(hidden)[:, yes_token]
            projected = projector(hidden)
            torch.testing.assert_close(projected, captured["input"], atol=1e-6, rtol=1e-6)
            bias = None if mlm_head.decoder.bias is None else mlm_head.decoder.bias[yes_token:yes_token + 1]
            yes = F.linear(projected, mlm_head.decoder.weight[yes_token:yes_token + 1], bias).squeeze(-1)
            torch.testing.assert_close(yes, reference_yes, atol=2e-4, rtol=2e-4)
    finally:
        hook.remove()
    return {"passed": True, "n_option_vectors": len(hidden),
            "scope": "Official MLM forward on cached vectors, not a fresh encoder forward"}


def project_cache(cache, projector, decoder, transform_spec, device, batch_size=128,
                  source_path="", source_sha256=""):
    if batch_size < 1 or cache.get("schema_version") != 1 or not cache.get("samples"):
        raise ValueError("Need positive batch size and a nonempty schema-1 cache")
    parent_signature = cache["metadata"]["feature_signature"]
    if "option_representation" in parent_signature:
        raise ValueError("Expected original encoder features, not an already projected cache")
    if parent_signature["hidden_dim"] != projector.input_dim:
        raise ValueError("Source features and official MLM input dimensions differ")
    yes_token = parent_signature["yes_token"]
    if not 0 <= yes_token < decoder.out_features or decoder.in_features != projector.output_dim:
        raise ValueError("MLM decoder dimensions/yes token do not match")
    samples = []
    sum_error, max_error, n_options, same_predictions = 0.0, 0.0, 0, 0
    with torch.inference_mode():
        for start in range(0, len(cache["samples"]), batch_size):
            original = cache["samples"][start:start + batch_size]
            hidden = torch.cat([row["hidden"] for row in original]).to(device=device, dtype=torch.float32)
            projected = projector(hidden)
            if not torch.isfinite(projected).all():
                raise FloatingPointError("Non-finite MLM features")
            bias = None if decoder.bias is None else decoder.bias[yes_token:yes_token + 1]
            reconstructed = F.linear(projected, decoder.weight[yes_token:yes_token + 1], bias).squeeze(-1).cpu()
            projected = projected.to(dtype=torch.float16).cpu()
            if not torch.isfinite(projected).all():
                raise FloatingPointError("Non-finite FP16 MLM features")
            offset = 0
            for row in original:
                k = len(row["choices"])
                yes = reconstructed[offset:offset + k]
                error = (yes - row["base_logits"].float()).abs()
                sum_error += error.sum().item()
                max_error = max(max_error, error.max().item())
                same_predictions += int(yes.argmax() == row["base_logits"].argmax())
                n_options += k
                samples.append(dict(row, hidden=projected[offset:offset + k].clone()))
                offset += k
    metadata = copy.deepcopy(cache["metadata"])
    signature = copy.deepcopy(parent_signature)
    signature["hidden_dim"] = projector.output_dim
    signature["option_representation"] = {
        **transform_spec,
        "parent_feature_signature": copy.deepcopy(parent_signature),
        "parent_cached_hidden_dtype": "float16",
    }
    metadata["feature_signature"] = signature
    metadata["n"] = len(samples)
    metadata["feature_projection"] = {
        "source_cache": str(source_path), "source_cache_sha256": source_sha256,
        "base_logits_preserved": True, "label_independent": True,
        "input_hidden_dim": projector.input_dim, "output_hidden_dim": projector.output_dim,
        "reconstructed_yes_mean_abs_error": sum_error / n_options,
        "reconstructed_yes_max_abs_error": max_error,
        "reconstructed_yes_prediction_agreement": same_predictions / len(samples),
        "reconstruction_note": "Diagnostics use FP16 cached encoder vectors; original base logits are retained exactly.",
    }
    if "official_prediction_check" in metadata:
        metadata["official_prediction_check"]["inherited_from_original_base_logits"] = True
    return {"schema_version": 1, "metadata": metadata, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory, same cache basenames")
    parser.add_argument("--model", required=True, help="Original local UniMC model directory")
    parser.add_argument("--fengshen-root", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("Threads/batch size must be positive")
    torch.set_num_threads(args.threads)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    inputs = [Path(path) for path in args.cache]
    if len({path.name for path in inputs}) != len(inputs):
        raise ValueError("Cache basenames must be unique")
    validated = FeatureDataset(inputs)
    signature = validated.signature
    model_dir = Path(args.model).resolve()
    if file_sha256(model_dir / "config.json") != signature["config_sha256"]:
        raise ValueError("Original cache/model config differs")
    for filename, digest in signature["weights_sha256"].items():
        if file_sha256(model_dir / filename) != digest:
            raise ValueError(f"Original cache/model weights differ: {filename}")
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    sys.path.insert(0, str(Path(args.fengshen_root).resolve()))
    from fengshen.models.albert.modeling_albert import AlbertForMaskedLM
    model = AlbertForMaskedLM.from_pretrained(str(model_dir)).eval()
    mlm_head = model.predictions
    del model  # Retain the small MLM head, release the encoder without running it.
    mlm_head.requires_grad_(False)
    mlm_head.to(device).eval()
    projector = FrozenMlmPredecoder(mlm_head).to(device)
    if signature["hidden_dim"] != projector.input_dim:
        raise ValueError("Official head input dimension differs from the raw cache")
    probes = torch.cat([row["hidden"] for row in validated.samples[:4]]).to(device=device, dtype=torch.float32)
    verification = verify_predecoder(mlm_head, projector, probes, signature["yes_token"])
    print("OFFICIAL MLM LAYER CHECK PASSED", flush=True)
    transform_spec = {
        "kind": "frozen_official_mlm_predecoder_v1",
        "official_mlm_class_sha256": hashlib.sha256(inspect.getsource(type(mlm_head)).encode()).hexdigest(),
        "converter_sha256": file_sha256(__file__),
    }
    del validated, probes
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for path in inputs:
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if cache["metadata"]["feature_signature"] != signature:
            raise ValueError("Raw feature signatures differ")
        projected = project_cache(cache, projector, mlm_head.decoder, transform_spec, device,
                                  args.batch_size, path, file_sha256(path))
        projected["metadata"]["feature_projection"]["official_mlm_verification"] = verification
        target = output / path.name
        torch.save(projected, target)
        write_json(str(target) + ".metadata.json", projected["metadata"])
        summary = {"file": str(target), "n": len(projected["samples"]),
                   **projected["metadata"]["feature_projection"]}
        summaries.append(summary)
        print("PROJECTED " + json.dumps(summary), flush=True)
    write_json(output / "projection_manifest.json", {"transform": transform_spec,
                                                    "verification": verification, "files": summaries})
    print("FROZEN MLM PROJECTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
