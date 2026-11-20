"""Subtract each example's option-vector mean before the existing head's LayerNorm.

This is a label-independent set operation. Even self/MLP then use set information;
self only removes cross-option attention, not the shared centering operation.
"""
import argparse
import copy
from pathlib import Path

import torch

from data_io import FeatureDataset, file_sha256, write_json


def center_cache(cache, converter_sha256):
    parent = cache["metadata"]["feature_signature"]
    if "option_centering" in parent:
        raise ValueError("Cache is already centered")
    if cache.get("schema_version") != 1 or not cache.get("samples"):
        raise ValueError("Expected a nonempty schema-1 feature cache")
    samples = []
    with torch.inference_mode():
        for row in cache["samples"]:
            hidden = row["hidden"].float()
            if hidden.ndim != 2 or len(hidden) != len(row["choices"]) or len(hidden) < 2:
                raise ValueError("Invalid option-vector shapes")
            if not torch.isfinite(hidden).all():
                raise ValueError("Non-finite original option vectors")
            centered = hidden - hidden.mean(dim=0, keepdim=True)
            samples.append(dict(row, hidden=centered.contiguous()))
    metadata = copy.deepcopy(cache["metadata"])
    signature = copy.deepcopy(parent)
    signature["cached_hidden_dtype"] = "float32"
    signature["option_centering"] = {
        "kind": "subtract_mean_over_actual_options_v1", "converter_sha256": converter_sha256,
        "parent_feature_signature": copy.deepcopy(parent),
    }
    metadata["feature_signature"] = signature
    metadata["n"] = len(samples)
    metadata["option_centering"] = {"base_logits_preserved": True, "labels_used": False,
                                     "axis": "Actual options within each example; before head LayerNorm"}
    return {"schema_version": 1, "metadata": metadata, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = [Path(path) for path in args.cache]
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Input cache basenames must be unique")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    FeatureDataset(paths)  # Validate all inputs and their shared feature signature.
    output.mkdir(parents=True, exist_ok=False)
    digest = file_sha256(__file__)
    manifest = {"converter_sha256": digest, "operation": "Label-independent option centering",
                "files": []}
    for path in paths:
        cache = torch.load(path, map_location="cpu", weights_only=True)
        centered = center_cache(cache, digest)
        centered["metadata"]["option_centering"].update(source_cache=str(path), source_sha256=file_sha256(path))
        target = output / path.name
        torch.save(centered, target)
        write_json(str(target) + ".metadata.json", centered["metadata"])
        manifest["files"].append({"path": str(target), "n": len(centered["samples"]),
                                  "sha256": file_sha256(target)})
        print(f"CENTERED {target} n={len(centered['samples'])}", flush=True)
    write_json(output / "centering_manifest.json", manifest)
    print("OPTION CENTERING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
