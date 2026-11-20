"""Run a saved UniMC LoRA adapter online; frozen feature caches are not inputs."""
import argparse
from pathlib import Path

import torch

from data_io import file_sha256, select_device, write_json
from unimc_lora import KIND, adapters_disabled, definition_hashes, inject_lora, restore_adapter
from unimc_online import OnlineDataset, build_runtime, load_records, predict_online


def load_adapter_for_runtime(runtime, path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["model_kind"] != KIND or saved["base_signature"] != runtime.signature:
        raise ValueError("Adapter/base model or official encoding signature differs")
    if saved["definition_sha256"] != definition_hashes():
        raise ValueError("LoRA/online implementation differs from the saved model")
    spec = inject_lora(runtime.model, **saved["adapter_options"])
    if spec != saved["adapter_spec"]:
        raise ValueError("Actual shared modules differ from the saved adapter")
    restore_adapter(runtime.model, saved["adapter_state"])
    runtime.model.eval()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--role", choices=["source_val", "source_cal", "target_eval"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="../models/unimc-albert-235m")
    parser.add_argument("--fengshen-root", default="../Fengshenbang-LM-main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    runtime = build_runtime(args.model, args.fengshen_root, device, saved["base_signature"]["max_length"])
    load_adapter_for_runtime(runtime, args.checkpoint)
    dataset = OnlineDataset(load_records([args.input], args.role), runtime.encoder, masked=False)
    with adapters_disabled(runtime.model):
        original, _ = predict_online(runtime.model, dataset, runtime.collator, device, progress=True)
    baseline = {(row["dataset"], str(row["id"])): row for row in original}
    _, metrics = predict_online(runtime.model, dataset, runtime.collator, device, output=output, baseline=baseline, progress=True)
    write_json(str(output)+".metadata.json", {"uncalibrated": True, "role": args.role,
               "checkpoint_sha256": file_sha256(args.checkpoint), "input_sha256": file_sha256(args.input),
               "base_signature": runtime.signature, "evaluation_amp": "none", "definition_sha256": definition_hashes()})
    print(metrics)


if __name__ == "__main__":
    main()
