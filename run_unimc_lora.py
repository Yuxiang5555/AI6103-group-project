"""Check, then train shared-Q/V LoRA on the original UniMC native objective."""
import argparse
import copy
import importlib.metadata
import json
import math
import platform
from collections import Counter
from pathlib import Path
import time
import zipfile

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_io import file_sha256, seed_everything, select_device, write_json
from unimc_lora import (KIND, adapter_state, adapters_disabled, checkpoint_shared_groups, definition_hashes,
                        frozen_parameter_hash, inject_lora, restore_adapter)
from unimc_online import (OnlineDataset, amp_context, build_runtime, disjoint_records, forward_batch,
                          load_records, metrics_from_predictions, predict_online)

CODE_FILES = ("unimc_lora.py", "unimc_online.py", "run_unimc_lora.py", "predict_unimc_lora.py", "collect_unimc_lora.py")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "train"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default="data/pilot_seed42_v2")
    parser.add_argument("--model", default="../models/unimc-albert-235m")
    parser.add_argument("--fengshen-root", default="../Fengshenbang-LM-main")
    parser.add_argument("--reference-metadata", default="cache/pilot_55518/piqa_val.pt.metadata.json")
    parser.add_argument("--reference-predictions", default="runs/centering_confirm_55757/predictions_resume_55764/baseline_val.jsonl")
    parser.add_argument("--check-report", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_reference(path, records):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline = {(row["dataset"], str(row["id"])): row for row in rows}
    if len(baseline) != len(rows) or baseline.keys() != {(r["dataset"], str(r["id"])) for r in records}:
        raise ValueError("Reference baseline ID set differs")
    for row in records:
        previous = baseline[(row["dataset"], str(row["id"]))]
        if row["choices"] != previous["choices"] or row["gold_label"] != previous["gold_label"]:
            raise ValueError("Reference choices/labels differ")
        if previous["logits"] != previous["base_logits"]:
            raise ValueError("Expected original baseline logits")
    return baseline, metrics_from_predictions(rows)


def pad_memory_probe(packed, length):
    source = packed["features"]
    n = source["input_ids"].shape[1]
    if n > length:
        raise ValueError("Probe exceeds configured maximum")
    result = dict(packed, features={})
    for key, value in source.items():
        if key == "clslabels":
            result["features"][key] = value
        elif key == "attention_mask":
            out = value.new_zeros(value.shape[0], length, length)
            out[:, :n, :n] = value
            result["features"][key] = out
        else:
            fill = -100 if key == "mlmlabels" else -10000 if key == "clslabels_mask" else 0
            out = value.new_full((value.shape[0], length), fill)
            out[:, :n] = value
            result["features"][key] = out
    return result


def adapter_options(args):
    return {"rank": args.rank, "alpha": args.alpha, "dropout": args.lora_dropout}


def code_hashes():
    return {name: file_sha256(Path(__file__).with_name(name)) for name in CODE_FILES}


def initialize(runtime, args):
    seed_everything(args.seed)
    spec = inject_lora(runtime.model, **adapter_options(args))
    expected = 2 * runtime.model.config.num_hidden_groups * runtime.model.config.inner_group_num
    if spec["physical_adapters"] != expected:
        raise ValueError("Unexpected number of physical ALBERT Q/V modules")
    groups = [] if args.no_gradient_checkpointing else checkpoint_shared_groups(runtime.model)
    return spec, groups


def initialization_parity(runtime, val, baseline, device):
    chosen = [row for task in ("piqa", "commonsenseqa") for row in [r for r in val if r["dataset"] == task][:4]]
    if len(chosen) != 8:
        raise ValueError("Need four validation examples from each source task for the parity check")
    dataset = OnlineDataset(chosen, runtime.encoder, masked=False)
    with adapters_disabled(runtime.model):
        before, _ = predict_online(runtime.model, dataset, runtime.collator, device)
    after, _ = predict_online(runtime.model, dataset, runtime.collator, device)
    max_official_error = 0.
    for a, b in zip(before, after):
        torch.testing.assert_close(torch.tensor(a["logits"]), torch.tensor(b["logits"]), atol=0, rtol=0)
        previous = baseline[(a["dataset"], str(a["id"]))]
        torch.testing.assert_close(torch.tensor(a["logits"]), torch.tensor(previous["logits"]), atol=5e-4, rtol=1e-5)
        if a["prediction"] != previous["prediction"]:
            raise ValueError("Original baseline prediction differs during initialization check")
        max_official_error = max(max_official_error, max(abs(x-y) for x, y in zip(a["logits"], previous["logits"])))
    return {"n": len(chosen), "zero_adapter_exactly_preserves_base": True,
            "max_original_export_logit_error": max_official_error}


def check_feasibility(runtime, args, train, val, baseline, output, data_hashes, device):
    spec, groups = initialize(runtime, args)
    parity = initialization_parity(runtime, val, baseline, device)
    frozen_before = frozen_parameter_hash(runtime.model)
    initial = adapter_state(runtime.model)
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    longest = sorted(train, key=lambda row: len(row["texta"])+len(row.get("textb", ""))+sum(map(len, row["choices"])), reverse=True)
    dataset = OnlineDataset(longest[:args.micro_batch], runtime.encoder, masked=True)
    packed = pad_memory_probe(runtime.collator([dataset[i] for i in range(len(dataset))]), args.max_length)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    steps = []
    for step in range(2):
        runtime.model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with amp_context(device, args.amp):
            loss, _ = forward_batch(runtime.model, packed, device, training=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite native training loss")
        loss.backward()
        gradients = {name: float(p.grad.detach().float().norm().cpu()) for name, p in runtime.model.named_parameters() if p.requires_grad and p.grad is not None}
        if not gradients or not any(value > 0 for value in gradients.values()) or not all(math.isfinite(v) for v in gradients.values()):
            raise FloatingPointError("No finite nonzero adapter gradients")
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        steps.append({"step": step+1, "loss": float(loss.detach().cpu()), "seconds": time.perf_counter()-started,
                      "adapter_gradient_norms": gradients})
    if frozen_before != frozen_parameter_hash(runtime.model):
        raise AssertionError("Frozen base parameters changed")
    final = adapter_state(runtime.model)
    changed = [name for name in initial if not torch.equal(initial[name], final[name])]
    if not changed:
        raise AssertionError("Adapters did not update")
    dataset = OnlineDataset(val[:2], runtime.encoder, masked=False)
    with adapters_disabled(runtime.model):
        restored, _ = predict_online(runtime.model, dataset, runtime.collator, device)
    for row in restored:
        torch.testing.assert_close(torch.tensor(row["logits"]), torch.tensor(baseline[(row["dataset"], str(row["id"]))]["logits"]), atol=5e-4, rtol=1e-5)
    report = {"passed": True, "scope": "Memory/gradient feasibility; two optimizer steps, not a performance experiment",
              "base_signature": runtime.signature, "source_data_sha256": data_hashes, "code_sha256": code_hashes(),
              "adapter_options": adapter_options(args), "amp": args.amp, "micro_batch": args.micro_batch,
              "max_length": args.max_length, "gradient_checkpointing": not args.no_gradient_checkpointing,
              "adapter_spec": spec, "checkpointed_physical_groups": groups, "initialization_parity": parity,
              "base_parameters_unchanged": True, "changed_adapter_parameters": changed,
              "padded_probe_sequence_length": packed["features"]["input_ids"].shape[1], "steps": steps,
              "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device)/(1024**3) if device.type == "cuda" else None,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "source_cal_used": False, "target_data_used": False, "torch": str(torch.__version__)}
    write_json(output / "check_report.json", report)
    archive = Path.cwd().parent / "UniMC_lora_check.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(output / "check_report.json", "check_report.json")
    print(f"TRAINABLE_PARAMETERS={spec['trainable_parameters']}", flush=True)
    print(f"PEAK_GPU_GIB={report['peak_gpu_allocated_gib']}", flush=True)
    print(f"CHECK_REPORT={output/'check_report.json'}", flush=True)
    print("UNIMC LORA CHECK COMPLETE", flush=True)


def matching_check(args, signature, data_hashes):
    candidates = [Path(args.check_report)] if args.check_report != "auto" else sorted(
        Path("runs").glob("lora_check_*/check_report.json"), key=lambda p:p.stat().st_mtime, reverse=True)
    for path in candidates:
        report = read_json(path)
        if (report.get("passed") and report["base_signature"] == signature and report["source_data_sha256"] == data_hashes
                and report["code_sha256"] == code_hashes() and report["adapter_options"] == adapter_options(args)
                and report["amp"] == args.amp and report["micro_batch"] == args.micro_batch
                and report["max_length"] == args.max_length
                and report["gradient_checkpointing"] == (not args.no_gradient_checkpointing)):
            return str(path)
    raise ValueError("Run the matching Slurm feasibility check first; no compatible passing report found")


def train_online(runtime, args, train, val, baseline_rows, baseline_metrics, output, data_hashes, device):
    spec, groups = initialize(runtime, args)  # fresh B=0; never reuse check-job updates
    parity = initialization_parity(runtime, val, baseline_rows, device)
    dataset = OnlineDataset(train, runtime.encoder, masked=True)
    validation = OnlineDataset(val, runtime.encoder, masked=False)
    counts = Counter(row["dataset"] for row in train)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler([1/counts[row["dataset"]] for row in train], len(train), replacement=True, generator=generator)
    loader = DataLoader(dataset, batch_size=args.micro_batch, sampler=sampler, generator=generator,
                        collate_fn=runtime.collator, num_workers=0)
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    write_json(output / "config.json", {
        "model_kind": KIND, "arguments": vars(args), "adapter_options": adapter_options(args), "adapter_spec": spec,
        "definition_sha256": definition_hashes(), "code_sha256": code_hashes(), "base_signature": runtime.signature,
        "source_data_sha256": data_hashes, "initialization_parity": parity,
        "loss": "Official UniMC OP + masked-token CE (option yes/no O-MLM and randomly masked passage MLM); padding labels corrected to -100",
        "selection": "source macro_accuracy; tie-break macro_nll", "baseline_source_validation": baseline_metrics,
        "checkpointed_physical_groups": groups, "source_cal_used": False, "target_data_used": False,
        "environment": {"torch": str(torch.__version__), "python": platform.python_version(),
                        "transformers": importlib.metadata.version("transformers") if not runtime.signature.get("synthetic") else "synthetic-test"},
    })

    def save(name, epoch, metrics):
        torch.save({"model_kind": KIND, "adapter_state": adapter_state(runtime.model), "adapter_options": adapter_options(args),
                    "adapter_spec": spec, "base_signature": runtime.signature, "definition_sha256": definition_hashes(),
                    "seed": args.seed, "epoch": epoch, "source_validation": metrics}, output / name)

    save("initial_adapter.pt", 0, baseline_metrics)
    best, stale, history, updates = (-float("inf"), -float("inf")), 0, [], 0
    for epoch in range(1, args.epochs+1):
        runtime.model.train()
        optimizer.zero_grad(set_to_none=True)
        started, total_loss, n, grad_max = time.perf_counter(), 0., 0, 0.
        for i, packed in enumerate(loader):
            group_start = (i//args.grad_accum)*args.grad_accum
            group_examples = min(args.grad_accum*args.micro_batch, len(train)-group_start*args.micro_batch)
            batch_n = len(packed["items"])
            with amp_context(device, args.amp):
                loss, _ = forward_batch(runtime.model, packed, device, training=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite native training loss")
            (loss*(batch_n/group_examples)).backward()
            total_loss += float(loss.detach().cpu())*batch_n
            n += batch_n
            if (i+1) % args.grad_accum == 0 or i+1 == len(loader):
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                grad_max = max(grad_max, float(norm.detach().cpu()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            if (i+1) % 100 == 0:
                print(f"LORA TRAIN epoch={epoch} examples={n}/{len(train)} updates={updates} loss={total_loss/n:.5f}", flush=True)
        print(f"LORA VALIDATING epoch={epoch}", flush=True)
        rows, metrics = predict_online(runtime.model, validation, runtime.collator, device, baseline=baseline_rows, progress=True)
        row = {"epoch": epoch, "train_native_loss": total_loss/n, "seconds": time.perf_counter()-started,
               "optimizer_updates_total": updates, "max_preclip_gradient_norm": grad_max,
               "source_validation": metrics,
               "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device)/(1024**3) if device.type == "cuda" else None}
        history.append(row)
        write_json(output / "history.json", history)
        save("last_adapter.pt", epoch, metrics)
        score = (metrics["macro_accuracy"], -metrics["macro_nll"])
        if score > best:
            best, stale = score, 0
            save("best_adapter.pt", epoch, metrics)
            with (output / "best_val.jsonl").open("w", encoding="utf-8") as stream:
                for record in rows:
                    stream.write(json.dumps(record, ensure_ascii=False)+"\n")
            write_json(output / "best_val.jsonl.metrics.json", metrics)
            write_json(output / "best_val.jsonl.metadata.json", {"uncalibrated": True, "role": "source_val",
                       "checkpoint_sha256": file_sha256(output / "best_adapter.pt"), "evaluation_amp": "none",
                       "base_signature": runtime.signature, "definition_sha256": definition_hashes()})
        else:
            stale += 1
        print(f"LORA EPOCH {epoch} Macro={100*metrics['macro_accuracy']:.4f}% NLL={metrics['macro_nll']:.6f}", flush=True)
        if stale >= args.patience:
            break
    selected = torch.load(output / "best_adapter.pt", map_location="cpu", weights_only=True)
    if not any(torch.count_nonzero(value) > 0 for key, value in selected["adapter_state"].items() if key.endswith("lora_B")):
        raise AssertionError("No learned LoRA B parameters in best checkpoint")
    summary = {"seed": args.seed, "best_epoch": selected["epoch"], "trainable_parameters": spec["trainable_parameters"],
               "baseline_source_validation": baseline_metrics, "best_source_validation": selected["source_validation"],
               "delta_macro_pp": 100*(selected["source_validation"]["macro_accuracy"]-baseline_metrics["macro_accuracy"]),
               "source_cal_used": False, "target_data_used": False, "synthetic_only": bool(runtime.signature.get("synthetic"))}
    write_json(output / "summary.json", summary)
    return summary


def package_run(output, seed):
    archive = Path.cwd().parent / f"UniMC_lora_seed{seed}_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in output.iterdir():
            if path.is_file():
                bundle.write(path, path.name)
        for name in CODE_FILES + ("data_io.py",):
            bundle.write(name, "code/"+name)
        bundle.write("README_UniMC_LoRA实施.md", "README.md")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip():
            raise ValueError("Result archive CRC failed")
    return archive


def main():
    args = parse_args()
    if min(args.epochs, args.patience, args.micro_batch, args.grad_accum, args.threads) < 1 or args.lr <= 0:
        raise ValueError("Training counts/rate must be positive")
    if args.max_length != 512:
        raise ValueError("First controlled experiment keeps the original 512-token setting")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    if device.type != "cuda":
        raise ValueError("Run the real 235M experiment on the allocated CUDA compute node")
    torch.cuda.set_device(0 if device.index is None else device.index)
    # Validate support before loading the large model.
    amp_context(device, args.amp)
    train_paths = [str(Path(args.data_root)/task/"train.jsonl") for task in ("piqa", "cqa")]
    val_paths = [str(Path(args.data_root)/task/"val.jsonl") for task in ("piqa", "cqa")]
    train, val = load_records(train_paths, "source_train"), load_records(val_paths, "source_val")
    disjoint_records(train, val)
    if (Counter(r["dataset"] for r in train) != {"piqa": 2000, "commonsenseqa": 2000}
            or Counter(r["dataset"] for r in val) != {"piqa": 919, "commonsenseqa": 598}):
        raise ValueError("Expected the unchanged 4000/1517 pilot split")
    data_hashes = {path: file_sha256(path) for path in train_paths+val_paths}
    baseline_rows, baseline_metrics = read_reference(args.reference_predictions, val)
    runtime = build_runtime(args.model, args.fengshen_root, device, args.max_length)
    if runtime.signature != read_json(args.reference_metadata)["feature_signature"]:
        raise ValueError("Base model/tokenizer/official encoding differs from the original experiment")
    check = None if args.mode == "check" else matching_check(args, runtime.signature, data_hashes)
    output.mkdir(parents=True, exist_ok=False)
    if args.mode == "check":
        check_feasibility(runtime, args, train, val, baseline_rows, output, data_hashes, device)
    else:
        print(f"MATCHED_FEASIBILITY_CHECK={check}", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        summary = train_online(runtime, args, train, val, baseline_rows, baseline_metrics, output, data_hashes, device)
        archive = package_run(output, args.seed)
        print(f"LORA RESULT seed={args.seed} Macro={100*summary['best_source_validation']['macro_accuracy']:.4f}% Delta={summary['delta_macro_pp']:+.4f}pp", flush=True)
        print(f"REPORT_ARCHIVE={archive}", flush=True)
        print("UNIMC LORA TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
