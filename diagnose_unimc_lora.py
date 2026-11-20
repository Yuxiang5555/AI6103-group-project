"""Read-only gradient decomposition on fixed source-train examples; no optimizer."""
import argparse
import csv
import json
from pathlib import Path
import random
import statistics
import zipfile

import torch
from torch.nn import functional as F

from data_io import file_sha256, seed_everything, select_device, write_json
from predict_unimc_lora import load_adapter_for_runtime
from unimc_lora import adapter_state, checkpoint_shared_groups, frozen_parameter_hash, restore_adapter
from unimc_online import OnlineDataset, amp_context, build_runtime, load_records


def loss_components(logits, labels, cls_mask, cls_labels, yes_token):
    valid = labels != -100
    option = valid & (cls_mask == 0)
    passage = valid & ~option
    n_option, n_passage = int(option.sum()), int(passage.sum())
    if n_option < len(labels) or not valid.any():
        raise ValueError("Expected supervised option markers")
    option_loss = F.cross_entropy(logits[option].float(), labels[option])
    passage_loss = F.cross_entropy(logits[passage].float(), labels[passage]) if n_passage else logits.new_zeros((), dtype=torch.float32)
    total = n_option+n_passage
    weighted_option, weighted_passage = option_loss*(n_option/total), passage_loss*(n_passage/total)
    op = F.cross_entropy(logits[:, :, yes_token].float()+cls_mask, cls_labels)
    return {"op": op, "option_mlm": option_loss, "passage_mlm": passage_loss,
            "weighted_option_mlm": weighted_option, "weighted_passage_mlm": weighted_passage,
            "mlm": weighted_option+weighted_passage, "native": op+weighted_option+weighted_passage,
            "option_targets": n_option, "passage_targets": n_passage}


def audit_loss_precision(logits, labels, components, official_loss, strict_official=True):
    """Validate the partition against a full FP32 CE on the same logits/targets.

    A low-precision scalar is recorded separately; its dtype/value alone cannot
    identify why a remote model's loss differs from an FP32 reconstruction.
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        reference = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                    labels.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(components['mlm'], reference, atol=2e-5, rtol=1e-5)
    if not torch.isfinite(official_loss).all():
        raise FloatingPointError('Non-finite official MLM loss')
    delta = float((official_loss.float()-reference).abs().detach())
    matches = bool(torch.isclose(official_loss.float(), reference, atol=2e-5, rtol=1e-5))
    result = {'logits_dtype': str(logits.dtype), 'official_loss_dtype': str(official_loss.dtype),
              'official_mlm_loss': float(official_loss.detach()), 'full_fp32_mlm_loss': float(reference.detach()),
              'partition_fp32_absolute_error': float((components['mlm']-reference).abs().detach()),
              'official_vs_fp32_absolute_error': delta, 'official_matches_fp32_tolerance': matches}
    if strict_official and not matches:
        raise AssertionError('Official loss differs from same-logit FP32 CE: '+json.dumps(result))
    return result


def gradient_vector(loss, parameters, retain_graph):
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    result = torch.cat([(g.detach().float() if g is not None else torch.zeros_like(p).float()).reshape(-1)
                        for p, g in zip(parameters, grads)])
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite component gradient")
    return result


def cosine(a, b):
    denominator = float(a.norm()*b.norm())
    return float((a@b)/denominator) if denominator > 0 else None


def summarize(rows):
    result = []
    for checkpoint in ("initial", "best", "last"):
        for task in ("piqa", "commonsenseqa", "all"):
            selected = [r for r in rows if r["checkpoint"] == checkpoint and (task == "all" or r["dataset"] == task)]
            fields = ("op_loss", "option_mlm_loss", "passage_mlm_loss", "native_mlm_loss", "native_loss",
                      "weighted_option_mlm_loss", "weighted_passage_mlm_loss",
                      "op_gradient_norm", "option_mlm_weighted_gradient_norm", "passage_mlm_weighted_gradient_norm",
                      "mlm_gradient_norm", "native_gradient_norm", "mlm_to_op_gradient_ratio",
                      "op_option_mlm_gradient_cosine", "op_passage_mlm_gradient_cosine",
                      "option_passage_mlm_gradient_cosine", "op_mlm_gradient_cosine", "op_native_gradient_cosine")
            values = {key: statistics.mean(r[key] for r in selected if r[key] is not None)
                      if any(r[key] is not None for r in selected) else None for key in fields}
            comparable = [r["op_mlm_gradient_cosine"] for r in selected if r["op_mlm_gradient_cosine"] is not None]
            values.update(checkpoint=checkpoint, dataset=task, n=len(selected),
                          examples_with_passage_targets=sum(r["passage_targets"] > 0 for r in selected),
                          negative_op_mlm_cosine_count=sum(v < 0 for v in comparable),
                          comparable_cosine_count=len(comparable))
            ratios = [r['mlm_to_op_gradient_ratio'] for r in selected if r['mlm_to_op_gradient_ratio'] is not None]
            values['median_mlm_to_op_gradient_ratio'] = statistics.median(ratios) if ratios else None
            result.append(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/lora_seed42_55840")
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples-per-task", type=int, default=16)
    parser.add_argument("--sampling-seed", type=int, default=20261118)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.examples_per_task < 1:
        raise ValueError("Positive sample count required")
    output, folder, device = Path(args.output), Path(args.run_dir), select_device(args.device)
    if output.exists():
        raise FileExistsError(output)
    if device.type != "cuda":
        raise ValueError("Use an allocated GPU compute node")
    torch.cuda.set_device(0 if device.index is None else device.index)
    torch.set_num_threads(4)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    source_root = Path(config["arguments"]["data_root"])
    paths = [str(source_root/task/"train.jsonl") for task in ("piqa", "cqa")]
    if any(file_sha256(path) != config["source_data_sha256"][path] for path in paths):
        raise ValueError("Source-train file hashes differ from the completed run")
    records = load_records(paths, "source_train")
    chosen = []
    rng = random.Random(args.sampling_seed)
    for task in ("piqa", "commonsenseqa"):
        pool = [r for r in records if r["dataset"] == task]
        if len(pool) < args.examples_per_task:
            raise ValueError("Insufficient source training examples")
        chosen.extend(rng.sample(pool, args.examples_per_task))
    runtime = build_runtime(config["arguments"]["model"], config["arguments"]["fengshen_root"], device,
                            config["arguments"]["max_length"])
    load_adapter_for_runtime(runtime, folder / "initial_adapter.pt")
    if not config["arguments"]["no_gradient_checkpointing"]:
        checkpoint_shared_groups(runtime.model)
    # Encode once so every checkpoint sees exactly the same sampled input masks.
    seed_everything(args.sampling_seed)
    dataset = OnlineDataset(chosen, runtime.encoder, masked=True)
    batches = [runtime.collator([dataset[i]]) for i in range(len(dataset))]
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    base_before = frozen_parameter_hash(runtime.model)
    rows, checkpoint_hashes = [], {}
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats(device)
    # Preserve the failed mixed-precision comparison as evidence, rather than
    # assuming its cause or merely loosening a tolerance. These are no-update
    # forward probes on the first fixed source-train input, with matched seeds.
    first = batches[0]['features']
    first_inputs = {key: first[key].to(device) for key in
                    ('input_ids', 'attention_mask', 'position_ids', 'token_type_ids')}
    precision_probe = []
    for mode in dict.fromkeys((config['arguments']['amp'], 'none')):
        seed_everything(args.sampling_seed)
        runtime.model.train()
        with torch.no_grad(), amp_context(device, mode):
            original = runtime.model.bert(**first_inputs, labels=first['mlmlabels'].to(device))
        with torch.autocast(device_type=device.type, enabled=False):
            parts = loss_components(original.logits, first['mlmlabels'].to(device),
                                    first['clslabels_mask'].to(device), first['clslabels'].to(device),
                                    runtime.model.yes_token)
            probe = audit_loss_precision(original.logits, first['mlmlabels'].to(device), parts,
                                         original.loss, strict_official=False)
        probe.update(forward_amp=mode, dataset=chosen[0]['dataset'], id=chosen[0]['id'],
                     probe_scope='No-grad first-example forward; not a replay of an optimizer step')
        precision_probe.append(probe)
        print('PRECISION PROBE '+json.dumps(probe), flush=True)
    del original, parts, first_inputs
    write_json(output / 'precision_probe.json', precision_probe)
    for name in ("initial", "best", "last"):
        path = folder / f"{name}_adapter.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["base_signature"] != runtime.signature or saved["definition_sha256"] != config["definition_sha256"]:
            raise ValueError("Checkpoint signature differs")
        restore_adapter(runtime.model, saved["adapter_state"])
        before = adapter_state(runtime.model)
        checkpoint_hashes[name] = file_sha256(path)
        for i, packed in enumerate(batches):
            seed_everything(args.sampling_seed+i)
            runtime.model.train()
            source = packed["features"]
            keys = ("input_ids", "attention_mask", "position_ids", "token_type_ids")
            inputs = {key: source[key].to(device) for key in keys}
            labels = source["mlmlabels"].to(device)
            cls_mask, cls_labels = source["clslabels_mask"].to(device), source["clslabels"].to(device)
            # Use FP32 for the entire diagnostic, not only the reconstructed CE.
            # This is an explicitly FP32 post-hoc analysis, not a BF16 step replay.
            with torch.autocast(device_type=device.type, enabled=False):
                original = runtime.model.bert(**inputs, labels=labels)
                if original.logits.dtype != torch.float32:
                    raise TypeError(f'Expected FP32 diagnostic logits, got {original.logits.dtype}')
                losses = loss_components(original.logits, labels, cls_mask, cls_labels, runtime.model.yes_token)
                precision = audit_loss_precision(original.logits, labels, losses, original.loss)
            g_op = gradient_vector(losses["op"], parameters, retain_graph=True)
            has_text = losses["passage_targets"] > 0
            g_option = gradient_vector(losses["weighted_option_mlm"], parameters, retain_graph=has_text)
            g_passage = gradient_vector(losses["weighted_passage_mlm"], parameters, retain_graph=False) if has_text else torch.zeros_like(g_op)
            g_mlm, g_native = g_option+g_passage, g_op+g_option+g_passage
            record = packed["items"][0]["record"]
            op_norm = float(g_op.norm())
            rows.append({"checkpoint": name, "epoch": saved["epoch"], "dataset": record["dataset"], "id": record["id"],
                         **precision,
                         "option_targets": losses["option_targets"], "passage_targets": losses["passage_targets"],
                         "op_loss": float(losses["op"].detach()), "option_mlm_loss": float(losses["option_mlm"].detach()),
                         "passage_mlm_loss": float(losses["passage_mlm"].detach()), "native_mlm_loss": float(losses["mlm"].detach()),
                         "weighted_option_mlm_loss": float(losses["weighted_option_mlm"].detach()),
                         "weighted_passage_mlm_loss": float(losses["weighted_passage_mlm"].detach()),
                         "native_loss": float(losses["native"].detach()), "op_gradient_norm": op_norm,
                         "option_mlm_weighted_gradient_norm": float(g_option.norm()),
                         "passage_mlm_weighted_gradient_norm": float(g_passage.norm()), "mlm_gradient_norm": float(g_mlm.norm()),
                         "native_gradient_norm": float(g_native.norm()),
                         "mlm_to_op_gradient_ratio": float(g_mlm.norm())/op_norm if op_norm else None,
                         "op_option_mlm_gradient_cosine": cosine(g_op, g_option),
                         "op_passage_mlm_gradient_cosine": cosine(g_op, g_passage),
                         "option_passage_mlm_gradient_cosine": cosine(g_option, g_passage),
                         "op_mlm_gradient_cosine": cosine(g_op, g_mlm), "op_native_gradient_cosine": cosine(g_op, g_native)})
            if (i+1) % 8 == 0:
                print(f"LOSS DIAGNOSTIC checkpoint={name} examples={i+1}/{len(batches)}", flush=True)
        after = adapter_state(runtime.model)
        if any(not torch.equal(before[key], after[key]) for key in before):
            raise AssertionError("Read-only diagnostic changed adapter parameters")
    if base_before != frozen_parameter_hash(runtime.model):
        raise AssertionError("Read-only diagnostic changed frozen parameters")
    result = {"scope": "Post-hoc component-gradient diagnostic on fixed source-train examples, not a new training or causal performance test",
              "run_dir": str(folder), "run_config_sha256": file_sha256(folder / "config.json"),
              "checkpoint_sha256": checkpoint_hashes, "diagnostic_code_sha256": file_sha256(__file__),
              "diagnostic_forward_amp": "none", "diagnostic_loss_dtype": "float32",
              "original_training_amp": config['arguments']['amp'], "precision_probe": precision_probe,
              "examples_per_task": args.examples_per_task, "sampling_seed": args.sampling_seed,
              "same_input_masks_and_dropout_seed_across_checkpoints": True, "source_val_examples_used": False,
              "source_cal_used": False, "target_data_used": False, "optimizer_steps": 0,
              "all_parameters_unchanged_within_each_diagnostic": True,
              "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device)/(1024**3), "summary": summarize(rows), "rows": rows}
    write_json(output / "diagnostic.json", result)
    with (output / "per_example.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# UniMC LoRA: fixed source-train gradient diagnostic", "",
             "No optimizer, no validation/calibration/target examples, no parameter updates.", "",
             "Negative cosine describes local objective gradients on this small fixed pool; it does not establish a cause of validation accuracy or prove OP-only training will improve it.", ""]
    for group in result["summary"]:
        line = (f"{group['checkpoint']} {group['dataset']} n={group['n']} OP={group['op_loss']:.5f} "
                f"MLM={group['native_mlm_loss']:.5f} negative_cos={group['negative_op_mlm_cosine_count']}/{group['comparable_cosine_count']}")
        lines.append("- "+line)
        print("COMPONENT "+line, flush=True)
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    archive = Path.cwd().parent / "UniMC_lora_diagnostics.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in output.iterdir():
            bundle.write(path, path.name)
        bundle.write(__file__, "diagnose_unimc_lora.py")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip():
            raise ValueError("Diagnostic archive CRC failed")
    print(f"REPORT_ARCHIVE={archive}", flush=True)
    print("UNIMC LORA LOSS DIAGNOSTIC COMPLETE", flush=True)


if __name__ == "__main__":
    main()
