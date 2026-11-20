"""Small LoRA adapters for the *shared* query/value modules of official ALBERT."""
import contextlib
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from data_io import file_sha256

KIND = "unimc_shared_qv_lora_v1"


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16.0, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 0 < rank <= min(base.in_features, base.out_features):
            raise ValueError("Expected a linear layer and valid LoRA rank")
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid alpha/dropout")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha, self.scaling = rank, alpha, alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.enabled = True

    def forward(self, inputs):
        result = self.base(inputs)
        if self.enabled:
            result = result + self.scaling * F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return result


def inject_lora(model, rank=8, alpha=16.0, dropout=0.05):
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("Model already contains LoRA")
    # ALBERT repeats physical layer groups. named_modules deduplicates the same
    # physical group, so a logical 12-layer pass does not create 12 adapters.
    targets = [(name, module) for name, module in model.named_modules()
               if name.endswith((".attention.query", ".attention.value"))]
    if not targets:
        raise ValueError("No official ALBERT attention.query/value targets found")
    model.requires_grad_(False)
    physical = {}
    specs = []
    for name, layer in targets:
        if not isinstance(layer, nn.Linear):
            raise ValueError(f"Unexpected target type: {name}")
        wrapper = physical.get(id(layer))
        if wrapper is None:
            wrapper = LoRALinear(layer, rank, alpha, dropout)
            physical[id(layer)] = wrapper
        parent_name, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), attribute, wrapper)
        specs.append({"name": name, "in_features": layer.in_features, "out_features": layer.out_features,
                      "rank": rank, "alpha": alpha, "dropout": dropout})
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any(not name.endswith((".lora_A", ".lora_B")) for name, _ in trainable):
        raise ValueError("Only LoRA A/B parameters may be trainable")
    return {"targets": specs, "physical_adapters": len(physical),
            "trainable_parameters": sum(p.numel() for _, p in trainable)}


@contextlib.contextmanager
def adapters_disabled(model):
    modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    states = [m.enabled for m in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, state in zip(modules, states):
            module.enabled = state


def adapter_state(model):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()
            if name.endswith((".lora_A", ".lora_B"))}


def restore_adapter(model, state):
    expected = adapter_state(model)
    if state.keys() != expected.keys():
        raise ValueError("Adapter parameter names differ")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != parameters[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid adapter tensor: {name}")
            parameters[name].copy_(value.to(parameters[name]))


def frozen_parameter_hash(model):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            digest.update(name.encode())
            digest.update(param.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_shared_groups(model):
    groups = [(name, module) for name, module in model.named_modules()
              if "." in name and name.rsplit(".", 1)[0].endswith("albert_layer_groups")
              and name.rsplit(".", 1)[-1].isdigit()]
    if not groups:
        raise ValueError("No physical ALBERT layer groups found")
    for _, group in groups:
        if getattr(group, "_unimc_lora_checkpointed", False):
            continue
        original = group.forward

        def wrap(original_forward, owner):
            def forward(*args, **kwargs):
                if owner.training and torch.is_grad_enabled():
                    return checkpoint(original_forward, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
                return original_forward(*args, **kwargs)
            return forward
        group.forward = wrap(original, group)
        group._unimc_lora_checkpointed = True
    return [name for name, _ in groups]


def definition_hashes():
    root = Path(__file__).parent
    return {name: file_sha256(root / name) for name in ("unimc_lora.py", "unimc_online.py")}
