"""A shared-weight self/full attention mixture; legacy residual gate is unchanged."""
import math
from pathlib import Path

import torch
from torch import nn

from data_io import file_sha256
from decision_head import HeadConfig, ResidualOptionHead

MODEL_KIND = "self_full_interaction_gate_v1"


class InteractionGatedOptionHead(ResidualOptionHead):
    def __init__(self, config: HeadConfig, gate_init=0.5):
        if config.mode != "full" or config.gate:
            raise ValueError("Interaction gate requires mode=full and legacy gate=false")
        if not math.isfinite(gate_init) or not 0 < gate_init < 1:
            raise ValueError("gate_init must be strictly between zero and one")
        super().__init__(config)
        # Adding a deterministic scalar after super() preserves all shared RNG initialization.
        self.interaction_logit = nn.Parameter(torch.tensor(math.log(gate_init / (1 - gate_init))))

    def gate_value(self):
        return float(self.interaction_logit.detach().sigmoid().cpu())

    def _self_mask(self, valid):
        b, k = valid.shape
        allowed = valid[:, None, :].expand(b, k, k).clone()
        identity = torch.eye(k, dtype=torch.bool, device=valid.device)[None]
        allowed &= identity | ~valid[:, :, None]
        return (~allowed).repeat_interleave(self.config.heads, dim=0)

    def forward(self, hidden, base_logits, valid, return_delta=False, gate_override=None):
        if hidden.ndim != 3 or hidden.shape[:2] != base_logits.shape:
            raise ValueError("expected hidden [B,K,H] and base_logits [B,K]")
        if valid.dtype != torch.bool or valid.shape != base_logits.shape:
            raise ValueError("valid must be a bool tensor [B,K]")
        if not valid.any(dim=1).all():
            raise ValueError("every example must have at least one valid option")
        if gate_override is not None and (not math.isfinite(gate_override) or not 0 <= gate_override <= 1):
            raise ValueError("gate_override must be in [0,1]")
        z = self.project(self.input_norm(hidden.float())).masked_fill(~valid[:, :, None], 0.0)
        a = self.norm1(z)
        # One attention module, evaluated with two masks. No second set of weights.
        # Exact fixed endpoints are for implementation checks, not trained controls.
        if gate_override == 0:
            message, _ = self.attention(a, a, a, attn_mask=self._self_mask(valid), need_weights=False)
        elif gate_override == 1:
            message, _ = self.attention(a, a, a, attn_mask=self._blocked_attention(valid), need_weights=False)
        else:
            self_message, _ = self.attention(a, a, a, attn_mask=self._self_mask(valid), need_weights=False)
            full_message, _ = self.attention(a, a, a, attn_mask=self._blocked_attention(valid), need_weights=False)
            gate = self.interaction_logit.sigmoid() if gate_override is None else gate_override
            message = self_message + gate * (full_message - self_message)
        u = z + self.drop1(message)
        u = u + self.drop2(self.ffn(self.norm2(u)))
        delta = self.scorer(self.output_norm(u)).squeeze(-1).masked_fill(~valid, 0.0)
        logits = (base_logits.float() + delta).masked_fill(~valid, -torch.inf)
        return (logits, delta) if return_delta else logits


def definition_hashes():
    return {"interaction_gate_head.py": file_sha256(__file__),
            "decision_head.py": file_sha256(Path(__file__).with_name("decision_head.py"))}


def load_gate_checkpoint(path, signature, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_kind") != MODEL_KIND:
        raise ValueError("Not an interaction-gate checkpoint")
    if checkpoint["feature_signature"] != signature:
        raise ValueError("Checkpoint/cache extraction signatures differ")
    if checkpoint["definition_sha256"] != definition_hashes():
        raise ValueError("Interaction-gate or parent head code differs from the saved model")
    model = InteractionGatedOptionHead(HeadConfig(**checkpoint["head_config"]), checkpoint["gate_init"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
