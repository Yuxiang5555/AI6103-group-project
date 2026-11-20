"""Residual option scorers. No option-index embeddings or fixed label vocabulary."""
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class HeadConfig:
    hidden_dim: int = 4096
    model_dim: int = 256
    heads: int = 4
    feedforward_dim: int = 1024
    dropout: float = 0.1
    mode: str = "full"  # mlp, self, full
    gate: bool = False

    def to_dict(self):
        return asdict(self)


class ResidualOptionHead(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        if config.mode not in {"mlp", "self", "full"}:
            raise ValueError("mode must be mlp, self, or full")
        if config.model_dim % config.heads:
            raise ValueError("model_dim must be divisible by heads")
        self.config = config
        d = config.model_dim
        self.input_norm = nn.LayerNorm(config.hidden_dim)
        self.project = nn.Linear(config.hidden_dim, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.drop1 = nn.Dropout(config.dropout)
        self.drop2 = nn.Dropout(config.dropout)
        if config.mode == "mlp":
            self.pointwise = nn.Sequential(nn.Linear(d, d), nn.GELU())
        else:
            self.attention = nn.MultiheadAttention(
                d, config.heads, dropout=config.dropout, batch_first=True
            )
        self.ffn = nn.Sequential(
            nn.Linear(d, config.feedforward_dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.feedforward_dim, d),
        )
        self.output_norm = nn.LayerNorm(d)
        self.scorer = nn.Linear(d, 1)
        # Exact baseline at initialization, for both valid logits and predictions.
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)
        if config.gate:
            # This gate controls correction magnitude, not only cross-option attention.
            self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def _blocked_attention(self, valid):
        b, k = valid.shape
        allowed = valid[:, None, :].expand(b, k, k).clone()
        if self.config.mode == "self":
            identity = torch.eye(k, dtype=torch.bool, device=valid.device)[None]
            # A padded query can read valid keys; its output is discarded. This avoids
            # all-masked softmax rows, which can create NaNs during backward.
            allowed &= identity | ~valid[:, :, None]
        # MHA boolean mask: True = blocked. Key padding is already included here.
        return (~allowed).repeat_interleave(self.config.heads, dim=0)

    def forward(self, hidden, base_logits, valid, return_delta=False):
        if hidden.ndim != 3 or hidden.shape[:2] != base_logits.shape:
            raise ValueError("expected hidden [B,K,H] and base_logits [B,K]")
        if valid.dtype != torch.bool or valid.shape != base_logits.shape:
            raise ValueError("valid must be a bool tensor [B,K]")
        if not valid.any(dim=1).all():
            raise ValueError("every example must have at least one valid option")
        # Cast cached half-precision features back for the small head.
        z = self.project(self.input_norm(hidden.float()))
        z = z.masked_fill(~valid[:, :, None], 0.0)
        a = self.norm1(z)
        if self.config.mode == "mlp":
            message = self.pointwise(a)
        else:
            message, _ = self.attention(
                a, a, a, attn_mask=self._blocked_attention(valid), need_weights=False
            )
        u = z + self.drop1(message)
        u = u + self.drop2(self.ffn(self.norm2(u)))
        delta = self.scorer(self.output_norm(u)).squeeze(-1)
        if self.config.gate:
            delta = delta * self.gate_logit.sigmoid()
        delta = delta.masked_fill(~valid, 0.0)
        logits = (base_logits.float() + delta).masked_fill(~valid, -torch.inf)
        return (logits, delta) if return_delta else logits
