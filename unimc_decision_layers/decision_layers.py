"""
decision_layers.py -- the new decision layers that sit on top of the frozen UniMC.

Input  (per sample, M = number of option slots, padded):
    feats : (B, M, 4096)  option representation at each [O-MASK]
    raw   : (B, M)        UniMC raw "yes" logit of each option
    mask  : (B, M) bool   True = real option, False = padding
Output:
    logits: (B, M)        final score per option, -inf at padding.
                          final = raw + correction   (residual scoring, unless residual=False)

Four variants, all share the same input projection 4096 -> d (default 256) and the same
scoring head d -> 1, so that they differ ONLY in the block in the middle:

    mlp         : per-option residual MLP.                 no interaction between options
    tf_noint    : 1 Transformer encoder layer whose attention mask is the identity:
                  every option can attend only to itself.  SAME parameter count as tf_int.
                  -> answers "does extra depth alone help?"
    tf_int      : same layer, options attend to all valid options of the same sample.
                  -> answers "does letting options compare each other help?"
    tf_gate     : tf_int + a learned gate g in (0,1) per option that scales how much of the
                  interaction output is added:  h <- h + g * Attn(h)   (g from [h ; Attn(h)])
                  -> answers "is controlling the interaction strength better?"

Design choices (keep in the report):
  * The last linear layer of the scoring head is zero-initialised, so at step 0 every model
    outputs exactly the UniMC raw logits -> training starts from the baseline and can only
    move away from it if the data says so.
  * The raw logit is also fed back in as a feature (use_raw_feat=True): the module knows how
    confident the original model was.
  * Padding options are excluded from attention (key padding mask), from the loss (logit = -inf)
    and from normalisation.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class InputProj(nn.Module):
    def __init__(self, in_dim=4096, d=256, dropout=0.1, use_raw_feat=True):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d)
        self.use_raw_feat = use_raw_feat
        if use_raw_feat:
            self.raw_proj = nn.Linear(1, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, feats, raw, mask):
        h = self.proj(self.norm(feats.float()))
        if self.use_raw_feat:
            # standardise the raw logits inside each sample (over valid options) before using them
            r = raw.masked_fill(~mask, 0.0)
            n = mask.sum(-1, keepdim=True).clamp(min=1).float()
            mu = r.sum(-1, keepdim=True) / n
            var = ((r - mu) ** 2 * mask).sum(-1, keepdim=True) / n
            z = ((r - mu) / (var.sqrt() + 1e-3)) * mask
            h = h + self.raw_proj(z.unsqueeze(-1))
        return self.drop(F.gelu(h))


class ScoreHead(nn.Module):
    def __init__(self, d=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return self.net(h).squeeze(-1)


class OptionBlock(nn.Module):
    """One pre-LN Transformer encoder layer over the option axis, with optional gate."""
    def __init__(self, d=256, heads=4, ff=512, dropout=0.1, interact=True, gate=False):
        super().__init__()
        self.interact, self.gate = interact, gate
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
        if gate:
            self.gate_net = nn.Linear(2 * d, 1)
            nn.init.zeros_(self.gate_net.weight)
            nn.init.constant_(self.gate_net.bias, 0.0)   # g = sigmoid(0) = 0.5 at start
        self.last_gate = None

    def forward(self, h, mask):
        B, M, _ = h.shape
        x = self.ln1(h)
        if self.interact:
            attn_mask = None
        else:
            # identity attention: option i may only look at option i (same params, no interaction)
            attn_mask = ~torch.eye(M, dtype=torch.bool, device=h.device)
        # key padding: never attend to padded options.  Padded *queries* would then have no
        # valid key when interact=False -> give them themselves to avoid NaN.
        kpm = ~mask
        if not self.interact:
            attn_mask = attn_mask.clone()
            kpm = torch.zeros_like(kpm)      # identity mask already restricts keys; padded rows are dropped later
        a, _ = self.attn(x, x, x, key_padding_mask=kpm, attn_mask=attn_mask, need_weights=False)
        a = self.drop(a)
        if self.gate:
            g = torch.sigmoid(self.gate_net(torch.cat([h, a], -1)))   # (B, M, 1)
            self.last_gate = g.detach()
            h = h + g * a
        else:
            h = h + a
        h = h + self.drop(self.ffn(self.ln2(h)))
        return h * mask.unsqueeze(-1)


class DecisionLayer(nn.Module):
    KINDS = ("mlp", "tf_noint", "tf_int", "tf_gate")

    def __init__(self, kind="tf_int", in_dim=4096, d=256, heads=4, ff=512, dropout=0.1,
                 use_raw_feat=True, residual=True):
        super().__init__()
        assert kind in self.KINDS, kind
        self.kind, self.residual = kind, residual
        self.inp = InputProj(in_dim, d, dropout, use_raw_feat)
        if kind == "mlp":
            # residual MLP block of comparable size (per option, no interaction)
            self.block = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        else:
            self.block = OptionBlock(d, heads, ff, dropout,
                                     interact=(kind != "tf_noint"), gate=(kind == "tf_gate"))
        self.head = ScoreHead(d, dropout)
        self.cfg = dict(kind=kind, in_dim=in_dim, d=d, heads=heads, ff=ff, dropout=dropout,
                        use_raw_feat=use_raw_feat, residual=residual)

    def forward(self, feats, raw, mask):
        h = self.inp(feats, raw, mask)
        if self.kind == "mlp":
            h = h + self.block(h)
        else:
            h = self.block(h, mask)
        corr = self.head(h)
        logits = raw.float() + corr if self.residual else corr
        return logits.masked_fill(~mask, float("-inf"))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------------------
# inference interface for the other members (E: evaluation, F: demo)
# ---------------------------------------------------------------------------------------
class DecisionScorer:
    """
    scorer = DecisionScorer.load("runs/tf_int/seed0/best.pt", device="cuda")
    logits = scorer.score(feats, raw, mask)      # torch tensors, shapes as above
    probs  = logits.softmax(-1)                  # padding -> prob 0
    """
    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu")
        m = DecisionLayer(**ck["cfg"])
        m.load_state_dict(ck["state_dict"])
        return cls(m, device)

    @torch.no_grad()
    def score(self, feats, raw, mask):
        feats = torch.as_tensor(feats).to(self.device)
        raw = torch.as_tensor(raw).to(self.device).float()
        mask = torch.as_tensor(mask).to(self.device).bool()
        if feats.dim() == 2:          # single sample
            return self.model(feats[None], raw[None], mask[None])[0]
        return self.model(feats, raw, mask)
