"""
unimc_core.py  --  Minimal re-implementation of the official UniMC inference path.

Why this file exists
--------------------
The official code (Fengshenbang-LM/fengshen/models/unimc/modeling_unimc.py) pulls in
pytorch_lightning, deepspeed-style checkpoints, a private copy of ALBERT, etc.  For our
project we only need three things from it:

  1. `encode()`      : how a sample {texta, question, choice, label} becomes model input
                       ([O-MASK] before every option, block-diagonal attention mask so
                       options cannot see each other, per-option position ids).
                       The code below is copied from the official UniMCDataset.encode /
                       get_att_mask / get_position_ids, unchanged in logic.
  2. the forward pass: standard HF AlbertForMaskedLM, but the attention mask is 3-D
                       (batch, L, L) instead of (batch, L).  We therefore call the ALBERT
                       sub-modules ourselves and build the additive 4-D mask by hand.
  3. the score       : for every [O-MASK] position p, the logit of the word "yes" from the
                       MLM head is the option's raw score.  Softmax over the [O-MASK]
                       positions gives the official probabilities.

What we export per option (the "unified handoff format" of the group document)
  - option representation : ALBERT last hidden state at the [O-MASK] position (4096-d)
  - raw logit             : the "yes" logit at that position (float)
  - valid mask            : True for real options, False for padding
"""
import copy
import json
import numpy as np
import torch
from torch import nn
from transformers import AlbertForMaskedLM, AlbertTokenizer


# --------------------------------------------------------------------------------------
# 1. Encoding (copied from the official UniMCDataset)
# --------------------------------------------------------------------------------------
class UniMCEncoder:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # official: yes_token = tokenizer.encode('yes')[1]  ("▁yes" for the English ALBERT)
        self.yes_token = tokenizer.encode('yes')[1]
        self.no_token = tokenizer.encode('no')[1]

    # ---- official helpers -------------------------------------------------------------
    def get_position_ids(self, label_idx, max_length, question_len):
        question_position_ids = np.arange(question_len)
        label_position_ids = np.arange(question_len, label_idx[-1])
        for i in range(len(label_idx) - 1):
            label_position_ids[label_idx[i] - question_len:label_idx[i + 1] - question_len] = np.arange(
                question_len, question_len + label_idx[i + 1] - label_idx[i])
        max_len_label = max(label_position_ids)
        text_position_ids = np.arange(
            max_len_label + 1, max_length + max_len_label + 1 - label_idx[-1])
        position_ids = list(question_position_ids) + \
            list(label_position_ids) + list(text_position_ids)
        if max_length <= 512:
            return position_ids[:max_length]
        else:
            for i in range(512, max_length):
                if position_ids[i] > 511:
                    position_ids[i] = 511
            return position_ids[:max_length]

    def get_att_mask(self, attention_mask, label_idx, question_len):
        max_length = len(attention_mask)
        attention_mask = np.array(attention_mask)
        attention_mask = np.tile(attention_mask[None, :], (max_length, 1))
        zeros = np.zeros(shape=(label_idx[-1] - question_len, label_idx[-1] - question_len))
        attention_mask[question_len:label_idx[-1], question_len:label_idx[-1]] = zeros
        for i in range(len(label_idx) - 1):
            label_token_length = label_idx[i + 1] - label_idx[i]
            if label_token_length <= 0:
                continue
            ones = np.ones(shape=(label_token_length, label_token_length))
            attention_mask[label_idx[i]:label_idx[i + 1], label_idx[i]:label_idx[i + 1]] = ones
        return attention_mask

    # ---- official encode (inference branch, used_mask=False) --------------------------
    def encode(self, item):
        item = copy.deepcopy(item)
        while len(self.tokenizer.encode('[MASK]'.join(item['choice']))) > self.max_length - 32:
            item['choice'] = [c[:int(len(c) / 2)] for c in item['choice']]

        if 'textb' in item.keys() and item['textb'] != '':
            if 'question' in item.keys() and item['question'] != '':
                texta = '[MASK]' + '[MASK]'.join(item['choice']) + '[SEP]' + \
                    item['question'] + '[SEP]' + item['texta'] + '[SEP]' + item['textb']
            else:
                texta = '[MASK]' + '[MASK]'.join(item['choice']) + '[SEP]' + \
                    item['texta'] + '[SEP]' + item['textb']
        else:
            if 'question' in item.keys() and item['question'] != '':
                texta = '[MASK]' + '[MASK]'.join(item['choice']) + '[SEP]' + \
                    item['question'] + '[SEP]' + item['texta']
            else:
                texta = '[MASK]' + '[MASK]'.join(item['choice']) + '[SEP]' + item['texta']

        encode_dict = self.tokenizer.encode_plus(texta, max_length=self.max_length,
                                                 padding='max_length', truncation='longest_first')
        encode_sent = encode_dict['input_ids']
        attention_mask = encode_dict['attention_mask']
        sample_max_length = sum(attention_mask)

        if 'label' not in item.keys():
            item['label'] = 0

        question_len = 1
        label_idx = [question_len]
        for choice in item['choice']:
            cur_mask_idx = label_idx[-1] + len(self.tokenizer.encode(choice, add_special_tokens=False)) + 1
            label_idx.append(cur_mask_idx)

        token_type_ids = [0] * question_len + [1] * (label_idx[-1] - label_idx[0] + 1) + [0] * self.max_length
        token_type_ids = token_type_ids[:self.max_length]
        attention_mask = self.get_att_mask(attention_mask, label_idx, question_len)
        position_ids = self.get_position_ids(label_idx, self.max_length, question_len)

        source = np.array(encode_sent)
        source[label_idx[:-1]] = self.tokenizer.mask_token_id

        # [补充] safety: the official code can produce position ids > 511 for very long
        # inputs with long options (it would crash there); we clamp instead.
        position_ids = np.minimum(np.array(position_ids), 511)

        return {
            "input_ids": torch.tensor(source[:sample_max_length]).long(),
            "token_type_ids": torch.tensor(token_type_ids[:sample_max_length]).long(),
            "attention_mask": torch.tensor(attention_mask[:sample_max_length, :sample_max_length]).float(),
            "position_ids": torch.tensor(position_ids[:sample_max_length]).long(),
            "option_pos": torch.tensor(label_idx[:-1]).long(),   # positions of the [O-MASK] tokens
            "label": int(item['label']),
            "n_opt": len(item['choice']),
        }

    @staticmethod
    def collate(batch):
        B = len(batch)
        L = max(b['input_ids'].shape[0] for b in batch)
        M = max(b['n_opt'] for b in batch)
        out = {
            "input_ids": torch.zeros(B, L, dtype=torch.long),
            "token_type_ids": torch.zeros(B, L, dtype=torch.long),
            "position_ids": torch.zeros(B, L, dtype=torch.long),
            "attention_mask": torch.zeros(B, L, L),
            "option_pos": torch.zeros(B, M, dtype=torch.long),
            "option_mask": torch.zeros(B, M, dtype=torch.bool),
            "label": torch.tensor([b['label'] for b in batch], dtype=torch.long),
            "n_opt": torch.tensor([b['n_opt'] for b in batch], dtype=torch.long),
        }
        for i, b in enumerate(batch):
            l = b['input_ids'].shape[0]
            out['input_ids'][i, :l] = b['input_ids']
            out['token_type_ids'][i, :l] = b['token_type_ids']
            out['position_ids'][i, :l] = b['position_ids']
            out['attention_mask'][i, :l, :l] = b['attention_mask']
            m = b['n_opt']
            out['option_pos'][i, :m] = b['option_pos']
            out['option_mask'][i, :m] = True
        return out


# --------------------------------------------------------------------------------------
# 2. Model wrapper: frozen official UniMC, returns option representations + raw logits
# --------------------------------------------------------------------------------------
class UniMCFeatureModel(nn.Module):
    def __init__(self, model_path, yes_token):
        super().__init__()
        self.bert = AlbertForMaskedLM.from_pretrained(model_path, attn_implementation="eager")
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad_(False)
        self.yes_token = yes_token
        self.hidden_size = self.bert.config.hidden_size   # 4096 for the 235M English model

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, token_type_ids, position_ids, option_pos, option_mask, **_):
        """
        attention_mask: (B, L, L) float, 1 = may attend.  Same semantics as the official code:
        the official private ALBERT does  ext = (1 - mask[:, None, :, :]) * -10000.
        """
        albert = self.bert.albert
        ext = (1.0 - attention_mask[:, None, :, :]).to(dtype=self.bert.dtype) * -10000.0
        emb = albert.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        head_mask = [None] * albert.config.num_hidden_layers
        enc = albert.encoder(emb, ext, head_mask=head_mask, return_dict=True)
        seq = enc[0]                                                     # (B, L, 4096)
        # gather the [O-MASK] positions
        idx = option_pos[:, :, None].expand(-1, -1, seq.shape[-1])
        opt_rep = torch.gather(seq, 1, idx)                              # (B, M, 4096)
        # MLM head only on the gathered positions (saves memory vs. full vocab logits)
        mlm_logits = self.bert.predictions(opt_rep)                      # (B, M, vocab)
        raw = mlm_logits[:, :, self.yes_token].float()                   # (B, M)
        raw = raw.masked_fill(~option_mask, -1e4)
        return opt_rep, raw


def load_unimc(model_path, device="cuda", dtype=torch.float32, max_length=512):
    tok = AlbertTokenizer.from_pretrained(model_path)
    enc = UniMCEncoder(tok, max_length=max_length)
    model = UniMCFeatureModel(model_path, enc.yes_token).to(device=device, dtype=dtype)
    return tok, enc, model


def official_probs(raw_logits, option_mask):
    """softmax over valid [O-MASK] positions == official UniMCPredict scores."""
    raw = raw_logits.masked_fill(~option_mask, float('-inf'))
    return torch.softmax(raw, dim=-1)
