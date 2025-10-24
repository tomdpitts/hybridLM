# =========================
# model.py — Stage‑1 AR (Encoder LM)
# =========================
# Shared tokenizer: SentencePiece (spm.model)
# Special tokens assumed:
#   [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3, and optional [MASK]
#
# Exposes:
#   ARLanguageModel: standard causal LM that also provides `encode()` → latent z
#
# Usage (see train_ar.py):
#   from model import ARLanguageModel, ARConfig
#



from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------
@dataclass
class ARConfig:
    vocab_size: int
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    max_seq_len: int = 1024
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    rope: bool = False  # plain sinusoidal pos emb by default
    latent_dim: int = 512  # z dimension returned by encode()

# -----------------------------
# Positional embeddings
# -----------------------------
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        return self.pe[:L].unsqueeze(0)

# -----------------------------
# Attention block
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, 4096, 4096)), persistent=False)
    def forward(self, x, attn_mask=None):
        B, L, C = x.size()
        q = self.query(x).view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(x).view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        causal = self.mask[:, :, :L, :L]
        att = att.masked_fill(causal == 0, float("-inf"))
        if attn_mask is not None:
            # attn_mask: (B, L) 1 for real tokens, 0 for pad
            m = attn_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
            att = att.masked_fill(m == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, n_head, L, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.fc(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, eps):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, eps=eps)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd, eps=eps)
        self.mlp = MLP(n_embd, dropout)
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

# -----------------------------
# AR Language Model with latent encoder
# -----------------------------
class ARLanguageModel(nn.Module):
    def __init__(self, cfg: ARConfig, pad_id: int = 0, eos_id: int = 3, bos_id: int = 2):
        super().__init__()
        self.cfg = cfg
        self.pad_id, self.eos_id, self.bos_id = pad_id, eos_id, bos_id
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = SinusoidalPositionalEmbedding(cfg.n_embd, cfg.max_seq_len)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, cfg.dropout, cfg.layer_norm_eps) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # latent projection
        self.z_proj = nn.Linear(cfg.n_embd, cfg.latent_dim)
        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, attn_mask=None):
        # idx: (B, L) token ids
        B, L = idx.shape
        assert L <= self.cfg.max_seq_len
        x = self.tok_emb(idx)  # (B, L, D)
        x = x + self.pos_emb(x)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def encode(self, idx, attn_mask=None, pool: str = "mean"):
        """Return a global latent z for the sequence.
        pool: 'mean' over non-PAD tokens, or 'last' (use last non-PAD / EOS position).
        """
        self.eval()
        B, L = idx.shape
        x = self.tok_emb(idx)
        x = x + self.pos_emb(x)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.ln_f(x)  # (B, L, D)
        if attn_mask is None:
            attn_mask = (idx != self.pad_id).long()
        if pool == "mean":
            denom = attn_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (x * attn_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:  # 'last'
            # find last non-pad, prefer EOS if present
            last_idx = (attn_mask * torch.arange(L, device=idx.device)).argmax(dim=1)
            gathered = x[torch.arange(B, device=idx.device), last_idx]
            pooled = gathered
        z = self.z_proj(pooled)
        return z  # (B, latent_dim)