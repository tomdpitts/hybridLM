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
        # self.z_proj = nn.Linear(cfg.n_embd, cfg.latent_dim) # deprecated design choice I'm abandoning
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
    def encodeOLD(self, idx, attn_mask=None, pool: str = "mean", raw: bool = True):
        # DEPRECATED, NOT IN USE
        """Return a global latent z for the sequence.
        pool: 'mean' over non-PAD tokens, or 'last' (use last non-PAD / EOS position).
        raw: if True, return pooled pre-projection representation
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
            pooled = x[torch.arange(B, device=idx.device), last_idx]
        
        # Update: always return the raw latent vector
        if raw:
            return pooled  # (B, n_embd) semantic latent vector

        z = self.z_proj(pooled)
        return z  # (B, latent_dim)


    @torch.no_grad()
    def encode(
        self,
        idx,
        attn_mask=None,
        pool: str = "mean",
        raw: bool = True,
        mode: str = "global",
        chunk_size: int = 32,
    ):
        """
        Returns latent representations of the sequence.

        mode:
            "global"  -> single vector [B, D] (previous behavior)
            "chunked" -> multiple vectors per example [B, num_chunks, D]

        pool:
            "mean" -> mean over selected tokens
            "last" -> last non-pad token

        raw:
            if True, return pooled contextual state(s) directly (B, D) or (B, C, D)
            if False, would apply projection head (deprecated in our design)

        chunk_size:
            only used when mode="chunked". We take non-pad tokens, break them
            into contiguous spans of length `chunk_size`, and pool each span.

        Notes:
            - This does NOT change training. It's just a readout function.
            - We assume BOS/EOS are just tokens like any others; they get pooled, too.
        """
        self.eval()
        B, L = idx.shape

        # Embedding + transformer forward
        x = self.tok_emb(idx)            # (B, L, D)
        x = x + self.pos_emb(x)          # (B, L, D)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.ln_f(x)                 # (B, L, D)

        # Build attention mask if not given
        if attn_mask is None:
            attn_mask = (idx != self.pad_id).long()  # (B, L)

        # ============ MODE: GLOBAL (legacy single-z) ============
        if mode == "global":
            if pool == "mean":
                denom = attn_mask.sum(dim=1, keepdim=True).clamp(min=1)          # (B,1)
                pooled = (x * attn_mask.unsqueeze(-1)).sum(dim=1) / denom        # (B,D)
            else:  # 'last'
                last_idx = (attn_mask * torch.arange(L, device=idx.device)).argmax(dim=1)
                pooled = x[torch.arange(B, device=idx.device), last_idx]         # (B,D)

            if raw:
                return pooled  # (B, D)

            # fallback path if we ever reintroduce z_proj
            return self.z_proj(pooled)

        # ============ MODE: CHUNKED (multi-z) ============
        elif mode == "chunked":
            # We'll produce multiple pooled vectors per sequence.
            # Steps:
            # 1. For each example in batch, gather only the valid (non-pad) tokens.
            # 2. Break them into contiguous spans of up to chunk_size tokens.
            # 3. Mean-pool x over each span.
            # 4. Pad the list of spans across batch so we can return a tensor.

            device = idx.device
            B, L, D = x.shape
            chunked_reprs = []
            max_chunks = 0

            for b in range(B):
                valid_mask = attn_mask[b].bool()        # (L,)
                x_valid = x[b][valid_mask]              # (T_valid, D)

                if x_valid.size(0) == 0:
                    # edge case: sequence is all PAD?
                    pooled_chunks = x_valid.new_zeros((1, D))  # [1, D]
                else:
                    # break into spans of chunk_size
                    spans = []
                    for start in range(0, x_valid.size(0), chunk_size):
                        end = start + chunk_size
                        span = x_valid[start:end]       # (<=chunk_size, D)
                        span_mean = span.mean(dim=0)    # (D,)
                        spans.append(span_mean)
                    pooled_chunks = torch.stack(spans, dim=0)  # (num_chunks, D)

                chunked_reprs.append(pooled_chunks)
                if pooled_chunks.size(0) > max_chunks:
                    max_chunks = pooled_chunks.size(0)

            # Now we need to batch these into a single tensor.
            # We pad with zeros for sequences that have fewer chunks.
            out = x.new_zeros((B, max_chunks, D))  # (B, C, D)
            chunk_mask = torch.zeros(B, max_chunks, dtype=torch.bool, device=device)

            for b in range(B):
                Cb = chunked_reprs[b].size(0)
                out[b, :Cb] = chunked_reprs[b]
                chunk_mask[b, :Cb] = True

            # out: [B, C, D]
            # chunk_mask: [B, C] tells you which chunks are "real"

            if raw:
                return out, chunk_mask  # (B, C, D), (B, C)

            # if you ever reintroduce z_proj to change dim:
            # projected_out = self.z_proj(out)  # would become (B, C, latent_dim)
            # return projected_out, chunk_mask

        else:
            raise ValueError(f"Unknown encode mode {mode!r}")