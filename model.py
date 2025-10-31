# =========================
# model.py — Stage-1 AR (Encoder LM)
# =========================
# Shared tokenizer: SentencePiece (spm.model)
# Special tokens assumed:
#   [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3, and optional [MASK]
#
# Exposes:
#   ARLanguageModel: standard causal LM that also provides `encode()` → latent z
# =========================

from dataclasses import dataclass
import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = os.environ.get("HYBRIDLM_DEBUG", "0") == "1"

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
    rope: bool = False
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
        L = x.size(1)
        return self.pe[:L].unsqueeze(0)

# -----------------------------
# Utilities
# -----------------------------
def _normalize_token_mask(idx: torch.Tensor, attn_mask) -> torch.Tensor:
    """
    Returns a robust 2D token mask (B, L) with 1 for real tokens and 0 for PAD.
    Accepts None, 2D (B,L), or 4D (B,1,1,L). If malformed, rebuilds from idx.
    """
    B, L = idx.shape
    if attn_mask is None:
        m2d = (idx != 0).long()  # assume PAD=0
        if DEBUG: print(f"[mask] built from idx -> {tuple(m2d.shape)}")
        return m2d

    if not torch.is_tensor(attn_mask):
        raise TypeError("attn_mask must be a tensor or None")

    if attn_mask.dim() == 2:
        if attn_mask.size(0) != B or attn_mask.size(1) != L:
            if DEBUG: print(f"[mask] WARNING 2D mismatch {tuple(attn_mask.shape)} vs (B,L)=({B},{L}); rebuilding")
            return (idx != 0).long()
        if DEBUG: print(f"[mask] using provided 2D -> {tuple(attn_mask.shape)}")
        return attn_mask.long()

    if attn_mask.dim() == 4:
        # Expect (B,1,1,L). If not, try to squeeze to (B,L), else rebuild.
        if attn_mask.size(-1) == L and attn_mask.size(0) == B:
            m2d = attn_mask.squeeze(1).squeeze(1).contiguous().long()
            if m2d.dim() != 2 or m2d.size(1) != L:
                if DEBUG: print(f"[mask] WARNING 4D squeeze bad {tuple(attn_mask.shape)} -> {tuple(m2d.shape)}; rebuilding")
                return (idx != 0).long()
            if DEBUG: print(f"[mask] squeezed 4D -> 2D {tuple(m2d.shape)}")
            return m2d
        else:
            if DEBUG: print(f"[mask] WARNING 4D mismatch {tuple(attn_mask.shape)} vs (B,*,*,L)=({B},*,*,{L}); rebuilding")
            return (idx != 0).long()

    # Any other rank: rebuild safely
    if DEBUG: print(f"[mask] WARNING rank {attn_mask.dim()} unsupported; rebuilding")
    return (idx != 0).long()

def _expand_to_att_mask(m2d: torch.Tensor, heads: int, L: int) -> torch.Tensor:
    """
    From 2D (B,L) -> 4D (B,1,1,L) boolean mask that broadcasts over (B,H,L,L).
    """
    if m2d.dim() != 2 or m2d.size(1) != L:
        raise RuntimeError(f"Token mask must be (B,L); got {tuple(m2d.shape)} with L={L}")
    return (m2d != 0).to(torch.bool)[:, None, None, :]  # (B,1,1,L)

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
        q = self.query(x).view(B, L, self.n_head, C // self.n_head).transpose(1, 2)  # (B,H,L,Dh)
        k = self.key(x).view(B, L, self.n_head, C // self.n_head).transpose(1, 2)    # (B,H,L,Dh)
        v = self.value(x).view(B, L, self.n_head, C // self.n_head).transpose(1, 2)  # (B,H,L,Dh)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (B,H,L,L)
        causal = self.mask[:, :, :L, :L]
        att = att.masked_fill(causal == 0, float("-inf"))

        if attn_mask is not None:
            # Always normalize to a safe 2D token mask (B,L), then expand.
            m2d = _normalize_token_mask(idx=torch.empty(B, L, dtype=torch.long, device=x.device),  # dummy idx for shape
                                        attn_mask=attn_mask)
            m = _expand_to_att_mask(m2d, self.n_head, L)  # (B,1,1,L)
            if DEBUG: print(f"[attn] att {tuple(att.shape)} ; mask {tuple(m.shape)}")
            att = att.masked_fill(~m, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B,H,L,Dh)
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.resid_drop(self.proj(y))
        return y

# -----------------------------
# MLP + Transformer block
# -----------------------------
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
# AR Language Model
# -----------------------------
class ARLanguageModel(nn.Module):
    def __init__(self, cfg: ARConfig, pad_id: int = 0, eos_id: int = 3, bos_id: int = 2):
        super().__init__()
        self.cfg = cfg
        self.pad_id, self.eos_id, self.bos_id = pad_id, eos_id, bos_id
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = SinusoidalPositionalEmbedding(cfg.n_embd, cfg.max_seq_len)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg.n_embd, cfg.n_head, cfg.dropout, cfg.layer_norm_eps) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, attn_mask=None):
        B, L = idx.shape
        assert L <= self.cfg.max_seq_len
        # Normalize to a robust (B,L) token mask; tolerate malformed callers
        attn_mask_2d = _normalize_token_mask(idx, attn_mask)

        if DEBUG:
            print(f"[fwd] idx {tuple(idx.shape)} ; attn_mask_2d {tuple(attn_mask_2d.shape)}")

        x = self.tok_emb(idx)
        x = x + self.pos_emb(x)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, attn_mask_2d)  # blocks/attention will expand internally
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

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
        self.eval()
        B, L = idx.shape
        attn_mask_2d = _normalize_token_mask(idx, attn_mask)

        if DEBUG:
            print(f"[enc] idx {tuple(idx.shape)} ; attn_mask_2d {tuple(attn_mask_2d.shape)}")

        # Forward through transformer
        x = self.tok_emb(idx)
        x = x + self.pos_emb(x)
        for blk in self.blocks:
            x = blk(x, attn_mask_2d)
        x = self.ln_f(x)  # (B,L,D)

        # Pool
        if mode == "global":
            if pool == "mean":
                denom = attn_mask_2d.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = (x * attn_mask_2d.unsqueeze(-1)).sum(dim=1) / denom
            else:  # 'last'
                last_idx = (attn_mask_2d * torch.arange(L, device=idx.device)).argmax(dim=1)
                pooled = x[torch.arange(B, device=idx.device), last_idx]
            return pooled if raw else pooled  # (B,D)

        elif mode == "chunked":
            device = idx.device
            B, L, D = x.shape
            chunked_reprs, max_chunks = [], 0
            for b in range(B):
                valid_mask = attn_mask_2d[b].bool()
                x_valid = x[b][valid_mask]
                if x_valid.size(0) == 0:
                    pooled_chunks = x_valid.new_zeros((1, D))
                else:
                    spans = []
                    for start in range(0, x_valid.size(0), chunk_size):
                        end = start + chunk_size
                        spans.append(x_valid[start:end].mean(dim=0))
                    pooled_chunks = torch.stack(spans, dim=0)
                chunked_reprs.append(pooled_chunks)
                max_chunks = max(max_chunks, pooled_chunks.size(0))

            out = x.new_zeros((B, max_chunks, D))
            chunk_mask = torch.zeros(B, max_chunks, dtype=torch.bool, device=device)
            for b in range(B):
                Cb = chunked_reprs[b].size(0)
                out[b, :Cb] = chunked_reprs[b]
                chunk_mask[b, :Cb] = True
            return (out, chunk_mask) if raw else (out, chunk_mask)

        else:
            raise ValueError(f"Unknown encode mode {mode!r}")