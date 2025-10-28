# ================================================
# model_diffusion.py — Stage‑2 Discrete Diffusion Decoder
# ================================================
# Shared tokenizer: SentencePiece (tokenizer/spm.model)
# Special tokens assumed:
#   [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3, [MASK] present in SPM `user_defined_symbols`
#
# Exposes:
#   DiffConfig, DiffusionDecoder
#   Utilities: linear_mask_rate, cosine_mask_rate, corrupt_with_masks,
#              masked_xent_loss, sample_iterative
#


from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ARConfig

# -----------------------------
# Config
# -----------------------------
@dataclass
class DiffConfig:
    vocab_size: int
    pad_id: int = 0
    mask_id: int = 4  # NOTE: update at runtime after loading SPM if needed
    bos_id: int = 2
    eos_id: int = 3
    max_len: int = ARConfig.max_seq_len #512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = ARConfig.n_embd #512
    dropout: float = 0.1
    T: int = 200  # diffusion steps for training
    cond_dim: int = ARConfig.n_embd #512 # z dimension from AR encode() - this should be kept consistent with the AR model!

# -----------------------------
# Building blocks
# -----------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, T: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(T + 1, dim)
        nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, t: torch.Tensor):
        return self.emb(t)

class CausalMasklessSelfAttention(nn.Module):
    """Plain self-attention (no causal mask) for bidirectional denoising."""
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
    def forward(self, x, key_padding_mask=None):
        B, L, C = x.size()
        q = self.query(x).view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(x).view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if key_padding_mask is not None:
            m = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
            att = att.masked_fill(m == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, eps=1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, eps=eps)
        self.attn = CausalMasklessSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd, eps=eps)
        self.mlp = MLP(n_embd, dropout)
    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x

# -----------------------------
# Diffusion Decoder
# -----------------------------
class DiffusionDecoder(nn.Module):
    def __init__(self, cfg: DiffConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.n_embd)
        self.time_emb = TimeEmbedding(cfg.T, cfg.n_embd)
        # simple additive conditioning on z, with small adaptor
        self.cond_ln = nn.LayerNorm(cfg.cond_dim)
        # self.cond_proj = nn.Linear(cfg.cond_dim, cfg.n_embd) # deprecated 
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, cfg.dropout) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self._reset()
    def _reset(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.01)
        nn.init.normal_(self.lm_head.weight, std=0.02)
    def forward(self, x_noisy: torch.Tensor, attn_mask: torch.Tensor, t: torch.Tensor, z: torch.Tensor):
        """
        x_noisy: (B, L) tokens (with [MASK] on corrupted positions)
        attn_mask: (B, L) 1 for real tokens (including MASK), 0 for PAD
        t: (B,) diffusion step (1..T)
        z: (B, cond_dim) conditioning from AR.encoder
        returns: logits (B, L, V)
        """
        B, L = x_noisy.shape
        pos = torch.arange(L, device=x_noisy.device).unsqueeze(0).expand(B, L)
        # embeddings
        h = self.tok_emb(x_noisy) + self.pos_emb(pos) + self.time_emb(t).unsqueeze(1)
        # cz = self.cond_proj(self.cond_ln(z)).unsqueeze(1)  # (B,1,D) # deprecated
        h = self.drop(h)  # broadcast add
        key_padding_mask = attn_mask  # 1=keep, 0=pad -> pass directly
        for blk in self.blocks:
            h = blk(h, key_padding_mask)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits

# -----------------------------
# Diffusion utilities
# -----------------------------
@torch.no_grad()
def linear_mask_rate(t: torch.Tensor, T: int, min_rate: float = 0.05) -> torch.Tensor:
    return torch.clamp(min_rate + (t.float() / T) * (1.0 - min_rate), 0.0, 1.0)

@torch.no_grad()
def cosine_mask_rate(t: torch.Tensor, T: int, min_rate: float = 0.05) -> torch.Tensor:
    # cosine from 1.0 → min_rate as t decreases
    s = 0.5 * (1 + torch.cos(math.pi * t.float() / T))
    return torch.clamp(min_rate + (1 - s) * (1.0 - min_rate), 0.0, 1.0)

@torch.no_grad()
def corrupt_with_masks(x: torch.Tensor, pad_id: int, mask_id: int, t: torch.Tensor, T: int, schedule: str = 'linear') -> Tuple[torch.Tensor, torch.Tensor]:
    B, L = x.shape
    if schedule == 'linear':
        rates = linear_mask_rate(t, T).unsqueeze(1)
    else:
        rates = cosine_mask_rate(t, T).unsqueeze(1)
    not_pad = (x != pad_id)
    rand = torch.rand((B, L), device=x.device)
    to_mask = (rand < rates) & not_pad
    x_noisy = x.clone()
    x_noisy[to_mask] = mask_id
    predict_mask = to_mask
    return x_noisy, predict_mask

def masked_xent_loss(logits: torch.Tensor, target: torch.Tensor, predict_mask: torch.Tensor, pad_id: int) -> torch.Tensor:
    B, L, V = logits.shape
    labels = target.clone()
    labels[~predict_mask] = -100
    labels[target == pad_id] = -100
    return F.cross_entropy(logits.view(B * L, V), labels.view(B * L), ignore_index=-100)

# -----------------------------
# Sampling
# -----------------------------
@torch.no_grad()
def sample_iterative(model: DiffusionDecoder, lengths: torch.Tensor, z: torch.Tensor, pad_id: int, mask_id: int, T: int, bos_id: int, eos_id: int, topk: Optional[int] = 50, device: str = 'cuda') -> torch.Tensor:
    """Start from [BOS] + MASK ... + [EOS] + PAD and iteratively unmask.
    lengths: desired non-pad lengths INCLUDING BOS/EOS if you use both
    """
    model.eval()
    B = lengths.size(0)
    L = model.cfg.max_len
    x = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, L), dtype=torch.long, device=device)
    for b in range(B):
        ell = int(lengths[b].item())
        ell = max(3, min(ell, L))  # at least BOS, one token, EOS
        seq = torch.full((ell,), mask_id, dtype=torch.long, device=device)
        seq[0] = bos_id
        seq[-1] = eos_id
        x[b, :ell] = seq
        attn[b, :ell] = 1
    for step in range(T, 0, -1):
        t = torch.full((B,), step, dtype=torch.long, device=device)
        logits = model(x, attn, t, z)
        probs = F.softmax(logits, dim=-1)
        if topk is not None and topk < model.cfg.vocab_size:
            topk_vals, topk_idx = torch.topk(probs, k=topk, dim=-1)
            probs = torch.zeros_like(probs).scatter_(-1, topk_idx, topk_vals)
            probs = probs / (probs.sum(-1, keepdim=True) + 1e-8)
        # fill fraction of remaining MASKs
        frac = 1.0 - (step - 1) / T
        mask_pos = (x == mask_id)
        bern = (torch.rand_like(x.float()) < frac) & mask_pos
        if bern.any():
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(B, L)
            x[bern] = sampled[bern]
        # keep BOS/EOS fixed
        x[(x != pad_id) & (attn == 1) & (x == bos_id)] = bos_id
        x[(x != pad_id) & (attn == 1) & (x == eos_id)] = eos_id
    return x