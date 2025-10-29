# ================================================
# generate.py — Phase 4: Two‑Stage Inference
# ================================================
# Uses shared SentencePiece tokenizer (tokenizer/spm.model), Stage‑1 AR encoder
# (ckpts/ar.pt) to compute latent z, then Stage‑2 diffusion decoder (ckpts/diff.pt)
# to iteratively unmask and generate text. Preserves PAD, fixes BOS/EOS.


#  example usage:
'''
python generate.py \
  --spm tokenizer/spm.model \
  --ar_ckpt ckpts/ar.pt \
  --diff_ckpt ckpts/diff.pt \
  --prompt "Explain why transformers scale so well with data." \
  --steps 64 --topk 50 --max_len 512
'''

import argparse
import torch
import sentencepiece as spm
from pathlib import Path

from model import ARLanguageModel, ARConfig
from model_diffusion import DiffusionDecoder, DiffConfig, sample_iterative

@torch.no_grad()
def encode_prompt_to_z(ar: ARLanguageModel, sp, prompt: str, max_len: int, device: str):
    pad_id, bos_id, eos_id = 0, 2, 3
    ids = [bos_id] + sp.encode(prompt, out_type=int) + [eos_id]
    ids = ids[:max_len]
    attn = [1] * len(ids)
    if len(ids) < max_len:
        ids = ids + [pad_id] * (max_len - len(ids))
        attn = attn + [0] * (max_len - len(ids))
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    m = torch.tensor(attn, dtype=torch.long, device=device).unsqueeze(0)
    z = ar.encode(x, m, pool='mean')  # (1, cond_dim)
    return z

@torch.no_grad()
def build_initial_lengths(sp, prompt: str, max_len: int, include_bos_eos: bool = True):
    # target length heuristic: prompt length + margin
    bos_id, eos_id = 2, 3
    toks = [bos_id] + sp.encode(prompt, out_type=int) + [eos_id]
    # length to generate (you can also pass as CLI)
    return min(len(toks) + 64, max_len)

@torch.no_grad()
def decode_tokens(sp, tokens, bos_id=2, eos_id=3, pad_id=0):
    toks = [int(t) for t in tokens if int(t) != pad_id]
    if toks and toks[0] == bos_id: toks = toks[1:]
    if toks and toks[-1] == eos_id: toks = toks[:-1]
    return sp.decode(toks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spm', type=str, default='tokenizer/spm.model')
    parser.add_argument('--ar_ckpt', type=str, default='ckpts/ar/ar.pt')
    parser.add_argument('--diff_ckpt', type=str, default='ckpts/diff/diff.pt')
    parser.add_argument('--prompt', type=str, default='Once upon a time,')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--steps', type=int, default=64, help='Sampling steps (<= training T)')
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--length', type=int, default=0, help='Optional explicit target length (incl. BOS/EOS)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Tokenizer and special IDs
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    pad_id, bos_id, eos_id = 0, 2, 3
    mask_id = sp.piece_to_id('[MASK]')
    assert mask_id >= 0, 'Your tokenizer must include [MASK] as user_defined_symbols.'

    # Load AR encoder
    ar_ckpt = torch.load(args.ar_ckpt, map_location='cpu')
    ar_cfg = ARConfig(**ar_ckpt['config'])
    ar = ARLanguageModel(ar_cfg).to(args.device)
    ar.load_state_dict(ar_ckpt['state_dict'])
    ar.eval()

    # Load diffusion decoder
    diff_ckpt = torch.load(args.diff_ckpt, map_location='cpu')
    diff_cfg = DiffConfig(**diff_ckpt['config'])
    # allow overriding max_len/T for faster sampling
    diff_cfg.max_len = args.max_len
    diff_cfg.T = max(args.steps, 1)
    diff = DiffusionDecoder(diff_cfg).to(args.device)
    diff.load_state_dict(diff_ckpt['state_dict'])
    diff.eval()

    # Encode prompt to z
    z = encode_prompt_to_z(ar, sp, args.prompt, args.max_len, args.device)  # (1, cond_dim)

    # Determine target length
    target_len = args.length if args.length > 0 else build_initial_lengths(sp, args.prompt, args.max_len)
    lengths = torch.tensor([target_len], device=args.device)

    # Sample
    toks = sample_iterative(
        model=diff,
        lengths=lengths,
        z=z,
        pad_id=pad_id,
        mask_id=mask_id,
        T=diff_cfg.T,
        bos_id=bos_id,
        eos_id=eos_id,
        topk=args.topk,
        device=args.device,
    )[0]

    text = decode_tokens(sp, toks, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)
    print("\n=== PROMPT ===\n" + args.prompt)
    print("\n=== GENERATION ===\n" + text + "\n")

if __name__ == '__main__':
    main()
