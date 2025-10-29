
# ================================================
# train_diffusion.py — Stage‑2 training script
# ================================================
# Loads AR encoder (frozen), computes z per batch, trains diffusion decoder
# with masked denoising objective on NON-PAD masked positions.


'''
python train_diffusion.py --corpus data/train.txt --spm tokenizer/spm.model \
  --ar_ckpt ckpts/ar.pt --max_len 512 --batch_size 32 --lr 1e-4 \
  --T 200 --schedule linear --eval_every 1000
'''

import math
import os
import time
import argparse
from pathlib import Path
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model_diffusion import DiffConfig, DiffusionDecoder, corrupt_with_masks, masked_xent_loss, sample_iterative
from model import ARLanguageModel, ARConfig  # from Phase 2

from utils_train import (
    split_dataset,
    EarlyStopper,
    save_checkpoint,
    plot_losses
)
class TextDataset(Dataset):
    def __init__(self, path_txt: str, sp_model: str, max_seq_len: int):
        self.lines = []
        with open(path_txt, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.lines.append(line)
        self.sp = spm.SentencePieceProcessor(model_file=sp_model)
        self.max_len = max_seq_len
        self.pad_id, self.bos_id, self.eos_id = 0, 2, 3
        # try to get [MASK] id; fallback to adding a piece
        try:
            self.mask_id = self.sp.piece_to_id('[MASK]')
            if self.mask_id < 0:
                raise ValueError
        except Exception:
            # if tokenizer lacks [MASK], reserve an unused id (last) — but you should retrain tokenizer with [MASK]
            self.mask_id = self.sp.get_piece_size() - 1
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, i):
        text = self.lines[i]
        ids = [self.bos_id] + self.sp.encode(text, out_type=int) + [self.eos_id]
        ids = ids[: self.max_len]
        attn = [1] * len(ids)
        if len(ids) < self.max_len:
            pad = [self.pad_id] * (self.max_len - len(ids))
            padm = [0] * (self.max_len - len(ids))
            ids = ids + pad
            attn = attn + padm
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)

def batchify(batch):
    ids = torch.stack([b[0] for b in batch], dim=0)
    attn = torch.stack([b[1] for b in batch], dim=0)
    return ids, attn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True)
    parser.add_argument('--spm', type=str, default='tokenizer/spm.model')
    parser.add_argument('--ar_ckpt', type=str, default='ckpts/ar/ar.pt')
    parser.add_argument('--out_dir', type=str, default='ckpts/diff')
    parser.add_argument('--max_len', type=int, default=256) #512
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3) #1e-4
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=10) #100? Maybe 1000
    parser.add_argument('--max_steps', type=int, default=101) #20000
    parser.add_argument('--eval_every', type=int, default=20) #1000
    parser.add_argument('--T', type=int, default=200)
    parser.add_argument('--schedule', type=str, default='linear', choices=['linear','cosine'])
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (eval intervals)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--plot', action='store_true', help='Show live training/validation loss plot', default=True)
    parser.add_argument('--print_interval', type=int, default=5, help='Print verbose training information every N steps')
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    vocab_size = sp.get_piece_size()
    pad_id, bos_id, eos_id = 0, 2, 3
    mask_id = sp.piece_to_id('[MASK]') if sp.piece_to_id('[MASK]') >= 0 else vocab_size - 1

    # Data
    # ds = TextDataset(args.corpus, args.spm, args.max_len)
    # dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True, collate_fn=batchify)

    ds = TextDataset(args.corpus, args.spm, args.max_len)
    train_ds, val_ds = split_dataset(ds, val_ratio=args.val_ratio, seed=args.seed) #

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True, collate_fn=batchify) # 
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2, pin_memory=True, collate_fn=batchify)
    
    
    # --- Load AR encoder (frozen) ---
    ar_ckpt = torch.load(args.ar_ckpt, map_location='cpu')
    ar_cfg = ARConfig(**ar_ckpt['config'])
    ar = ARLanguageModel(ar_cfg).to(args.device)
    ar.load_state_dict(ar_ckpt['state_dict'], strict=False)

    # 🔧 ensure sinusoidal pos_emb covers full requested length
    if hasattr(ar.pos_emb, "pe"):
        old_len, dim = ar.pos_emb.pe.shape
        if old_len < args.max_len:
            print(f"Extending sinusoidal PE from {old_len} → {args.max_len}")
            device = ar.pos_emb.pe.device
            position = torch.arange(0, args.max_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
            pe = torch.zeros(args.max_len, dim, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            ar.pos_emb.pe = pe  # overwrite buffer with longer version

    # freeze encoder
    for p in ar.parameters():
        p.requires_grad_(False)
    ar.eval()

    # Build diffusion decoder
    diff_cfg = DiffConfig(
        vocab_size=vocab_size,
        pad_id=pad_id,
        mask_id=mask_id,
        bos_id=bos_id,
        eos_id=eos_id,
        max_len=args.max_len,
        n_layer=8,
        n_head=8,
        n_embd=ar_cfg.n_embd,
        dropout=0.1,
        T=args.T,
        cond_dim=ar_cfg.latent_dim,
    )
    diff = DiffusionDecoder(diff_cfg).to(args.device)

    # Optim
    opt = torch.optim.AdamW(diff.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps, eta_min=1e-5)

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        return sched.get_last_lr()[0]

    stopper = EarlyStopper(patience=args.patience)
    plotter = plot_losses() if args.plot else None

    best_val_loss = float('inf')

    # Train loop
    step = 0
    running = 0.0
    t0 = time.time()
    diff.train()
    while step < args.max_steps:
        for ids, attn in train_dl:
            ids = ids.to(args.device)
            attn = attn.to(args.device)
            B = ids.size(0)
            # get conditioning z from AR
            with torch.no_grad():
                # z = ar.encode(ids, attn, pool='mean')  # (B, cond_dim)
                z_global = ar.encode(ids, attn, pool='mean', raw=True, mode="global")         # (B, n_embd)
                z_chunks, chunk_mask = ar.encode(ids, attn, pool='mean', raw=True, mode="chunked")  # (B, num_chunks, D),
            # sample random timestep
            t = torch.randint(low=1, high=diff_cfg.T + 1, size=(B,), device=args.device)
            # corrupt
            x_noisy, predict_mask = corrupt_with_masks(ids, pad_id, mask_id, t, diff_cfg.T, schedule=args.schedule)
            # forward
            logits = diff(x_noisy, attn, t, z_global)
            loss = masked_xent_loss(logits, ids, predict_mask, pad_id)
            # step
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
            # warmup + cosine
            for g in opt.param_groups:
                g['lr'] = get_lr(step)
            opt.step()
            if step >= args.warmup_steps:
                sched.step()

            running += loss.item()
            step += 1
            if step % args.print_interval == 0:
                dt = time.time() - t0
                print(f"step {step:06d} | loss {running/100:.4f} | lr {opt.param_groups[0]['lr']:.2e} | {(step)/dt:.1f} it/s | time {dt:.2f} s")
                running = 0.0
            if step % args.eval_every == 0:
                # --- Validation + Sampling ---
                diff.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    # compute validation loss
                    for v_ids, v_attn in val_dl:
                        v_ids, v_attn = v_ids.to(args.device), v_attn.to(args.device)
                        Bv = v_ids.size(0)
                        z_val = ar.encode(v_ids, v_attn, pool='mean')
                        t_val = torch.randint(low=1, high=diff_cfg.T + 1, size=(Bv,), device=args.device)
                        x_noisy, predict_mask = corrupt_with_masks(v_ids, pad_id, mask_id, t_val, diff_cfg.T, schedule=args.schedule)
                        v_logits = diff(x_noisy, v_attn, t_val, z_val)
                        v_loss = masked_xent_loss(v_logits, v_ids, predict_mask, pad_id)
                        val_loss_total += v_loss.item()
                    val_loss = val_loss_total / len(val_dl)
                print(f"\nValidation loss: {val_loss:.4f}\n")

                # --- Qualitative samples ---
                with torch.no_grad():
                    lengths = torch.tensor([64, 96], device=args.device)
                    z_small = z_global[:2]
                    gens = sample_iterative(diff, lengths, z_small, pad_id, mask_id, diff_cfg.T, bos_id, eos_id, topk=50, device=args.device)
                    for i, seq in enumerate(gens):
                        seq_nopad = seq[seq != pad_id]
                        tokens = [int(x.item()) for x in seq_nopad]
                        if tokens and tokens[0] == bos_id:
                            tokens = tokens[1:]
                        if tokens and tokens[-1] == eos_id:
                            tokens = tokens[:-1]
                        text = sp.decode(tokens)
                        print(f"\n=== SAMPLE {i} ===\n{text[:400]}\n")

                # --- Early stopping, checkpointing, plotting ---
                save_checkpoint(diff, opt, step, val_loss, args.out_dir, tag='last')
                improved = stopper.step(val_loss)
                if improved:
                    save_checkpoint(diff, opt, step, val_loss, args.out_dir, tag=f'step{step}', best=True)
                    best_val_loss = val_loss
                if stopper.should_stop:
                    print(f"Early stopping triggered (no improvement for {args.patience} evals). Best val_loss={best_val_loss:.4f}")
                    break

                # if plotter:
                #     plotter.update(step, running / max(1, step % args.eval_every), val_loss)

                diff.train()
            if step >= args.max_steps:
                break

    # if plotter:
    #     plotter.close()

    # Save
    path = os.path.join(args.out_dir, 'diff.pt')
    torch.save({'config': diff_cfg.__dict__, 'state_dict': diff.state_dict(), 'spm': args.spm}, path)
    print(f"✅ Saved diffusion decoder to {path}")

    end_time = time.time()
    total_time = end_time - t0

    print(f"Total training time: {total_time:.2f} s")
    print(f"Average latency per iteration: {total_time/args.max_steps:.2f} s")
    print(f"Total tokens: {ids.numel()}")
    print(f"Throughput: {ids.numel()/total_time:.2f} tokens/s")

if __name__ == '__main__':
    main()

