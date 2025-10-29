# =========================
# train_ar.py — Stage‑1 AR training
# =========================
# Minimal, single-file trainer. Uses SentencePiece for tokenization.
# Saves: ckpts/ar.pt (state_dict) and a small training log.

'''
# assuming you already ran tokenizer.py
python train_ar.py --corpus data/train.txt --spm tokenizer/spm.model \
  --max_len 512 --n_layer 8 --n_head 8 --n_embd 512 --latent_dim 512 \
  --batch_size 32 --lr 3e-4 --max_steps 20000
'''

import os
import time
import json
import argparse
import random
from pathlib import Path
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ARLanguageModel, ARConfig

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
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, i):
        text = self.lines[i]
        ids = [self.bos_id] + self.sp.encode(text, out_type=int) + [self.eos_id]
        # truncate / pad
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
    parser.add_argument('--corpus', type=str, required=True, help='Path to training text (one doc per line).')
    parser.add_argument('--spm', type=str, default='tokenizer/spm.model', help='SentencePiece model path.')
    parser.add_argument('--out_dir', type=str, default='ckpts/ar')
    parser.add_argument('--vocab_size', type=int, default=None, help='If None, inferred from spm.')
    parser.add_argument('--max_len', type=int, default=256) #512
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=128) #512
    parser.add_argument('--latent_dim', type=int, default=256) #512
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-3) #3e-4
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=10) #2000
    parser.add_argument('--max_steps', type=int, default=100) #20000
    parser.add_argument('--eval_every', type=int, default=20) #1000 
    parser.add_argument('--seed', type=int, default=5337)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else ('cpu' if torch.backends.mps.is_available() else 'cpu'))
    
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (eval intervals)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--plot', action='store_true', help='Show live training/validation loss plot', default=True)
    parser.add_argument('--print_interval', type=int, default=20, help='Print verbose training information every N steps')

    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Tokenizer and vocab size
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    vocab_size = args.vocab_size or sp.get_piece_size()

    # Data
    # ds = TextDataset(args.corpus, args.spm, args.max_len)
    # dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True, collate_fn=batchify)

    ds = TextDataset(args.corpus, args.spm, args.max_len)
    train_ds, val_ds = split_dataset(ds, val_ratio=args.val_ratio, seed=args.seed)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True, collate_fn=batchify)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2, pin_memory=True, collate_fn=batchify)
    
    
    # Model
    cfg = ARConfig(vocab_size=vocab_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, max_seq_len=args.max_len, latent_dim=args.latent_dim)
    model = ARLanguageModel(cfg).to(args.device)

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps, eta_min=1e-5)

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        return sched.get_last_lr()[0]

    stopper = EarlyStopper(patience=args.patience)
    plotter = plot_losses() if args.plot else None

    best_val_loss = float('inf')

    # Training loop
    step = 0
    model.train()
    running_loss = 0.0
    t0 = time.time()
    while step < args.max_steps:
        for ids, attn in train_dl:
            ids = ids.to(args.device)
            attn = attn.to(args.device)
            logits = model(ids, attn)
            # teacher forcing next-token loss
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size),
                ids[:, 1:].contiguous().view(-1),
                ignore_index=0,  # ignore PAD
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # manual warmup
            for g in opt.param_groups:
                g['lr'] = get_lr(step)
            opt.step()
            if step >= args.warmup_steps:
                sched.step()

            running_loss += loss.item()
            if (step + 1) % args.print_interval == 0:
                tok_per_step = ids.numel()
                dt = time.time() - t0
                print(f"step {step+1:06d} | loss {running_loss/100:.4f} | lr {opt.param_groups[0]['lr']:.2e} | {tok_per_step/1e6:.3f}M tok/it | {(step+1)/dt:.1f} it/s | time {dt:.2f} s")
                running_loss = 0.0
            if (step + 1) % args.eval_every == 0:
                # --- Validation pass ---
                model.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for v_ids, v_attn in val_dl:
                        v_ids, v_attn = v_ids.to(args.device), v_attn.to(args.device)
                        v_logits = model(v_ids, v_attn)
                        v_loss = F.cross_entropy(
                            v_logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size),
                            v_ids[:, 1:].contiguous().view(-1),
                            ignore_index=0,
                        )
                        val_loss_total += v_loss.item()
                val_loss = val_loss_total / len(val_dl)
                print(f"\nValidation loss: {val_loss:.4f}\n")

                # --- Early stopping & checkpointing ---
                save_checkpoint(model, opt, step, val_loss, args.out_dir, tag='last')
                improved = stopper.step(val_loss)
                if improved:
                    save_checkpoint(model, opt, step, val_loss, args.out_dir, tag=f'step{step}', best=True)
                    best_val_loss = val_loss
                if stopper.should_stop:
                    print(f"Early stopping triggered (no improvement for {args.patience} evals).\nBest val_loss={best_val_loss:.4f}")
                    break

                # --- Live plot update ---
                if plotter:
                    plotter.update(step, running_loss / max(1, step % args.eval_every), val_loss)
                model.train()
            step += 1
            if step >= args.max_steps:
                break


    if plotter:
        plotter.close()

    # Save
    save_path = os.path.join(args.out_dir, 'ar/ar.pt') # Note that the frozen model weights are saved to ar.pt
    torch.save({'config': cfg.__dict__, 'state_dict': model.state_dict(), 'spm': args.spm}, save_path)
    print(f"✅ Saved AR encoder to {save_path}")

    end_time = time.time()
    total_time = end_time - t0

    print(f"Total training time: {total_time:.2f} s")
    print(f"Average latency per iteration: {total_time/args.max_steps:.2f} s")
    print(f"Total tokens: {ids.numel()}")
    print(f"Throughput: {ids.numel()/total_time:.2f} tokens/s")

if __name__ == '__main__':
    main()
