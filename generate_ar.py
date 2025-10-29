"""
generate_ar.py
--------------
Simple text generation script for the Stage-1 ARLanguageModel.

Usage:
    python generate_ar.py --spm tokenizer/spm.model --ckpt ckpts/ar.pt \
        --prompt "The future of AI is" --max_new_tokens 64 --top_k 50 --temperature 0.9
"""

import argparse
import torch
import torch.nn.functional as F
import sentencepiece as spm
from model import ARLanguageModel, ARConfig


@torch.no_grad()
def sample(model, sp, prompt, max_new_tokens=64, top_k=50, temperature=1.0, device="cpu"):
    model.eval()
    ids = torch.tensor([sp.encode(prompt)], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)

    for _ in range(max_new_tokens):
        logits = model(ids, attn)[:, -1, :]  # logits for the last token
        logits = logits / temperature

        if top_k is not None and top_k > 0:
            topv, topi = torch.topk(logits, top_k)
            probs = torch.zeros_like(logits).scatter_(1, topi, F.softmax(topv, dim=-1))
        else:
            probs = F.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        attn = torch.ones_like(ids)

        if sp.id_to_piece(next_id.item()) in ["<eos>", "</s>"]:
            print("EOS reached")
            break

    text = sp.decode(ids[0].tolist())
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ckpts/ar/ar.pt")
    parser.add_argument("--spm", type=str, default="tokenizer/spm.model")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load tokenizer and model
    sp = spm.SentencePieceProcessor()
    sp.load(args.spm)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = ARConfig(**ckpt["config"])
    model = ARLanguageModel(cfg).to(args.device)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # Generate
    print(f"\n>>> Prompt: {args.prompt}\n")
    out = sample(model, sp, args.prompt, args.max_new_tokens, args.top_k, args.temperature, args.device)
    print("=== Generated Text ===")
    print(out)


if __name__ == "__main__":
    main()