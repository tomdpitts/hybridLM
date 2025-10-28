"""
inspect_latent.py
-----------------
Visualize raw vs. projected latent embeddings from the ARLanguageModel.

Usage:
    python inspect_latent.py --ckpt ckpts/ar.pt --spm tokenizer/spm.model \
        --prompts "the cat sat on the mat" "a dog on a rug" "quantum computing breakthroughs"
"""

import argparse
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.nn.functional import cosine_similarity
from model import ARLanguageModel, ARConfig
import sentencepiece as spm


@torch.no_grad()
def get_latents(model, sp, prompts, device="cpu"):
    raws, projs = [], []
    for p in prompts:
        ids = torch.tensor([sp.encode(p)], dtype=torch.long, device=device)
        attn = torch.ones_like(ids)
        raw = model.encode(ids, attn, pool="mean", raw=True)
        proj = model.encode(ids, attn, pool="mean", raw=False)
        raws.append(raw.cpu())
        projs.append(proj.cpu())
    return torch.cat(raws), torch.cat(projs)


def plot_pca(vectors, labels, title):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, lbl in enumerate(labels):
        plt.text(coords[i, 0], coords[i, 1], lbl, fontsize=9)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--spm", type=str, required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # load model & tokenizer
    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = ARConfig(**ckpt["config"])
    model = ARLanguageModel(cfg).to(args.device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm)

    raws, projs = get_latents(model, sp, args.prompts, device=args.device)

    print("\n--- Cosine Similarities (raw) ---")
    for i in range(len(args.prompts)):
        for j in range(i + 1, len(args.prompts)):
            sim = cosine_similarity(raws[i], raws[j], dim=-1)
            print(f"({args.prompts[i]!r}, {args.prompts[j]!r}) = {sim.item():.4f}")

    print("\n--- Cosine Similarities (projected) ---")
    for i in range(len(args.prompts)):
        for j in range(i + 1, len(args.prompts)):
            sim = cosine_similarity(projs[i], projs[j], dim=-1)
            print(f"({args.prompts[i]!r}, {args.prompts[j]!r}) = {sim.item():.4f}")

    plot_pca(raws.numpy(), args.prompts, "Raw latent space (PCA)")
    plot_pca(projs.numpy(), args.prompts, "Projected latent space (PCA)")


if __name__ == "__main__":
    main()