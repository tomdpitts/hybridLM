"""
count_params.py
---------------
Inspect total, category-wise parameter counts and memory footprint
for ARLanguageModel (Stage-1) and DiffusionDecoder (Stage-2).

Usage:
    python count_params.py --ckpt ckpts/ar/ar.pt
    python count_params.py --ckpt ckpts/diff/diff.pt --details
"""

import argparse
import torch
from pathlib import Path
from model import ARLanguageModel, ARConfig
from model_diffusion import DiffusionDecoder, DiffConfig

BYTES_FP32 = 4
BYTES_FP16 = 2


def count_params(model):
    return {name: p.numel() for name, p in model.named_parameters()}


def summarize_categories(param_dict):
    """Group parameters into logical categories based on name patterns."""
    categories = {
        "embeddings": 0,
        "transformer_blocks": 0,
        "output_heads": 0,
        "conditioning": 0,
        "time_positional": 0,
        "other": 0,
    }

    for name, n in param_dict.items():
        lname = name.lower()
        if any(k in lname for k in ["tok_emb", "embedding"]):
            categories["embeddings"] += n
        elif "blocks" in lname or "attn" in lname or "mlp" in lname:
            categories["transformer_blocks"] += n
        elif "lm_head" in lname or "proj" in lname and "z_proj" not in lname:
            categories["output_heads"] += n
        elif "z_proj" in lname or "cond_" in lname:
            categories["conditioning"] += n
        elif "time_emb" in lname or "pos_emb" in lname:
            categories["time_positional"] += n
        else:
            categories["other"] += n
    return categories


def print_breakdown(categories):
    total = sum(categories.values())
    print("\n--- Parameter category breakdown ---")
    for k, v in categories.items():
        print(f"{k:<20} {v/1e6:8.3f} M  ({100*v/total:5.1f}%)")
    print("------------------------------------")
    print(f"Total parameters: {total/1e6:.3f} M\n")
    return total


def print_memory(total_params):
    mb_fp32 = total_params * BYTES_FP32 / (1024 ** 2)
    mb_fp16 = total_params * BYTES_FP16 / (1024 ** 2)
    print("--- Memory footprint ---")
    print(f"FP32: {mb_fp32:,.1f} MB  (~{mb_fp32/1024:.2f} GB)")
    print(f"FP16: {mb_fp16:,.1f} MB  (~{mb_fp16/1024:.2f} GB)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--details", action="store_true", help="Show per-parameter details")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Detect model type
    if "latent_dim" in ckpt.get("config", {}):
        cfg = ARConfig(**ckpt["config"])
        model = ARLanguageModel(cfg)
        model_name = "ARLanguageModel"
    elif "cond_dim" in ckpt.get("config", {}):
        cfg = DiffConfig(**ckpt["config"])
        model = DiffusionDecoder(cfg)
        model_name = "DiffusionDecoder"
    else:
        raise ValueError("Checkpoint does not match known model types")

    model.load_state_dict(ckpt["state_dict"], strict=False)
    param_dict = count_params(model)
    total_params = sum(param_dict.values())

    print(f"\nModel: {model_name}")
    print(f"Total parameters: {total_params/1e6:.3f} M\n")

    # Category breakdown
    cats = summarize_categories(param_dict)
    print_breakdown(cats)

    # Memory footprint
    print_memory(total_params)

    if args.details:
        print("--- Per-parameter listing ---")
        for name, n in param_dict.items():
            print(f"{name:<60} {n/1e6:8.3f} M")


if __name__ == "__main__":
    main()