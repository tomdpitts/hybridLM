"""
tokenizer.py
-------------
Train and export a shared SentencePiece tokenizer for both the AR and Diffusion models.

Usage:
    python tokenizer.py --input data/corpus.txt --vocab_size 16000 --model_type unigram

Outputs:
    tokenizer/spm.model
    tokenizer/spm.vocab
    tokenizer/vocab.json  # plain mapping usable by your model.py
"""
# example usage:
# python tokenizer.py --input data/input.txt --vocab_size 11000 --model_type bpe

import os
import argparse
import json
import sentencepiece as spm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input text corpus (one document per line).")
    parser.add_argument("--output_dir", type=str, default="tokenizer",
                        help="Output directory for tokenizer files.")
    parser.add_argument("--vocab_size", type=int, default=16000,
                        help="Number of subword tokens.")
    parser.add_argument("--model_type", type=str, default="unigram",
                        choices=["unigram", "bpe"],
                        help="SentencePiece model type.")
    parser.add_argument("--character_coverage", type=float, default=1.0,
                        help="Character coverage; 1.0 for English.")
    parser.add_argument("--min_sentence_length", type=int, default=0,
                        help="Optional: discard very short lines (<N chars).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_prefix = os.path.join(args.output_dir, "spm")

    print(f"🔤 Training tokenizer on {args.input}")
    print(f"   vocab_size={args.vocab_size}, type={args.model_type}")

    # Train SentencePiece model
    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["[MASK]"],
        input_sentence_size=1000000,  # sample subset if huge
        shuffle_input_sentence=True,
    )

    # Load trained model to build vocab.json
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"✅ Tokenizer saved to {args.output_dir}/")
    print(f"   Files: spm.model, spm.vocab, vocab.json")
    print(f"   Example encoding:")
    print("   >>> import sentencepiece as spm")
    print(f"   >>> sp = spm.SentencePieceProcessor(); sp.load('{model_prefix}.model')")
    print("   >>> sp.encode('Hello world!', out_type=int)")

if __name__ == "__main__":
    main()