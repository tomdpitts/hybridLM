from datasets import load_dataset
import os

# Load the "clean" configuration explicitly
dataset = load_dataset("yehzw/wikitext-103", "clean")
os.makedirs("data", exist_ok=True)

for split in ["train", "validation", "test"]:
    path = f"data/{split}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for entry in dataset[split]["text"]:
            if isinstance(entry, list) and entry:
                text = " ".join(entry).strip()
            elif isinstance(entry, str):
                text = entry.strip()
            else:
                continue
            if text:
                f.write(text + "\n")
    print(f"✅ Saved {path} ({os.path.getsize(path)/1e6:.1f} MB)")