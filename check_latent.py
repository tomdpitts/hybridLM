import torch
from model import ARLanguageModel, ARConfig
import sentencepiece as spm

# Load model
ckpt = torch.load("ckpts/ar.pt", map_location="cpu")
cfg = ARConfig(**ckpt["config"])
model = ARLanguageModel(cfg)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/spm.model")

prompt = "Methinks the lady doth protest too much"
ids = torch.tensor([sp.encode(prompt)], dtype=torch.long)
attn = torch.ones_like(ids)

# Get latent embedding
with torch.no_grad():
    z = model.encode(ids, attn, pool="mean")   # (1, latent_dim)

print("Latent z shape:", z.shape)
print("First 10 values:", z[0, :10])

# import matplotlib.pyplot as plt
# plt.plot(z[0].cpu().numpy())
# plt.title("Latent Embedding z")
# plt.show()