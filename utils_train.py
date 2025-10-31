"""
utils_train.py
Shared utilities for training (AR and Diffusion)
------------------------------------------------
Includes:
- split_dataset(): deterministic 90/10 train/val split
- EarlyStopper: tracks validation loss, patience, and triggers early stopping
- save_checkpoint(): saves last.pt and best.pt
- plot_losses(): live interactive matplotlib plotting of training/validation loss curves
"""

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split

# ----------------------------------------------------
# Dataset splitting
# ----------------------------------------------------
def split_dataset(dataset, val_ratio=0.1, seed=1337):
    """Deterministically split dataset into train/val (default 90/10)."""
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    return train_ds, val_ds

# ----------------------------------------------------
# Checkpoint utilities
# ----------------------------------------------------
def save_checkpoint(model, optimizer, step, val_loss, out_dir, tag="last", best=False):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{tag}.pt")
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'val_loss': val_loss,
    }, ckpt_path)
    if best:
        best_path = os.path.join(out_dir, 'best.pt')
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'val_loss': val_loss,
        }, best_path)
    print(f"💾 Saved checkpoint to {ckpt_path} ({'best' if best else tag})")

# ----------------------------------------------------
# Early stopping
# ----------------------------------------------------
class EarlyStopper:
    def __init__(self, patience=5, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    def step(self, val_loss):
        improved = val_loss < (self.best_loss - self.delta)
        if improved:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved

class LiveLossPlot:
    def __init__(self, title="Training vs Validation Loss", save_path=None):
        """
        Colab-safe loss plotter.
        - No interactive display
        - Saves a single PNG plot that updates each eval
        """
        self.title = title
        self.save_path = save_path or "training_curve.png"
        self.train_losses = []
        self.val_losses = []
        self.steps = []

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def update(self, step, train_loss, val_loss=None):
        """Record new losses and update the saved plot."""
        self.steps.append(step)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

        plt.figure()
        plt.plot(self.steps, self.train_losses, label="Train Loss", color="tab:blue")
        if len(self.val_losses) > 0:
            val_x = self.steps[:len(self.val_losses)]
            plt.plot(val_x, self.val_losses, label="Val Loss", color="tab:orange")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(self.title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()
        print(f"📉 Plot updated → {self.save_path}")

    def close(self):
        """No-op placeholder for API compatibility."""
        pass


def plot_losses(save_dir=None, title="Training vs Validation Loss"):
    """
    Creates a non-interactive plotter that saves a single .png file.
    The file updates each eval and lives under results/ by default.
    """
    if save_dir is None:
        save_path = "training_curve.png"
    else:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "training_curve.png")
    return LiveLossPlot(title=title, save_path=save_path)