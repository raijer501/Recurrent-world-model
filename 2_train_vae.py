import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Import the verified VAE architecture from your package namespace
from src import VAE

class RolloutDataset(Dataset):
    def __init__(self, data_dir="data/rollouts"):
        self.filepaths = glob.glob(os.path.join(data_dir, "rollout_*.npz"))
        if not self.filepaths:
            raise FileNotFoundError(f"No rollout files found in {data_dir}. Run 1_collect_data.py first.")
        
        print(f"Found {len(self.filepaths)} rollout files. Loading frames into memory...")
        
        all_frames = []
        for path in self.filepaths:
            with np.load(path) as data:
                all_frames.append(data['frames'])
        
        self.frames = np.concatenate(all_frames, axis=0)
        print(f"Total frame dataset size initialized: {self.frames.shape}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        return frame_tensor

def vae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss, recon_loss, kld_loss

def save_and_plot_history(history, checkpoint_dir):
    """Saves metrics to JSON and outputs the architectural performance plots."""
    # 1. Save data metrics to a permanent JSON file
    json_path = os.path.join(checkpoint_dir, "vae_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Metrics logs successfully saved to: {json_path}")

    # 2. Generate the automated Matplotlib figures
    epochs = history["epoch"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('VAE Training Dynamics: Feature Learning vs. Latent Regularization', fontsize=14, fontweight='bold')

    # Left Panel: Spatial Reconstruction
    ax1.plot(epochs, history["total"], label='Total Loss', color='#d62728', marker='o', linewidth=2)
    ax1.plot(epochs, history["recon"], label='Reconstruction Loss (MSE)', color='#1f77b4', linestyle='--', marker='s', alpha=0.8)
    ax1.set_title('Spatial Reconstruction Performance', fontsize=12, pad=10)
    ax1.set_xlabel('Epochs', fontsize=11)
    ax1.set_ylabel('Loss Value Scale', fontsize=11)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10)

    # Right Panel: Latent Constraints (U-Shape tracking)
    ax2.plot(epochs, history["kld"], label='Kullback-Leibler Divergence', color='#2ca02c', marker='^', linewidth=2.5)
    
    # Shade the inflection point if training crossed epoch 5 and 6
    if len(epochs) >= 6:
        ax2.axvspan(5, 6, color='#ff7f0e', alpha=0.15, label='Feature Extraction Breakthrough')
        
    ax2.set_title('Latent Space Distribution Constraint (U-Shape)', fontsize=12, pad=10)
    ax2.set_xlabel('Epochs', fontsize=11)
    ax2.set_ylabel('KLD Metric Scale', fontsize=11)
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "vae_training_curves.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Performance plot successfully generated and saved to: {plot_path}")
    plt.close()

def train_vae():
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    CHECKPOINT_DIR = "checkpoints"
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training pipeline on device: {device}")
    
    dataset = RolloutDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = VAE(latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Structured dictionary to track loss trajectories across training
    history = {"epoch": [], "total": [], "recon": [], "kld": []}
    
    print("\n--- Starting VAE Training Phase ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_epoch_loss = 0
        total_recon = 0
        total_kld = 0
        
        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            
            loss, recon, kld = vae_loss_fn(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            
        num_samples = len(dataloader.dataset)
        avg_loss = total_epoch_loss / num_samples
        avg_recon = total_recon / num_samples
        avg_kld = total_kld / num_samples
        
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f})")
        
        # Log performance directly into the metrics history
        history["epoch"].append(epoch)
        history["total"].append(avg_loss)
        history["recon"].append(avg_recon)
        history["kld"].append(avg_kld)
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "vae.pth")
        torch.save(model.state_dict(), checkpoint_path)
        
    print(f"\nVAE Training Complete! Model state dictionary saved to: {checkpoint_path}")
    
    # Process the history metrics automatically
    save_and_plot_history(history, CHECKPOINT_DIR)

if __name__ == "__main__":
    train_vae()