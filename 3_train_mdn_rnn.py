import os
import glob
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Import verified architectures from your package namespace
from src import VAE, MDNRNN

class RNNSequenceDataset(Dataset):
    """
    Dataset that handles full sequential episodes. Pre-extracts visual frames 
    into latent vectors using a trained VAE to maximize training speed.
    """
    def __init__(self, data_dir="data/rollouts", vae_checkpoint="checkpoints/vae.pth", device="cpu"):
        self.filepaths = glob.glob(os.path.join(data_dir, "rollout_*.npz"))
        if not self.filepaths:
            raise FileNotFoundError(f"No rollout files found in {data_dir}.")
        
        # Load the trained VAE to encode frames once
        vae = VAE(latent_dim=32).to(device)
        vae.load_state_dict(torch.load(vae_checkpoint, map_location=device))
        vae.eval()
        
        self.z_sequences = []
        self.action_sequences = []
        
        print(f"Pre-processing and encoding {len(self.filepaths)} rollouts into latent space...")
        
        with torch.no_grad():
            for path in self.filepaths:
                with np.load(path) as data:
                    frames = data['frames']    # Shape: (N, 96, 96, 3)
                    actions = data['actions']  # Shape: (N, 3)
                    
                    # Process frames in small batches to prevent GPU memory overflow
                    z_list = []
                    for i in range(0, len(frames), 64):
                        batch_frames = frames[i:i+64]
                        # Format and normalize to [0, 1]
                        x = torch.from_numpy(batch_frames).float().permute(0, 3, 1, 2).to(device) / 255.0
                        _, mu, _ = vae(x) # Extract the deterministic mean as the latent code
                        z_list.append(mu.cpu().numpy())
                    
                    z_seq = np.concatenate(z_list, axis=0)
                    self.z_sequences.append(z_seq)
                    self.action_sequences.append(actions)
                    
        print("Latent extraction complete. All sequences cached.")

    def __len__(self):
        return len(self.z_sequences)

    def __getitem__(self, idx):
        # Return full time-series arrays for sequential LSTM processing
        return (torch.tensor(self.z_sequences[idx], dtype=torch.float32), 
                torch.tensor(self.action_sequences[idx], dtype=torch.float32))

def mdn_loss_fn(pi, mu, sigma, target):
    """
    Computes the Negative Log-Likelihood (NLL) of the target under the 
    predicted Mixture of Gaussians distribution.
    """
    # target shape: (Seq_Len, Batch, 32) -> expand to (Seq_Len, Batch, 1, 32)
    target = target.unsqueeze(2)
    
    # Calculate the log probability density function (PDF) for each Gaussian component
    # Formula: -0.5 * log(2*pi) - log(sigma) - (target - mu)^2 / (2 * sigma^2)
    log_gaussian = -0.5 * np.log(2 * np.pi) - torch.log(sigma) - (target - mu).pow(2) / (2 * sigma.pow(2))
    
    # Sum across the 32 latent dimensions to get the log-likelihood of the vector per component
    log_gaussian = torch.sum(log_gaussian, dim=-1) # Shape: (Seq_Len, Batch, 5)
    
    # Combine component weights with their respective likelihoods: log(pi) + log(N)
    log_pi = torch.log(pi + 1e-8)
    combined = log_pi + log_gaussian
    
    # Mathematically stable summation across the 5 mixture components using logsumexp
    log_total_likelihood = torch.logsumexp(combined, dim=-1) # Shape: (Seq_Len, Batch)
    
    # Return the negative mean log-likelihood
    return -torch.mean(log_total_likelihood)

def save_and_plot_rnn(history, checkpoint_dir):
    """Saves metrics to JSON and plots temporal loss reduction trends."""
    json_path = os.path.join(checkpoint_dir, "mdn_rnn_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=4)
        
    epochs = history["epoch"]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], color='#1f77b4', marker='o', linewidth=2, label='MDN-RNN NLL Loss')
    plt.title('Memory Module Training: Temporal Predictive Accuracy', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Negative Log-Likelihood (NLL)', fontsize=11)
    plt.xticks(epochs)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    
    plot_path = os.path.join(checkpoint_dir, "mdn_rnn_training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"MDN-RNN metrics saved. Performance curve generated at: {plot_path}")
    plt.close()

def train_mdn_rnn():
    # Hyperparameters
    BATCH_SIZE = 1 # Process one complete sequence sequence at a time to maintain clean timeline boundaries
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    CHECKPOINT_DIR = "checkpoints"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing MDN-RNN training pipeline on: {device}")
    
    # Initialize sequential loader
    dataset = RNNSequenceDataset(data_dir="data/rollouts", vae_checkpoint="checkpoints/vae.pth", device=device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model Initialization
    model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {"epoch": [], "loss": []}
    
    print("\n--- Starting MDN-RNN Training Phase ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        
        for z_seq, action_seq in dataloader:
            # Drop batch wrapper added by DataLoader: shape shifts to (Seq_Len, 32) and (Seq_Len, 3)
            z_seq = z_seq.squeeze(0).to(device)
            action_seq = action_seq.squeeze(0).to(device)
            
            # Temporal Shift Alignments:
            # Inputs look at step t, targets look forward to step t+1
            z_input = z_seq[:-1].unsqueeze(1)       # Shape: (Seq_Len-1, Batch=1, 32)
            actions = action_seq[:-1].unsqueeze(1)   # Shape: (Seq_Len-1, Batch=1, 3)
            z_target = z_seq[1:].unsqueeze(1)        # Shape: (Seq_Len-1, Batch=1, 32)
            
            optimizer.zero_grad()
            
            # Forward pass through recurrent layers
            pi, mu, sigma, _ = model(z_input, actions)
            
            # Compute loss against shifted target parameters
            loss = mdn_loss_fn(pi, mu, sigma, z_target)
            loss.backward()
            
            # Gradient clipping to eliminate potential exploding gradients inside the LSTM architecture
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | MDN-RNN NLL Loss: {avg_epoch_loss:.4f}")
        
        history["epoch"].append(epoch)
        history["loss"].append(avg_epoch_loss)
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "mdn_rnn.pth")
        torch.save(model.state_dict(), checkpoint_path)
        
    print(f"\nMDN-RNN Optimization Complete! Model weights stored to: {checkpoint_path}")
    save_and_plot_rnn(history, CHECKPOINT_DIR)

if __name__ == "__main__":
    train_mdn_rnn()