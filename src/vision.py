import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: Downsample from 3x96x96 to 256x6x6
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> 32x48x48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 64x24x24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# -> 128x12.x12
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# -> 256x6x6
            nn.ReLU(),
            nn.Flatten() # -> 256 * 6 * 6 = 9216
        )
        
        # Latent space representations
        self.fc_mu = nn.Linear(9216, latent_dim)
        self.fc_logvar = nn.Linear(9216, latent_dim)
        
        # Decoder input mapping
        self.decoder_input = nn.Linear(latent_dim, 9216)
        
        # Decoder: Upsample back from 256x6x6 to 3x96x96
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 6, 6)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 128x12x12
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 64x24x24
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> 32x48x48
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # -> 3x96x96
            nn.Sigmoid() # Scale output pixel values between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        """The reparameterization trick allows backpropagation through stochastic nodes."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(self.decoder_input(z))
        return reconstruction, mu, logvar

if __name__ == "__main__":
    import numpy as np
    import os

    print("Checking VAE module structure...")
    model = VAE(latent_dim=32)
    
    # Target path for your collected data
    file_path = "data/rollouts/rollout_1.npz"
    
    if os.path.exists(file_path):
        print(f"Loading real frame sequence from {file_path} for testing...")
        raw_data = np.load(file_path)
        # Grab a batch of 10 frames from your rollout file
        raw_frames = raw_data['frames'][:10] 
        
        # Convert Gym frames (0-255, HWC) to PyTorch layout (0.0-1.0, CHW)
        test_tensor = torch.from_numpy(raw_frames).float() / 255.0
        test_tensor = test_tensor.permute(0, 3, 1, 2) # (10, 96, 96, 3) -> (10, 3, 96, 96)
    else:
        print("Rollout file not found in src/ folder context, fallback to mock data...")
        test_tensor = torch.randn(10, 3, 96, 96)

    print(f"Input batch tensor shape: {test_tensor.shape}")
    
    # Run the tensor forward through your network
    recon, mu, logvar = model(test_tensor)
    
    print("\n--- Model Verification Pass ---")
    print(f"Reconstructed output shape : {recon.shape} (Should match input shape)")
    print(f"Latent mean (mu) shape      : {mu.shape}    (Should be batch_size x latent_dim)")
    print(f"Latent variance shape      : {logvar.shape}    (Should be batch_size x latent_dim)")
    print("Verification complete! VAE architecture compiles perfectly.")