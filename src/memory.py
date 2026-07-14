import torch
import torch.nn as nn

class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5):
        super(MDNRNN, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians

        # LSTM Layer: Takes concatenated (z_t + a_t) as input
        # Input size: 32 + 3 = 35
        self.lstm = nn.LSTM(input_size=latent_dim + action_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=False) # Shape: (Seq_Len, Batch, Features)

        # MDN Output Heads to parameterize the Mixture of Gaussians for z_{t+1}
        # Pi: Mixing coefficients (probabilities for each Gaussian component)
        self.fc_pi = nn.Linear(hidden_dim, num_gaussians)
        
        # Mu: Means of the Gaussians
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        
        # Sigma: Standard deviations of the Gaussians (must be positive)
        self.fc_sigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)

    def forward(self, z, action, hidden=None):
        """
        z: Tensor of shape (Seq_Len, Batch, Latent_Dim)
        action: Tensor of shape (Seq_Len, Batch, Action_Dim)
        hidden: Thread state tuple (h, c) for the LSTM
        """
        # Concatenate latent vector and action along the feature dimension
        # Resulting shape: (Seq_Len, Batch, Latent_Dim + Action_Dim)
        lstm_input = torch.cat([z, action], dim=-1)
        
        # Forward pass through LSTM
        output, hidden = self.lstm(lstm_input, hidden)
        
        # Output shape from LSTM: (Seq_Len, Batch, Hidden_Dim)
        seq_len, batch_size, _ = output.size()

        # Map LSTM hidden states to MDN parameters
        raw_pi = self.fc_pi(output) # (Seq_Len, Batch, Num_Gaussians)
        # Apply softmax over the Gaussians dimension so coefficients sum to 1
        pi = torch.softmax(raw_pi, dim=-1)

        mu = self.fc_mu(output) # (Seq_Len, Batch, Num_Gaussians * Latent_Dim)
        mu = mu.view(seq_len, batch_size, self.num_gaussians, self.latent_dim)

        raw_sigma = self.fc_sigma(output)
        raw_sigma = raw_sigma.view(seq_len, batch_size, self.num_gaussians, self.latent_dim)
        # Standard deviation must be strictly positive, use exp
        sigma = torch.exp(raw_sigma)

        return pi, mu, sigma, hidden

if __name__ == "__main__":
    print("Checking MDN-RNN module structure...")
    
    # Initialize the model using dimensions from the paper configuration
    model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5)
    
    # Simulate a true sequence processing pass based on your rollout dimensions
    # Sequence Length: 1000 time-steps, Batch Size: 1 rollout
    seq_len = 1000
    batch_size = 1
    
    mock_z = torch.randn(seq_len, batch_size, 32)
    mock_action = torch.randn(seq_len, batch_size, 3)
    
    print(f"Input Sequence Shape (z)      : {mock_z.shape} (Seq_Len x Batch x Latent)")
    print(f"Input Action Shape (action)   : {mock_action.shape} (Seq_Len x Batch x Action)")
    
    # Run forward pass
    pi, mu, sigma, (h, c) = model(mock_z, mock_action)
    
    print("\n--- Model Verification Pass ---")
    print(f"Pi (Mixing Weights) Shape     : {pi.shape} (Seq_Len x Batch x Gaussians)")
    print(f"Mu (Gaussian Means) Shape     : {mu.shape} (Seq_Len x Batch x Gaussians x Latent)")
    print(f"Sigma (Gaussian Std) Shape    : {sigma.shape} (Seq_Len x Batch x Gaussians x Latent)")
    print(f"LSTM Final Hidden State Shape : {h.shape} (Layers x Batch x Hidden_Dim)")
    
    print("\nVerification complete! MDN-RNN architecture compiles perfectly.")