import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, action_dim=3):
        super(Controller, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # A single linear layer mapping the combined state [z_t, h_t] to actions
        # Input size: 32 + 256 = 288
        # Output size: 3 (Steering, Gas, Brake)
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)

    def forward(self, z, h):
        """
        z: Latent vector from VAE of shape (Batch, Latent_Dim)
        h: Hidden state from RNN of shape (Batch, Hidden_Dim)
        """
        # Concatenate the current visual state and past memory vector
        # Shape: (Batch, Latent_Dim + Hidden_Dim)
        state_input = torch.cat([z, h], dim=-1)
        
        # Pass through the linear policy layer
        raw_action = self.fc(state_input)
        
        # Tailor outputs to match the CarRacing action space constraints:
        # Steering: [-1.0, 1.0] -> handled by Tanh
        # Gas:      [0.0, 1.0]  -> handled by Sigmoid
        # Brake:    [0.0, 1.0]  -> handled by Sigmoid
        steering = torch.tanh(raw_action[..., 0:1])
        gas = torch.sigmoid(raw_action[..., 1:2])
        brake = torch.sigmoid(raw_action[..., 2:3])
        
        # Recombine into a single action tensor
        action = torch.cat([steering, gas, brake], dim=-1)
        return action

if __name__ == "__main__":
    print("Checking Controller module structure...")
    
    # Initialize the Controller
    model = Controller(latent_dim=32, hidden_dim=256, action_dim=3)
    
    # Mock a single decision step (Batch size of 1)
    mock_z = torch.randn(1, 32)
    mock_h = torch.randn(1, 256)
    
    print(f"Input Latent Shape (z)      : {mock_z.shape} (Batch x Latent)")
    print(f"Input Hidden Shape (h)      : {mock_h.shape} (Batch x Hidden)")
    
    # Run a forward decision pass
    action = model(mock_z, mock_h)
    
    print("\n--- Model Verification Pass ---")
    print(f"Output Action Shape         : {action.shape} (Batch x Action_Dim)")
    
    # Extract the vector to verify the values fall within bounds
    act_values = action.detach().numpy()[0]
    print(f"Sample Generated Action     : Str: {act_values[0]:.3f} | Gas: {act_values[1]:.3f} | Brk: {act_values[2]:.3f}")
    
    print("\nVerification complete! Controller architecture compiles perfectly.")