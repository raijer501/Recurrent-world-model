import os
import gymnasium as gym
import numpy as np
import torch
import imageio

# Import verified architectures from your package namespace
from src import VAE, MDNRNN, Controller

def run_demo():
    CHECKPOINT_DIR = "checkpoints"
    device = torch.device("cpu")
    print("Initializing World Model demonstration on CPU...")
    
    # Use rgb_array so we can render without a physical monitor
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    # Load checkpoints
    vae = VAE(latent_dim=32).to(device)
    vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "vae.pth"), map_location=device))
    vae.eval()
    
    rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5).to(device)
    rnn.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "mdn_rnn.pth"), map_location=device))
    rnn.eval()
    
    controller = Controller(latent_dim=32, hidden_dim=512, action_dim=3).to(device)
    controller.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "controller.pth"), map_location=device))
    controller.eval()
    
    obs, _ = env.reset()
    hidden = None  
    lstm_state = torch.zeros(1, 512, device=device)
    
    frames = []
    max_steps = 1000  # Cap execution to exactly 1 full episode
    
    print("\nNetworks loaded successfully. Starting live evaluation episode...")
    
    try:
        for step in range(max_steps):
            # Capture the current frame as an RGB array
            frame = env.render()
            frames.append(frame)
            
            # 1. Process and normalize current visual frame
            x = torch.from_numpy(obs.copy()).float().permute(2, 0, 1).to(device) / 255.0
            x = x.unsqueeze(0)
            
            with torch.no_grad():
                _, z_t, _ = vae(x)
                
            # 2. Pass current latent features and LSTM state to Controller
            with torch.no_grad():
                action = controller(z_t, lstm_state)
                action_np = action.squeeze(0).cpu().numpy()
                
            # 3. Step the environment using the predicted action
            obs, reward, terminated, truncated, _ = env.step(action_np)
            
            if terminated or truncated:
                print(f"Episode completed or truncated at step {step}.")
                break
                
            # 4. Feed features forward through the RNN to update the hidden state
            with torch.no_grad():
                z_input = z_t.view(1, 1, -1)
                a_input = action.view(1, 1, -1)
                _, _, _, hidden = rnn(z_input, a_input, hidden)
                
                h_n, c_n = hidden[0], hidden[1]
                lstm_state = torch.cat([h_n.view(1, -1), c_n.view(1, -1)], dim=-1)
                
    except KeyboardInterrupt:
        print("\nDemonstration stopped by user.")
    finally:
        env.close()

    # Save the recorded frames as an MP4 file
    if len(frames) > 0:
        video_path = os.path.join(CHECKPOINT_DIR, "demo_drive.mp4")
        print(f"\nCompiling {len(frames)} frames into video...")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video successfully saved to {video_path}")
    else:
        print("No frames were captured.")

if __name__ == "__main__":
    run_demo()