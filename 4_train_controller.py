import os
import json
import glob
import gymnasium as gym
import numpy as np
import torch
import cma
import matplotlib.pyplot as plt

# Import verified architectures from your package namespace
from src import VAE, MDNRNN, Controller

def set_controller_params(model, flat_params):
    """
    Unflattens a 1D numpy array of weights and injects them 
    directly into the Controller's PyTorch parameters.
    """
    idx = 0
    for p in model.parameters():
        size = p.numel()
        # Extract slices matching the parameter shape
        p_data = flat_params[idx:idx+size].reshape(p.shape)
        p.data = torch.from_numpy(p_data).float().to(p.device)
        idx += size

def evaluate_agent(flat_params, env, vae, rnn, controller, device, max_steps=1000):
    """
    Runs a single complete episode rollout using the provided parameter vector,
    combining VAE latent codes and full LSTM states (h + c) for the Controller.
    """
    set_controller_params(controller, flat_params)
    
    obs, _ = env.reset()
    hidden = None  
    total_reward = 0
    
    # Initialize combined LSTM state (h is 256, c is 256 -> total 512) with a batch dimension
    lstm_state = torch.zeros(1, 512, device=device)
    
    for step in range(max_steps):
        # 1. Transform and normalize current visual frame
        x = torch.from_numpy(obs.copy()).float().permute(2, 0, 1).to(device) / 255.0
        x = x.unsqueeze(0) # Shape: (1, 3, 96, 96)
        
        with torch.no_grad():
            _, z_t, _ = vae(x) # Shape: (1, 32)
            
        # 2. Compute control action using both z_t (1, 32) and the full lstm_state (1, 512)
        with torch.no_grad():
            action = controller(z_t, lstm_state) # Combined input shape inside forward: 1, 544
            action_np = action.squeeze(0).cpu().numpy() # Squeeze batch dimension for Gym step (3,)
            
        # 3. Step environment execution
        obs, reward, terminated, truncated, _ = env.step(action_np)
        total_reward += reward
        
        if terminated or truncated:
            break
            
        # 4. Update temporal state trajectory tracking for the next time-step
        with torch.no_grad():
            z_input = z_t.view(1, 1, -1)     # Shape: (Seq=1, Batch=1, 32)
            a_input = action.view(1, 1, -1)  # Shape: (Seq=1, Batch=1, 3)
            _, _, _, hidden = rnn(z_input, a_input, hidden)
            
            # hidden is a tuple: (h_n, c_n), each with shape (num_layers, batch, hidden_dim) -> (1, 1, 256)
            h_n, c_n = hidden[0], hidden[1]
            
            # Concatenate h and c along the feature dimension to form the next 512-dim state
            lstm_state = torch.cat([h_n.view(1, -1), c_n.view(1, -1)], dim=-1) # Shape: (1, 512)
            
    return total_reward
def save_and_plot_controller(history, checkpoint_dir):
    """Logs evolution histories and saves optimization charts."""
    json_path = os.path.join(checkpoint_dir, "controller_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=4)
        
    generations = history["generation"]
    plt.figure(figsize=(8, 5))
    plt.plot(generations, history["best_reward"], color='#d62728', marker='o', linewidth=2, label='Best Reward')
    plt.plot(generations, history["avg_reward"], color='#1f77b4', linestyle='--', marker='s', label='Generation Avg')
    
    plt.title('Controller Evolution: Driving Policy Optimization', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Generations', fontsize=11)
    plt.ylabel('Cumulative Reward Score', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    
    plot_path = os.path.join(checkpoint_dir, "controller_training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Controller analytics plot successfully saved to: {plot_path}")
    plt.close()

def train_controller():
    # Optimization Parameters
    GENERATIONS = 10
    POPULATION_SIZE = 16
    CHECKPOINT_DIR = "checkpoints"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading feature pipelines on hardware device: {device}")
    
    # 1. Initialize complete evaluation environment (render_mode can be toggled)
    env = gym.make("CarRacing-v3")
    
    # 2. Instantiate and load pre-trained network states
    vae = VAE(latent_dim=32).to(device)
    vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "vae.pth"), map_location=device))
    vae.eval()
    
    rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5).to(device)
    rnn.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "mdn_rnn.pth"), map_location=device))
    rnn.eval()
    
    # 3. Instantiate the target Controller layer to find its parameter configuration bounds
    controller = Controller(latent_dim=288, action_dim=3).to(device)
    
    # Calculate the total flat parameter footprint dimension (867 parameters)
    initial_params = np.concatenate([p.cpu().detach().numpy().flatten() for p in controller.parameters()])
    num_params = len(initial_params)
    print(f"Total parameters found for Controller optimization: {num_params}")
    
    # 4. Configure CMA-ES Solver strategy
    # initial_params: center of search space, 0.1: coordinate mutation step size (sigma)
    es = cma.CMAEvolutionStrategy(num_params * [0.0], 0.1, {'popsize': POPULATION_SIZE})
    
    history = {"generation": [], "best_reward": [], "avg_reward": []}
    
    print("\n--- Starting Controller Evolution Phase ---")
    for gen in range(1, GENERATIONS + 1):
        # Query solver for candidate solution weight vectors
        solutions = es.ask()
        rewards = []
        
        print(f"Generation [{gen:02d}/{GENERATIONS:02d}] evaluating {POPULATION_SIZE} candidate agents...")
        for idx, candidate in enumerate(solutions):
            # Run simulation episode rollout
            reward = evaluate_agent(candidate, env, vae, rnn, controller, device)
            rewards.append(reward)
            print(f"  -> Agent {idx+1:02d}/{POPULATION_SIZE:02d} achieved score: {reward:.2f}")
            
        # CMA-ES minimizes objectives; invert rewards to optimize maximum parameters
        cost_objectives = [-r for r in rewards]
        es.tell(solutions, cost_objectives)
        
        # Calculate generation analytics
        best_gen_reward = max(rewards)
        avg_gen_reward = np.mean(rewards)
        print(f"Generation summary | Best Score: {best_gen_reward:.2f} | Average: {avg_gen_reward:.2f}\n")
        
        history["generation"].append(gen)
        history["best_reward"].append(best_gen_reward)
        history["avg_reward"].append(avg_gen_reward)
        
        # Extract the current optimal parameters mean vector and save to disk
        best_flat_params = es.result[0]
        set_controller_params(controller, best_flat_params)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "controller.pth")
        torch.save(controller.state_dict(), checkpoint_path)
        
    env.close()
    print(f"Evolution Process Complete! Best policy saved to: {checkpoint_path}")
    save_and_plot_controller(history, CHECKPOINT_DIR)

if __name__ == "__main__":
    train_controller()