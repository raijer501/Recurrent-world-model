import os
import gymnasium as gym
import numpy as np

def collect_random_rollouts(num_episodes=3, save_dir="data/rollouts"):
    os.makedirs(save_dir, exist_ok=True)
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        frames = []
        actions = []
        terminated = False
        truncated = False
        
        print(f"Starting rollout {episode + 1}...")
        
        while not (terminated or truncated):
            action = env.action_space.sample()  # Random agent decisions
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Keep track of frames and actions
            frames.append(observation)
            actions.append(action)
            
        # Save sequence down as a compressed numpy array file
        np.savez(os.path.join(save_dir, f"rollout_{episode}.npz"), 
                 frames=np.array(frames), 
                 actions=np.array(actions))
        
    env.close()
    print(f"Successfully saved {num_episodes} rollouts to {save_dir}!")

if __name__ == "__main__":
    collect_random_rollouts()