import gymnasium as gym
from stable_baselines3 import DQN
import time
import ale_py

# Create the same Atari environment used in training
gym.register_envs(ale_py)
env = gym.make("ALE/Boxing-v5", render_mode="human")

# Load the trained model with smaller buffer size
# Set buffer_size=1 since we're just doing inference and don't need a replay buffer
model = DQN.load("dqn2_model.zip", buffer_size=1)

# Run a few episodes to evaluate performance
num_episodes = 2

for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    
    while not (done or truncated):
        # For evaluation, use the greedy policy: select action with highest Q-value
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        # Render the environment
        env.render()
        time.sleep(0.01)  # Add a short delay for visualization purposes
    
    print(f"Episode {episode+1} finished with reward: {episode_reward}")

env.close()