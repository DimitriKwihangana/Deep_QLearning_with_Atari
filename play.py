import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import GreedyQPolicy
import time

# Create environment
env = gym.make("ALE/Boxing-v5", render_mode="human")

# Load the trained model
model = DQN.load("dqn2_model.zip")

# Set the policy to greedy evaluation mode
model.policy.set_training_mode(False)

num_episodes = 3

for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    
    while not (done or truncated):
        # Use model.predict with deterministic=True for greedy policy
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        env.render()
        time.sleep(0.01)
    
    print(f"Episode {episode+1} finished with reward: {episode_reward}")

env.close()