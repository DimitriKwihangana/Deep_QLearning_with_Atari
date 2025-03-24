import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import ale_py

# Register ALE environments
gym.register_envs(ale_py)

class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals.get("dones"):
            episode_info = self.locals.get("infos")[0].get("episode", {})
            self.episode_rewards.append(episode_info.get("r", 0))
            self.episode_lengths.append(episode_info.get("l", 0))
        return True

# Initialize environment
env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")

# Define hyperparameters
hyperparams = {
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "batch_size": 32,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "exploration_fraction": 0.1
}

# Create model
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=hyperparams["learning_rate"],
    gamma=hyperparams["gamma"],
    batch_size=hyperparams["batch_size"],
    exploration_initial_eps=hyperparams["exploration_initial_eps"],
    exploration_final_eps=hyperparams["exploration_final_eps"],
    exploration_fraction=hyperparams["exploration_fraction"],
    verbose=1
)

# Train the model
total_timesteps = 100000
logging_callback = LoggingCallback()
model.learn(total_timesteps=total_timesteps, callback=logging_callback)

# Save the trained model
model.save("dqn2_model.zip")

# Log final training details
print("Training complete!")
print("Average Episode Reward:", np.mean(logging_callback.episode_rewards))
print("Average Episode Length:", np.mean(logging_callback.episode_lengths))
