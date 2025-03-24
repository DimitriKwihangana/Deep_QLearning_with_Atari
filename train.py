import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import os

# Custom callback for logging rewards and episode lengths
class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if "dones" in self.locals:
            episode_info = self.locals.get("infos")[0].get("episode", {})
            self.episode_rewards.append(episode_info.get("r", 0))
            self.episode_lengths.append(episode_info.get("l", 0))
        return True

# Custom memory-mapped replay buffer
class MemReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, memmap_path):
        self.buffer_size = buffer_size
        self.obs_shape = observation_space.shape
        self.device = device
        self.idx = 0
        self.memmap_path = memmap_path

        # Allocate disk-based buffers using np.memmap
        self.observations = np.memmap(memmap_path + '_obs.dat', dtype=np.uint8, mode='w+', shape=(buffer_size, *self.obs_shape))
        self.actions = np.memmap(memmap_path + '_actions.dat', dtype=np.uint8, mode='w+', shape=(buffer_size,))
        self.rewards = np.memmap(memmap_path + '_rewards.dat', dtype=np.float32, mode='w+', shape=(buffer_size,))
        self.dones = np.memmap(memmap_path + '_dones.dat', dtype=np.bool_, mode='w+', shape=(buffer_size,))

    def store(self, observation, action, reward, done):
        idx = self.idx % self.buffer_size
        self.observations[idx] = observation
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.idx += 1

    def sample(self, batch_size):
        idxs = np.random.randint(0, min(self.idx, self.buffer_size), size=batch_size)
        return (
            self.observations[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.dones[idxs]
        )

# Initialize environment
env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")

# Define hyperparameters
hyperparams = {
    "learning_rate": 5e-5,
    "gamma": 0.99,
    "batch_size": 32,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.01,
    "exploration_fraction": 0.3
}

# Set up replay buffer to use disk storage on D:
buffer_size = 1000000
memmap_path = 'D:\\Study\\BSE\\MachineLearning\\Deep_QLearning_with_Atari\\replay_buffer'

# Create model
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=hyperparams["learning_rate"],
    gamma=hyperparams["gamma"],
    batch_size=hyperparams["batch_size"],
    buffer_size=buffer_size,
    replay_buffer_class=MemReplayBuffer,
    replay_buffer_kwargs={'memmap_path': memmap_path},
    exploration_initial_eps=hyperparams["exploration_initial_eps"],
    exploration_final_eps=hyperparams["exploration_final_eps"],
    exploration_fraction=hyperparams["exploration_fraction"],
    verbose=1
)

# Train the model
total_timesteps = 1000000
logging_callback = LoggingCallback()
model.learn(total_timesteps=total_timesteps, callback=logging_callback)

# Save the trained model
model.save("dqn2_model.zip")

# Log final training details
print("Training complete!")
print("Average Episode Reward:", np.mean(logging_callback.episode_rewards))
print("Average Episode Length:", np.mean(logging_callback.episode_lengths))
