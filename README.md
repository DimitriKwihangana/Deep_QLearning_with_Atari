## Deep Q Learning with Atari

This project uses Stable Baselines3 and Gymnasium to train and evaluate a Deep Q-network (DQN) agent in an Atari game environment.

## Project Structure

train.py - Script for training the DQN agent.

play.py - Script for playing with the trained agent.

train.ipynb - Jupyter notebook version of training script.

dqn2_model.zip - The trained model.

requirements.txt - Dependencies required for the project.

README.md - Project documentation.

## Environment Selection

An Atari environment from Gymnasium is used. Example: ALE/Boxing-v5.

Task 1: Training the Agent (train.py)

## Objective:

Train a DQN agent to play the chosen Atari game.

Steps:

Define the Agent

Use Stable Baselines3’s DQN.

Compare MLPPolicy and CNNPolicy to determine the best fit.

Train the Agent

Train the agent in the environment.

Save the trained model as dqn2_model.zip.

Log reward trends and episode lengths.

Hyperparameter Tuning

Experiment with different values for:

Learning rate (lr)

Discount factor (gamma)

Batch size (batch_size)

Exploration-exploitation trade-off (epsilon_start, epsilon_end, epsilon_decay)

Document Observations

Record hyperparameter effects in a table:

Learning Rate

Gamma

Batch Size

Epsilon Start

Epsilon End

Epsilon Decay

Notes

0.001

0.99

32

1.0

0.1

10,000

Initial training

0.0005

0.98

64

1.0

0.05

20,000

Improved performance

Task 2: Playing the Game (play.py)

Objective:

Evaluate the trained agent’s performance.

Steps:

Load the Model

DQN.load("dqn2_model.zip")

Set Up the Environment

Use the same Atari environment as in training.

Use Greedy Policy

The agent selects actions with the highest Q-value for evaluation.

Run and Render the Game

Play a few episodes.

Use env.render() to visualize performance.

Installation

Requirements:

Install dependencies using:

pip install -r requirements.txt

Running the Training Script:

python train.py

Running the Playing Script:

python play.py

Results

The trained agent should demonstrate improved gameplay after training.

Performance should be analyzed based on reward trends and evaluation runs.



