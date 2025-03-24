# Deep Q-Learning with Atari [(Video Link)](https://drive.google.com/file/d/1ZdCODAwb2h0Kl1zGzK1hkf5l8Pa9aS33/view?usp=sharing)

##  Project Overview
The aim of this project is to train an RL agent to play an Atari game by utilizing Stable Baselines3 and Gymnasium in conjunction with Deep Q-Learning. Training the agent and then assessing its performance with a trained model comprise the project.

## 🏆 Roles & Contributions  

### 1️ **Dimitri – Hyperparameter Optimization, Model Training & Environment Setup**  
Dimitri led the **optimization of model training**, testing multiple hyperparameter configurations to enhance agent performance. He experimented with different **learning rates, discount factors, batch sizes, and exploration strategies**, assessing their impact on training stability and convergence speed.  
- Set up and fine-tuned **Stable Baselines3** for efficient training.  
- Conducted controlled experiments with **CNN and MLP policies** to determine the best-performing architecture.  
- Managed training logs, ensuring smooth execution in Google Colab and resolving runtime issues.  
- Implemented **automated model checkpointing**, preventing data loss during long training sessions.  

### 2️ **Charite – Performance Analysis, Reward Tracking & Evaluation Framework**  
Charite was responsible for **analyzing and interpreting model performance** across different training runs. She ensured that episode rewards, learning curves, and model behaviors were properly documented.  
- Developed **custom evaluation scripts** (play.py) to test trained agents in real-time.  
- Processed raw **training logs** to extract meaningful insights on **reward progression and exploration efficiency**.  
- Implemented functions for **tracking episode performance metrics**, such as **average rewards, exploration rate decay, and policy stability**.  
- Contributed to the **comparison of CNN vs. MLP policies**, identifying strengths and weaknesses based on collected data.  

### 3️ **Guled – Visualization, Documentation & Results Interpretation**  
Guled handled the **visual representation of data**, making complex results more digestible through intuitive graphs and tables. His work ensured that model comparisons were clear and backed by evidence.  
- Designed **performance graphs** to visualize trends in **training rewards, loss functions, and episode durations**.  
- Created comparative plots to show how **different policies performed over time**.  
- Structured the **README documentation**, ensuring clear explanations of the **hyperparameter experiments and evaluation findings**.  
- Compiled **final conclusions** from all experiments, ensuring that results were presented effectively for decision-making.  

###  **Collaboration & Final Integration**  
Beyond individual tasks, we worked together to ensure:  
✔ **Seamless execution** of experiments, troubleshooting runtime issues collaboratively.  
✔ **Thorough documentation** of results, allowing for reproducibility.  
✔ **Well-structured comparisons**, ensuring that findings were backed by clear evidence.  


## Environment Selection
We selected an Atari game from the Gymnasium collection for training and evaluation. The environment provides a challenging yet structured reinforcement learning task that enables the agent to improve over time through deep Q-learning.


## 📜 Training Scripts
### 1️⃣ Training Script (train.py)
This script is responsible for training the DQN agent and saving the trained model for later evaluation.

**Key Steps:**
- Define the DQN agent using Stable Baselines3.
- Train the agent with different hyperparameters.
- Save the trained model as `dqn2_model.zip`.
- Log training details such as reward trends and episode length.

### 2️⃣ Playing Script (play.py)
This script loads the trained model and runs the agent in the environment for evaluation.

**Key Steps:**
- Load the trained DQN model.
- Use GreedyQPolicy to ensure optimal action selection.
- Render and visualize the game performance.

## 📊 Hyperparameter Tuning & Policy Comparison

| Experiment | Policy Used  | Hyperparameters | Observed Behavior |
|------------|-------------|----------------|--------------------|
| **Exp 1**  | **CNNPolicy** | lr=1e-4, gamma=0.99, batch=32, eps=0.1 → 0.01 | Initial training was slow with limited exploration, but the model learned stable strategies. |
| **Exp 2**  | **MLPPolicy** | lr=5e-4, gamma=0.99, batch=64, eps=0.2 → 0.01 | Faster initial learning but struggled with complex spatial dependencies, leading to lower performance. |
| **Exp 3**  | **CNNPolicy** | lr=5e-4, gamma=0.99, batch=64, eps=0.2 → 0.01 | Faster learning, better feature extraction, and improved long-term strategy development. |
| **Exp 4**  | **MLPPolicy** | lr=1e-3, gamma=0.95, batch=128, eps=0.3 → 0.05 | Faster initial learning but highly unstable in later episodes, failing to generalize well. |

## Reason why we chose CNNPolicy (Convolutional Neural Network) over MLPPolicy (Multilayer Perceptron)
From the experiments, **CNNPolicy outperformed MLPPolicy** by extracting **spatial features more effectively** from the image-based environment. While MLPPolicy learned quickly, it struggled to generalize due to its **lack of spatial awareness**. This confirms that CNNs are **better suited** for vision-based reinforcement learning tasks, making them the optimal choice for our Atari environment. 

##  Challenges Faced
1️⃣ **Training on CPU was slow** – We leveraged GPU resources for faster training.
2️⃣ **Rendering issues in headless environments** – We recorded gameplay for later evaluation.

## 🎬 Evaluation: Running play.py
After training, the agent was evaluated based on gameplay performance. The trained agent demonstrated improved decision-making and higher rewards compared to the baseline model. **The Episode Reward** achieved during evaluation was **10**.


## Conclusion
This project successfully trained an RL agent using Deep Q-Learning to play an Atari game. By tuning hyperparameters and evaluating performance, we demonstrated the effectiveness of reinforcement learning in complex environments. Future work could explore alternative RL algorithms such as PPO or A2C for comparison.

