## Autonomous Vehicles Simulator using Deep Reinforcement Learning 

## Contributor:
- **Louis Sungwoo Cho Civil Engineering (Transportation), Computer Science Minor**

## Project Description:
This project presents a Deep Reinforcement Learning (DRL) framework aimed at developing an intelligent agent that can control an autonomous vehicle in a simplified environment. The agent learns to navigate towards a goal, specifically another car, through trial and error.

Components:

Prioritized Replay Buffer:

A data structure used to store and retrieve experiences (transitions) that the agent observes, which is essential for Experience Replay in DRL.
Instead of sampling uniformly, this buffer samples transitions based on their importance, measured by their Temporal Difference (TD) error. This allows for a more efficient and meaningful learning experience.
AutonomousVehicleEnv (Environment):

Built on the gym framework, it simulates a one-dimensional environment where an autonomous car needs to approach a stationary leading car.
The observation space consists of distance and relative speed.
The agent can choose one of three possible actions: accelerate forward, remain in place, or decelerate.
The environment provides visual feedback using pygame library, depicting the positions of both cars and the current state information.
DeepQNetwork (Model):

A neural network architecture that approximates the Q-value function.
Consists of three linear layers, where the middle layer uses ReLU activation functions.
Agent:

Represents the decision-making entity that interacts with the environment.
Employs a Q-learning based algorithm, combined with a neural network for function approximation.
Utilizes the Epsilon-Greedy policy for exploration-exploitation tradeoff.
Stores experiences in the Prioritized Replay Buffer and samples from it to train the DeepQNetwork.
Over time, reduces its exploration rate (epsilon) and learns to make more informed decisions.
Training Procedure:
The agent is trained over a specified number of episodes, where in each episode:
- The agent selects an action based on the current state and its policy.
- The environment transitions to a new state and provides a reward.
- The agent stores this transition in its replay buffer.
- Periodically, the agent samples a batch from its buffer and trains its neural network.
- After a set number of episodes, the agent evaluates its performance, saves its model, and plots training metrics.

Metrics and Visualization:
The training progress is tracked using three primary metrics:
- Total rewards per episode: Indicates the cumulative reward the agent received in a single episode.
- Average rewards: Provides insight into the agent's performance over a range of episodes.
- Steps per episode: Reflects the number of actions the agent took in a single episode.
These metrics are visualized using matplotlib, allowing for a comprehensive understanding of the agent's learning progress.

Applications:
While this project presents a simplified representation, the concepts and methodologies can be scaled and applied to more complex autonomous driving scenarios, urban environments, and multi-agent systems.

Libraries and Tools:

PyTorch: For building and training the neural network.
gym: To define the reinforcement learning environment.
pygame: For visualizing the environment and rendering graphics.
numpy and matplotlib: For numerical operations and visualization, respectively.
