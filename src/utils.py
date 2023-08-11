import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 
import matplotlib.pyplot as plt 
import random 
import gym 
from gym import spaces
from collections import deque
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class PrioritizedReplayBuffer:
    def __init__(self, max_mem_size, input_dims, n_actions, alpha=0.6):
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.alpha = alpha
        self.transitions = deque(maxlen=max_mem_size)
        self.priorities = deque(maxlen=max_mem_size)
    
    def store_transition(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1)  # Default to 1 for the first transition
        self.transitions.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample_buffer(self, batch_size, beta=0.4):
        # Get the probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample transitions based on their probabilities
        indices = np.random.choice(len(self.transitions), batch_size, p=probs)
        samples = [self.transitions[idx] for idx in indices]

        # Calculate importance-sampling weights
        weights = (len(self.transitions) * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), indices, np.array(weights)

    def set_priorities(self, indices, errors, offset=1e-6):
        # Set the priorities based on TD errors
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + offset


class AutonomousVehicleEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Define action space (acceleration: -1, 0, 1)
        self.action_space = spaces.Discrete(3)

        # Define observation space (distance, relative_speed)
        self.observation_space = spaces.Box(low=np.array([0, -10]), high=np.array([100, 10]), dtype=np.float32)

        self.reset()

    def reset(self):
        # Initialize distance and relative speed
        self.distance = np.random.uniform(30, 70)
        self.relative_speed = np.random.uniform(-5, 5)
        return np.array([self.distance, self.relative_speed])

    def step(self, action):
        acceleration = action - 1  # Map 0, 1, 2 to -1, 0, 1

        # Update state
        self.relative_speed += acceleration
        self.distance += self.relative_speed

        # Check for collision
        done = False
        reward = 0
        if self.distance <= 0:
            reward = -100
            done = True
        elif 10 < self.distance < 50:  # Safe distance reward
            reward = 10 - abs(self.relative_speed)

        return np.array([self.distance, self.relative_speed]), reward, done, {}

    def render(self, mode='human'):
        print(f"Distance: {self.distance:.2f}, Relative Speed: {self.relative_speed:.2f}")

    def close(self):
        pass


class DeepQNetwork(nn.Module):
    def __init__(self,input_dims,hidden_layers,output_dims):
        super(DeepQNetwork,self).__init__()
        self.layer1 = nn.Linear(input_dims,hidden_layers)
        self.layer2 = nn.Linear(hidden_layers,hidden_layers)
        self.layer3 = nn.Linear(hidden_layers,output_dims)

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x 
