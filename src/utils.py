import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt 
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Initialize pygame
pygame.init()

# Constants
GRID_SIZE = 40
NUM_GRIDS = 15  # So, a 15x15 grid
SCREEN_WIDTH = GRID_SIZE * NUM_GRIDS
SCREEN_HEIGHT = GRID_SIZE * NUM_GRIDS

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class Environment:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('GridWorld with Streets Simulator')
        self.directions = [(0, -GRID_SIZE), (GRID_SIZE, 0), (0, GRID_SIZE), (-GRID_SIZE, 0)]  # UP, RIGHT, DOWN, LEFT
        
        # Defining street rows and columns
        self.street_rows = [2, 5, 8, 12]  # for example
        self.street_cols = [2, 5, 8, 12]  # for example

        
        self.reset()

    def reset(self):
        # Ensuring the car starts on a street
        row = random.choice(self.street_rows) * GRID_SIZE
        col = random.choice(self.street_cols) * GRID_SIZE
        self.car_pos = [col, row]

        row_goal = random.choice(self.street_rows) * GRID_SIZE
        col_goal = random.choice(self.street_cols) * GRID_SIZE
        self.goal_pos = [col_goal, row_goal]

        while self.car_pos == self.goal_pos:
            row_goal = random.choice(self.street_rows) * GRID_SIZE
            col_goal = random.choice(self.street_cols) * GRID_SIZE
            self.goal_pos = [col_goal, row_goal]

        return self.car_pos


    def step(self, action):
        def euclidean_distance(pos1, pos2):
            return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

        move = self.directions[action]
        next_pos = [self.car_pos[0] + move[0], self.car_pos[1] + move[1]]

        next_row, next_col = next_pos[1] // GRID_SIZE, next_pos[0] // GRID_SIZE
        if (next_row in self.street_rows or next_col in self.street_cols) and 0 <= next_pos[0] < SCREEN_WIDTH and 0 <= next_pos[1] < SCREEN_HEIGHT:
            prev_distance = euclidean_distance(self.car_pos, self.goal_pos)
            self.car_pos = next_pos
            new_distance = euclidean_distance(self.car_pos, self.goal_pos)

            if self.car_pos == self.goal_pos:
                return self.car_pos, 1000, True
            else:
                reward = -1
                return self.car_pos, reward, False
        else:
            # Penalty for invalid move
            return self.car_pos, -2, False


    def draw(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.screen.fill(GREEN)
        
        # Drawing the grey streets
        for row in self.street_rows:
            pygame.draw.rect(self.screen, (150, 150, 150), (0, GRID_SIZE*row, SCREEN_WIDTH, GRID_SIZE))
        
        for col in self.street_cols:
            pygame.draw.rect(self.screen, (150, 150, 150), (GRID_SIZE*col, 0, GRID_SIZE, SCREEN_HEIGHT))
        

        pygame.draw.rect(self.screen, BLUE, (self.car_pos[0], self.car_pos[1], GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(self.screen, RED, (self.goal_pos[0], self.goal_pos[1], GRID_SIZE, GRID_SIZE))
        pygame.display.flip()



class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=100000)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 1e-3
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-4

        self.model = QNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.rewards = [] # Store all the total rewards
        self.steps = [] # Store all the step values
        self.episodes = [] # Store the episodes
        self.avg_episodes = [1] # Store every 10 episodes for average rewards
        self.avg_rewards = [0.0] # Store all the average values

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward + self.gamma * torch.max(self.model(next_state)).item() * (not done)
            target = torch.tensor([target], dtype=torch.float32)
            current = self.model(state)[action].view(1)
            loss = self.criterion(current, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def train(self, env, episodes):
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            while not done:
                action = self.act(state)
                next_state, reward, done = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                env.draw()
                self.replay(32)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


            self.episodes.append(episode+1) # store the episode values
            self.rewards.append(total_reward) # store the total reward
            self.steps.append(steps) # store the step value

            print(f'Episode: {episode+1} | Total Reward: {total_reward} | Epsilon: {self.epsilon}') # print out the variables

            if (episode+1) % 10 == 0: # for each 10 episodes
                #self.save_model(os.path.join('models', 'rl_model' + str(episode+1))) # save the model
                self.avg_episodes.append(episode+1) # store the average rewards
                avg_reward = np.mean(self.rewards[-10:]) # get the mean reward value
                self.avg_rewards.append(avg_reward) # store the average reward
                self.plot_training_progress(episode+1) # plot the training progress
                print(f'Average Reward: {avg_reward}') # print the average reward
        
    def plot_training_progress(self, episode): # Function to plot training progress 
        fig, axs = plt.subplots(3)
        
        # Plotting Total Episode Rewards
        axs[0].plot(self.episodes, self.rewards)
        axs[0].set_title("Total Rewards")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Reward")

        # Plotting Average Rewards
        axs[1].plot(self.avg_episodes, self.avg_rewards)
        axs[1].set_title("Average Rewards")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Reward")
        
        # Plotting steps 
        axs[2].plot(self.episodes, self.steps)
        axs[2].set_title("Steps per episode")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Steps")
        
        # Saving Figures
        plt.tight_layout()
        name = 'training_curve' + str(episode) +'.png'
        plt.savefig(name)

