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
import pygame 
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
        self.mem_cntr += 1
        self.mem_cntr = self.mem_cntr % self.mem_size
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
        self.action_space = [-1,0,1]

        # Define observation space (distance, relative_speed)
        self.observation_space = spaces.Box(low=np.array([0, -100]), high=np.array([100, 10]))

        self.reset()

                # pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)

        # car positions and sizes
        self.car_width, self.car_height = 70, 40
        self.leading_car_x = 400
        self.leading_car_y = 100
        self.main_car_x = 400
        self.main_car_y = 500

    def reset(self):
        # Initialize distance and relative speed
        self.distance = np.random.uniform(-30, 70)
        self.relative_speed = np.random.uniform(-5, 5)
        return np.array([self.distance, self.relative_speed])

    def step(self, action):
        # Previous distance
        prev_distance = self.distance

        acceleration = action - 1  # Map 0, 1, 2 to -1, 0, 1

        # Update state
        self.relative_speed += acceleration
        self.distance += self.relative_speed

        # Reward calculation
        if self.distance == 100: # Assumes that 0 is the goal distance
            reward = 100
        elif self.distance < prev_distance:  # Car got closer to the goal
            reward = 10
        else:  # Car either stayed in the same position or moved away
            reward = -1

        # Check for terminal conditions, such as collisions or reaching a max episode length
        done = False
        if self.distance <= 0: # The car has arrived at or passed the goal
            done = True

        return np.array([self.distance, self.relative_speed]), reward, done


    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Fill screen
        self.screen.fill((255, 255, 255))
        
        # Draw cars
        pygame.draw.rect(self.screen, (0, 0, 255), (self.leading_car_x, self.leading_car_y, self.car_width, self.car_height))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.main_car_x, self.main_car_y - self.distance, self.car_width, self.car_height))

        # Display state info
        state_text = self.font.render(f"Distance: {self.distance:.2f}, Relative Speed: {self.relative_speed:.2f}", True, (0, 0, 0))
        self.screen.blit(state_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
 


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


class Agent:
    def __init__(self,gamma,epsilon,epsilon_min,epsilon_dec,lr,mem_size=100000000):
        self.env = AutonomousVehicleEnv() 
        self.input_dims = self.env.observation_space.shape[0]
        self.hidden_dims = 128
        self.output_dims = len(self.env.action_space)
        self.model = DeepQNetwork(self.input_dims,self.hidden_dims,self.output_dims)
        self.gamma = gamma 
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.lr = lr 
        self.loss_fn = nn.MSELoss() 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) # Create an optimizer
        self.memory = PrioritizedReplayBuffer(mem_size, self.input_dims, self.output_dims)
        self.scores = [] # Store all the total rewards
        self.steps = [] # Store all the step values
        self.episodes = [] # Store the episodes
        self.avg_episodes = [1] # Store every 10 episodes for average rewards
        self.avg_rewards = [0.0] # Store all the average values
        self.beta = 0.4 # Store the initial beta value
        self.beta_end = 1.0 # Store the final beta value
        self.beta_increment = 0.0001 # Slowly increment the beta value 
        self.batch_size = 64 # Assign the batch size

    def update_epsilon(self): # Update the epsilon value
        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min) 

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        print('louis')
        # Step 1: Sample from the prioritized replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample_buffer(self.batch_size, beta)

        states = torch.tensor(states, device=self.model.device).float()
        actions = torch.tensor(actions, device=self.model.device).long()
        rewards = torch.tensor(rewards, device=self.model.device).float()
        next_states = torch.tensor(next_states, device=self.model.device).float()
        dones = torch.tensor(dones, device=self.model.device).bool()
        weights = torch.tensor(weights, device=self.model.device).float()

        # Step 2: Compute Q-values
        q_eval = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q_next = self.target(next_states).detach()
        max_actions = q_next.argmax(dim=1)
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next.gather(1, max_actions.unsqueeze(-1)).squeeze(-1)

        # Step 3: Calculate TD errors
        errors = (q_target - q_eval).detach().cpu().numpy()
        self.memory.set_priorities(indices, errors)

        # Step 4 & 5: Adjust Q-value updates with importance-sampling weights and calculate the loss
        loss = self.loss(q_eval, q_target)
        weighted_loss = (weights * loss).mean()

        # Step 6: Perform gradient descent
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        # Step 7: Update beta
        self.beta = min(self.beta + self.beta_increment, self.beta_end)

    def select_action(self, state): # Select action
        if np.random.rand() < self.epsilon: # Epsilon-Greedy
            return np.random.choice(self.env.action_space)
        else:
            q_values = self.model(torch.tensor(state, dtype=torch.float32)) # Get the q-values
            action = torch.argmax(q_values[0]).item()
            return action # return the action value


    def train(self, episodes): # Train the agent
        for episode in range(episodes): # Total reward for each episode
            total_reward = 0 # Initialize the total reward
            done = False # Terminal condition is set to False
            state = self.env.reset() # Reset the state 
            avg_reward = 0 # Initialize the average reward
            steps = 0 # Initial the total steps

            while not done: # While not done
                self.env.render()
                action = self.select_action(state) # select an action
                next_state, reward, done = self.env.step(action) # call the step function
                total_reward = total_reward + reward  # accumulate the rewards
                self.learn() # call the learn function
                self.update_epsilon() # update the epsilon value
                state = next_state # assign the next state to the current state
                steps += 1 # increment the steps
                

            self.episodes.append(episode+1) # store the episode values
            self.scores.append(total_reward) # store the total reward
            self.steps.append(steps) # store the step value

            print(f'Episode: {episode+1} | Total Reward: {total_reward} | Epsilon: {self.epsilon}') # print out the variables

            if (episode+1) % 100 == 0: # for each 10 episodes
                self.save_model(os.path.join('models', 'rl_model' + str(episode+1))) # save the model
                self.avg_episodes.append(episode+1) # store the average rewards
                avg_reward = np.mean(self.scores[-10:]) # get the mean reward value
                self.avg_rewards.append(avg_reward) # store the average reward
                self.plot_training_progress(episode+1) # plot the training progress
                print(f'Average Reward: {avg_reward}') # print the average reward


    def save_model(self, file_name): # Function to save the model
        torch.save(self.model.state_dict(), file_name + "_model.pt")

    def load_model(self, file_name): # Function to load the model
        self.model.load_state_dict(torch.load(file_name + "_model.pt"))
        self.model.eval()


    def plot_training_progress(self, episode): # Function to plot training progress 
        fig, axs = plt.subplots(3)
        
        # Plotting Total Episode Rewards
        axs[0].plot(self.episodes, self.scores)
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