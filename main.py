'''
Project: Autonomous Vehicles Simulator using Deep Reinforcement Learning
Author: Louis Sungwoo Cho
Date Created: 8/11/2023
'''

from src.utils import DQNAgent, Environment 

def main():
    env = Environment()
    state_size = 2  # car x, car y
    action_size = 4 # left, right, up, down
    agent = DQNAgent(state_size, action_size)
    agent.train(env, 100)


if __name__ =='__main__':
    main()