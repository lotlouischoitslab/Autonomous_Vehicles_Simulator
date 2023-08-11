'''
Project: Autonomous Vehicles Simulator using Deep Reinforcement Learning
Author: Louis Sungwoo Cho
Date Created: 8/11/2023
'''

from src.utils import Agent 

def main():
    print('Hello World!')
    louis = Agent(gamma=0.995,epsilon=1.0,epsilon_min=1e-4,epsilon_dec=0.9995,lr=1e-4)
    louis.train(episodes=10)


if __name__ =='__main__':
    main()