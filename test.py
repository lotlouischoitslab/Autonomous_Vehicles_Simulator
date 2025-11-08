import numpy as np
import math 
import random 
import pygame
import Environment
 

from Double_Deep_Q import DDQNAgent
from collections import deque
 


TOTAL_GAMETIME = 10000
episodes = 10000
REPLACE_TARGET = 10

game = GameEnv.RacingEnv()
game.fps = 60

GameTime = 0 
GameHistory = []
renderFlag = True

input_dim = 20

ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=0.02, epsilon_end=0.01, epsilon_dec=0.999, replace_target= REPLACE_TARGET, batch_size=64, input_dims=input_dim,fname='ddqn_model.h5')

ddqn_agent.load_model()
ddqn_agent.update_network_parameters()

ddqn_scores = []
eps_history = []


def run(): 
    for e in range(episodes):
        game.reset()
        done = False
        score = 0
        counter = 0
        timer = 0 
        observation_, reward, done = game.step(0)
        observation = np.array(observation_)

        while not done:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    run = False
                    return

             
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            observation = observation_

            timer += 1

            if timer >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

 
run()        