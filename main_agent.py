import torch
import random
import numpy as np
from collections import deque 
# importing classes and object from self made module
from snake_AI_env import SnakeGameAI, Direction, Point

# setting parameters
MAX_MEMORY = 100_000 # underscore is neglected by interpreter
BATCH_SIZE = 1000
LR = 0.001

# this class stores the game and the model
class Agent:
    
    # creating constructor
    def __init__(self):
        self.n_games = 0  # number of games
        self.epsilon = 0 # randomness
        self.gamma = 0 # discount rate for Deep Q learning algo
        # if memory exceeds than the maximum value,
        # then it pops from the left of the deque
        self.memory = deque(maxlen=MAX_MEMORY) 
        # TODO: model, trainer
    
    # gets state out of 11 different variables.
    def get_state(self, game):
        pass
    
    # takes in the state, action, reward, next state and game over 
    def remember(self, state, action, reward, next_state, done): 
        pass

    def train_long_memory(self):
        pass
    
    def train_short_memory(self):
        pass

    # takes in the state and outputs the action
    def get_action(self, state):
        pass


# global function
def train():
    # empty list which keeps track of the scores and plot them
    plot_scores = []
    # average scores
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent() # creaing instance
    game = SnakeGameAI()

    # training loop - it will run until we quit the script
    while True:
        pass


if __name__ == "__main__":
    train()