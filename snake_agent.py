import torch
import random
import numpy as np
from collections import deque 
# importing classes and object from self made module
from snake_AI_env import SnakeGameAI, Direction, Point

# setting parameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# this class stores the game and the model
class Agent:
    
    # creating constructor
    def __init__(self):
        pass
    
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
