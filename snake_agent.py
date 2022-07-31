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
