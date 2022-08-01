""" 
    This module is responsible for holding and training of the
    game AI.
"""

# importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os # to save the model


# inheriting torch module and holding DQN
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size ):
        super().__init__() # init from parent class
        # creating two layer neural network
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size) 