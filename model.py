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