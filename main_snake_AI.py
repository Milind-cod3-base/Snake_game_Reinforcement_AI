# importing important libraries
import pygame
import random

from enum import Enum
from collections import namedtuple

# initialize all imported pygame modules
pygame.init()

# setting up the font
font = pygame.font.Font('arial.ttf', 25)

# functions needed are as follows:
# reset
# reward
# play(action) -> direction
# is_collision