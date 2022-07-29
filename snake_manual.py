""" This module consists of the same snake game
but this is controlled manually """

# importing important libraries
import pygame
import random

from enum import Enum
from collections import namedtuple

pygame.init()
# setting up the font
font = pygame.font.Font('arial.ttf', 25)

# making class of keys for the Direction 
# inheriting from class Enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP =3
    DOWN = 4