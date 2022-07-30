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

# making class of keys for the Direction 
# inheriting from class Enum
# Enum is a class for creating enumerations which are a set of
# symbolic names (members) bound to unique, constant values.
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP =3
    DOWN = 4

#  creating a namedtuple for the x and y cordinates
# basically it names the cordinates as x= and y= 
Point = namedtuple('Point','x, y')

# rgb colors
WHITE = (255,255,255)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
BLACK = (0,0,0)

# variable: size of food block
BLOCK_SIZE = 20

# variable: speed of the clock which refreshes the screen
# indirectly speed of the snake
SPEED = 2