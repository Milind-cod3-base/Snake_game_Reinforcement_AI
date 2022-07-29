""" This module consists of the same snake game
but this is controlled manually """

# importing important libraries
import pygame
import random

from enum import Enum
from collections import namedtuple

# initialize all imported pygame modules
pygame.init()

# setting up the font
font = pygame.font.Font('arial.ttf', 25)

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
Point = namedtuple('Point','x,y')

# rgb colors
WHITE = (255,255,255)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20


# creating a class for the game
class SnakeGame:
    
    # a constructor with pre-defined window size of the game
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h 
        
        # init display
        
        # set the window size
        self.display = pygame.display.set_mode((self.w, self.h))

        # set the caption
        pygame.display.set_caption('slave_snake')

        # setting the clock with clock class
        # this keeps track of the time
        self.clock = pygame.time.Clock()

        # init a game state
        # using RIGHT from the enum inherited class Direction
        self.direction = Direction.RIGHT

        # putting the head of the snake while init
        # in the middle of the screen
        self.head = Point(self.w/2, self.h/2)

        # creating a list which contains the attributes of 
        # the snake
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        # setting initial score to zero
        self.score = 0
        # food as none in init
        self.food = None 

        # placing the food in the game screen using method
        # using a leading underscore for making it for internal use
        # only
        self._place_food()

    
    # creating a method for placing the food randomly in game display
    # this method is for internal use only
    def _place_food(self):
        
        # setting cordinates which must inside the boundary of display
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE

        # assigning attribute for food position
        self.food = Point(x,y)
        
        # if cordinates of food and cordinates of snakes coincides
        # and it means the snake has eaten the food, new food has to
        # be placed again. 
        # Using recursive style on this method to replace the food
        if self.food in self.snake:
            self._place_food()  