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

# creating a class for the AI handled snake game 
class SnakeGameAI:
    
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

        # calling the reset method here in the constructor
        self.reset()

        

    # setting up a reset method for the ai
    # copied code from init constructor
    # which init the game state and pasted
    def reset(self):
        # init a game state
        # using RIGHT from the enum inherited class Direction
        self.direction = Direction.RIGHT

        # putting the head of the snake while init
        # in the middle of the screen
        self.head = Point(self.w/2, self.h/2)

        # creating a list which contains the attributes of 
        # the snake. Starting with length of 3 blocks.
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

        # setting the frame iteration counter to 0 
        self.frame_iteration = 0

    
    # creating a method for placing the food randomly in game display
    # this method is for internal use only
    def _place_food(self):
        
        # setting cordinates which must inside the boundary of display
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE

        # assigning attribute for food position
        self.food = Point(x,y)
        
        # if cordinates of food and cordinates of snakes coincides
        # and it means the snake has eaten the food, new food has to
        # be placed again. 
        # Using recursive style on this method to replace the food
        if self.food in self.snake:
            self._place_food()  

   
    # creating a method for the snake to take steps
    # this function now takes action parameter which is given by AI
    def play_step(self, action):
        self.frame_iteration +=1

        # step1: collecting user input

        # creating a for loop using pygame.event.get()
        # it registers all events from the user into an event queue
        # pygame has some predefined events in them, eg. QUIT, KEYDOWN
        for event in pygame.event.get():

            # checking if user wishes to quit
            # standard code for pygame
            if event.type == pygame.QUIT:
                # quits the pygame
                pygame.quit()
                #quits the code
                quit()  

        
        # step2: moving action
        
        # this updates the head of the snake
        # also resets the value of self.head attribute
        # it takes input action given by the AI
        self._move(action)
         


        # inserts self.head before the index 0 in the snake list
        # takes the new self.head value and puts it inside the snake
        # and makes its new head
        self.snake.insert(0, self.head) 

        # step3: check if game over

        # reward check, +10 for eat, death is -10
        reward = 0
        # setting game_over variable as False
        game_over = False

        #  if the collision occurs, game must be over 
        # results must be displayed
        # or if the frame keeps on going and snake length doesnt change
        # its jus moving without eating then stop using a custom formula
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = - 10   # collision or longer gameplay without increase
            return reward, game_over, self.score



        # step4: place new food if snake eats or just move
        if self.head == self.food:
            # increase score by one
            self.score +=1
            # after eating place the food again
            self._place_food()
        
        # if snake doesnt eat anything, pop the last part of the snake
        # this will keep the length of the snake same
        else:
            self.snake.pop()
        
        # step5:  update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # step6: return game over and score
        return game_over, self.score, reward
    
    # method for if collision happens -> returns boolean value
    # parameter pt is set to None, no need to put parameter when calling it
    # pt is introduced to detect the danger or states 
    """ 11 states are: [ danger straight, danger right, danger left,
                            direction left, direction right, 
                            direction up, direction down,
                            food left, food right,
                            food up, food down]"""
    
    def _is_collision(self, pt= None):

        if pt is None: # checking if the parameter is set to None
            # then assign the value self.head to the parameter
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        # hits itself
        if pt in self.snake[1:]:  
            # if the head is in the snake and its 3rd or later element
            # then it has hit itself

            return True

        return False
    
    # update the UI
    def _update_ui(self):
        
        # creating the background black
        self.display.fill(BLACK)

        # looping through the list of snake's length
        # and colouring the whole snake blue
        # using two colours and overlapping them
        # to give an illusion of amazing design
        for pt in self.snake:  # iterating using pt below

            # laying base color for foundation
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE ))
            # laying another layer of color on top of base color
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12 ))

        # this is to refresh the food block    
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # refreshing the UI

        # set the text with font
        text = font.render("Score: " + str(self.score), True, WHITE)

        # set the position for HUD
        self.display.blit(text, [0,0])

        # updating the entire screen by using flip
        #: if only portion of screen has to be updated, use update method instead of flip
        pygame.display.flip()

        
    # method to move the snake further
    def _move(self, action):
        x = self.head.x # calling attribute head of snake's x cordinate
        y = self.head.y

        if direction == Direction.RIGHT:
            # updating x
            x += BLOCK_SIZE
        
        elif direction == Direction.LEFT:
            # updating x
            x -= BLOCK_SIZE

        elif direction == Direction.DOWN:
            # y updating
            y += BLOCK_SIZE

        elif direction == Direction.UP:
            # y updating
            y -= BLOCK_SIZE

        # reseting self.head values to the new cordinates
        self.head = Point(x,y)