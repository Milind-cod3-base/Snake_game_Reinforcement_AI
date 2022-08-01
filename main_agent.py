import torch
import random
import numpy as np
from collections import deque 
# importing classes and object from self made module
from snake_AI_env import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer

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
        # memory is responsible to remember the parameters
        # if memory exceeds than the maximum value,
        # then it pops from the left of the deque
        self.memory = deque(maxlen=MAX_MEMORY) 

        # input size: 11 states, hidden layer size: custom, output_size: 3
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = None  #TODO
        
    
    # gets state out of 11 different variables.
    def get_state(self, game):
        # defining the head of the snake
        head = game.snake[0]
        
        # points around the head of the snake to check presence of a boundary
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # checking the game direction as a boolean if the current direction is one of the below
        # only one of them will be 1
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # establishing 11 states
        state = [

                # Danger straight states -> relative direction hence used 'or'
                (dir_r and game.is_collision(point_r)) or
                (dir_l and game.is_collision(point_l)) or
                (dir_u and game.is_collision(point_u)) or
                (dir_d and game.is_collision(point_d)),

                # Danger right
                (dir_u and game.is_collision(point_r)) or
                (dir_d and game.is_collision(point_l)) or
                (dir_l and game.is_collision(point_u)) or
                (dir_r and game.is_collision(point_d)),

                # Danger left
                (dir_d and game.is_collision(point_r)) or
                (dir_u and game.is_collision(point_l)) or
                (dir_r and game.is_collision(point_u)) or
                (dir_l and game.is_collision(point_d)),

                # Move direction -> one of them is true
                dir_l,
                dir_r,
                dir_u,
                dir_d,

                # food location
                game.food.x < game.head.x, # food left
                game.food.x > game.head.x, # food right
                game.food.y < game.head.y, # food up
                game.food.y > game.head.y # food down  
        ]


        

        # converting above boolean to 1 and 0 using numpy array
        return np.array(state, dtype= int)

    # takes in the state, action, reward, next state and game over and stores in memory
    def remember(self, state, action, reward, next_state, done): 
        # using two parantheses as it may go in as one element
        self.memory.append((state, action, reward, next_state, done)) # popleft if max memory is reached
        

    def train_long_memory(self):
        # we grab one batch(1000 samples in memory)

        # first check if we have more than 1000 samples
        if len(self.memory) > BATCH_SIZE:
            # then only taking batchsize randomly
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns list of tuples
        
        else:
            # then take the whole memory
            mini_sample = self.memory

        # getting multiple states, actions ,rewards etc. together
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        
        self.trainer.train_step(states, actions, rewards, next_states, dones)



    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # takes in the state and outputs the action
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation

        # more games we play, smaller epsilon will get
        self.epsilon = 80 - self.n_games
        # in beginning its all 0, but one of them will be true
        final_move = [0,0,0]

        # as the snake learns, epsilon shortens and chances of random moves decreases
        # at one point epsilon becomes negative, and below condition is no longer valid
        # no more random moves
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0, 2) # 2 is included here
            final_move[move] = 1
        
        else:
            # converting state into tensor
            state0 = torch.tensor(state, dtype=torch.float)
            # getting raw values
            prediction = self.model.predict(state0)
            # converting it using argmax
            move = torch.argmax(prediction).item()
            # gets integer
            final_move[move] = 1
        
        return final_move



# global function
def train():
    # empty list which keeps track of the scores and plot them
    plot_scores = []
    # average scores
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent() # creating instance
    game = SnakeGameAI()

    # training loop - it will run until user quit the script
    while True:
        # get old/ current state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get enw state
        reward, done, score = game.play_step(final_move)

        # get new state, with new game
        state_new = agent.get_state(game)    

        # train short memory of agent -> only for one step
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember -> and store it in to the memory
        agent.remember(state_old, final_move, reward, state_new, done)

        # if game overs, then we need to train long term memory, also called (experience) replay memory
        # it trains again on all previous moves in previous game
        if done:
            game.reset()
            # since the game is done, increase the number of games counter by 1
            agent.n_games +=1
            #train long term memory
            agent.train_long_memory()

            # if snake beats the previous record
            if score > record:
                record = score # set a new record
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
            # TODO: plot

if __name__ == "__main__":
    train()