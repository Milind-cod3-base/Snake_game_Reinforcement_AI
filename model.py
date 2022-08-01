""" 
    This module is responsible for holding and training of the
    game AI.
"""

# importing libraries
from turtle import forward
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

    # creating a feed forward
    def forward(self, x):
        # using function module and relu as activation function
        x = F.relu(self.linear1(x)) # linear layer with tensor x as input
        # we dont need activation function as its last layer
        x = self.linear2(x) # returning raw values
        
        return x
    
    # helper function to save the model later
    def save(self, file_name = 'model.pth' ):  # a default filename
        # new folder creation
        model_folder_path = './model'

        # create a new folder if doesnt exists already
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path) 

        # assigning the complete location of the file 
        file_name = os.path.join(model_folder_path, file_name)
        # save state dict, and path is file_name
        torch.save(self.state_dict(), file_name) 

# class for training the module
class QTrainer:
    # creating a constructor
    def __init__(self, model, lr, gamma): # lr = learning rate
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # using adam optimiser, optimise the parameters
        self.optimizer = optim.Adam(model.parameters(), lr= self.lr)
        # loss function -> mean square error
        self.criterion = nn.MSELoss()

    # training step needs all the old parameters
    def train_step(self, state, action, reward, next_state, done):
        # converting into a pytorch tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x) shape
        # done / game over is not performed as no tensor needed for it
        

        if len(state.shape) == 1: # means one dimension
            # reshape it into form (1, x) where 1 is no. of batches
            # adding one dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicited Q values wih current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]

            if not done[idx]: # if not game over
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # need this as an item and not tensor
            target[idx][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + y * max( next_predicted Q value) -> only do this if not done, other wise take the whole reward
        # # because we need in the same format, simply clone it to get the three values
        # pred.clone() 
        # preds[argmax(action)] = Q_new  # index of argmax sets new Q value
        self.optimizer.zero_grad()
        loss =  self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

