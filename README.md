# Snake_game_Reinforcement_AI

A snake game which teaches itself how to increase surviving time and get maximum score by using Linear_QNet (Reinforcement Learning with Deep Learning)

## Components:

### Agent:
    - Snake Game (game)
    - model: DQN (deep q learning: deep learning with reinforcement learning)

### Linear_QNet (DQN model):
    - model.predict(state)
        ->action
        
### Snake Game (game):
    - play_step(action) // takess the action predicted by NN
        -> reward, game_over, score // gives these outputs

### Training of Agent:
    - state = get_state(game)  // gets the state with game env as input
    - action = get_move(state):  // gets the move with state as input
        - model.predict()   // getting action from prediction of model
    - reward, game_over, score = game.play_step(action) // action is input, gives reward, game_over and score
    - new_state = get_state(game) // collecting the new state
    - remember // remember the rewards and losses
    - model.train() // a trained model