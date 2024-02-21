import numpy as np
import tictactoe

DIMENSION = 3
EMPTY_TABLE = np.zeros((DIMENSION, DIMENSION))

def make_random_move(boardState, player):
    possible_moves = np.where(boardState == 0)
    num_possible_moves = possible_moves[0].shape[0]
    if num_possible_moves == 0:
        return boardState, None
    move_index = np.random.choice(num_possible_moves)
    move = (possible_moves[0][move_index], possible_moves[1][move_index])
    new_boardState = boardState.copy()
    new_boardState[move] = player
    return new_boardState, move

def generate_random_games(num_games, buffer):
    for _ in range(num_games):
        boardState = EMPTY_TABLE.copy()
        player = 1
        game_history = []
        while True:
            next_boardState, move_made = make_random_move(boardState, player)
            if move_made is None: # Game Over
                winner = tictactoe.whoWins(boardState, DIMENSION)  # Assume a function determining the winner exists
                rewards = assign_rewards(game_history, winner)
                buffer.extend(rewards)
                break
            game_history.append((boardState.copy(), move_made, player))
            boardState = next_boardState
            player = 3 - player  # Switch Player

def assign_rewards(game_history, winner):
    rewards = []
    if winner == 1:
        reward = 1  # Reward for player 1 winning
    elif winner == 2:
        reward = -1  # Penalty for player 1 losing
    else:
        reward = 0.5  # Smaller reward for a draw
    
    for boardState, move, player in reversed(game_history):
        if player == 1:
            rewards.append((boardState, move, reward))
            reward *= -0.9  # Discount future rewards to prioritize winning sooner
        else:  
            reward *= -1  # Invert reward for player 2 actions
            rewards.append((boardState, move, reward))
            reward /= -0.9  # Still discounting, but manage sign for player 2
    
    return rewards
