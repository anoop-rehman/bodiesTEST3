from train import train_model
from play import play_game, play_game_with_random_agent
from model import TicTacToeTransformerSeq
from constants import EMPTY_TABLE, DIMENSION
from collections import deque
import numpy as np
import random
from mcts_code import MCTS, Board, play_mcts_vs_mcts, play_mcts_vs_random
from game_logic import generate_random_games


# Initialize Replay Buffer and Model
replay_buffer = deque(maxlen=10000)
generate_random_games(10000, replay_buffer)

# Train Model
model = TicTacToeTransformerSeq()
model = train_model(model, replay_buffer)  # Assuming train_model updates the model in-place

# Create MCTS instance with model
board = Board()
mcts = MCTS(model)

for i in range(2):
    # Generate data from self-play
    print("\nSelf play:")
    mcts.self_play(num_games=100)

    print("\nMCTS learning:")
    # Train networks on the generated data
    mcts.train_networks(num_epochs=10)

# Play Games using Trained Model
print("\nTransformer vs Transformer Games:")
play_game(model)
play_game(model)
play_game(model)

print("\nTransformer vs Random Games:")
play_game_with_random_agent(model, 1)
play_game_with_random_agent(model, 1)

print("\nRandom vs Transformer Games:")
play_game_with_random_agent(model, 2)
play_game_with_random_agent(model, 2)

# Play Games using MCTS hybrid
print("\nMCTS vs MCTS Games:")
play_mcts_vs_mcts(model)

print("\nMCTS vs Random Games:")
play_mcts_vs_random(model, 2, 0)
