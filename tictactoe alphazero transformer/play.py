import torch
import numpy as np
import tictactoe
from model import TicTacToeTransformerSeq
from constants import DIMENSION, EMPTY_TABLE
import random

def play_game(model, dimension=3):
    boardState = tictactoe.emptyTable.copy()
    player = 1

    while not tictactoe.winningState(boardState, dimension) and not tictactoe.fullBoard(boardState, dimension):
        print("\nCurrent Board:")
        tictactoe.printFormmating(boardState)

        with torch.no_grad():
            input_tensor = torch.tensor(boardState, dtype=torch.long).unsqueeze(0)
            move_probabilities = model(input_tensor).squeeze().numpy().reshape((dimension, dimension))
            print("\nMove Heat Map:")
            print(np.round(move_probabilities, 2))  # Showing probabilities in a grid
            possible_moves = np.where(boardState == 0)

            if player == 1:
                print("\nAI Player 1 making move...")
                valid_moves = [move for move in zip(*possible_moves)]
                if valid_moves:
                    # Add some randomness to the decision
                    move_probabilities_with_noise = [move_probabilities[move] + random.uniform(-0.1, 0.1) for move in valid_moves]
                    largest_moves = [move for move in valid_moves if move_probabilities[move] == np.max(move_probabilities_with_noise)]
                    move = random.choice(largest_moves) if largest_moves else random.choice(valid_moves)
                else:
                    print("No valid moves left for AI Player 1. It's a tie!")
                    break
            else:
                print("\nAI Player 2 making move...")
                valid_moves = [move for move in zip(*possible_moves)]
                if valid_moves:
                    # Add some randomness to the decision
                    move_probabilities_with_noise = [move_probabilities[move] + random.uniform(-0.1, 0.1) for move in valid_moves]
                    smallest_moves = [move for move in valid_moves if move_probabilities[move] == np.min(move_probabilities_with_noise)]
                    move = random.choice(smallest_moves) if smallest_moves else random.choice(valid_moves)
                else:
                    print("No valid moves left for AI Player 2. It's a tie!")
                    break

            boardState[move] = player
            player = 3 - player

    winner = tictactoe.whoWins(boardState, dimension)
    print("\nFinal Board:")
    tictactoe.printFormmating(boardState)
    if winner == -1:
        print("\nPlayer 1 (AI) Wins!")
    elif winner == 1:
        print("\nPlayer 2 (AI) Wins!")
    else:
        print("\nIt's a tie!")

def random_agent(boardState):
    possible_moves = np.where(boardState == 0)
    num_possible_moves = possible_moves[0].shape[0]
    if num_possible_moves == 0:
        return None
    move_index = np.random.choice(num_possible_moves)
    return (possible_moves[0][move_index], possible_moves[1][move_index])

def play_game_with_random_agent(model, model_as_player, dimension=3):
    boardState = EMPTY_TABLE.copy()
    player = 1

    while not tictactoe.winningState(boardState, dimension) and not tictactoe.fullBoard(boardState, dimension):
        print("\nCurrent Board:")
        tictactoe.printFormmating(boardState)

        with torch.no_grad():
            if player == model_as_player:
                input_tensor = torch.tensor(boardState, dtype=torch.long).unsqueeze(0)
                move_probabilities = model(input_tensor).squeeze().numpy().reshape((dimension, dimension))
                possible_moves = np.where(boardState == 0)
                valid_moves = [move for move in zip(*possible_moves)]
                if valid_moves:
                    move_probabilities_with_noise = [move_probabilities[move] + random.uniform(-0.1, 0.1) for move in valid_moves]
                    if player == 1:
                        largest_moves = [move for move in valid_moves if move_probabilities[move] == np.max(move_probabilities_with_noise)]
                        move = random.choice(largest_moves) if largest_moves else random.choice(valid_moves)
                    else:
                        smallest_moves = [move for move in valid_moves if move_probabilities[move] == np.min(move_probabilities_with_noise)]
                        move = random.choice(smallest_moves) if smallest_moves else random.choice(valid_moves)
                else:
                    break
            else:
                move = random_agent(boardState)

            if move is None:
                print(f"No valid moves left for Player {player}. It's a tie!")
                break
            boardState[move] = player
            player = 3 - player

    print("\nFinal Board:")
    tictactoe.printFormmating(boardState)