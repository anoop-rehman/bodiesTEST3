import torch
import torch.nn as nn
import numpy as np
from collections import deque
import tictactoe 
from tqdm import tqdm
import random

# Constants
DIMENSION = 3
EMPTY_TABLE = np.zeros((DIMENSION, DIMENSION))

# 1. Game Logic
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

# 2. Transformer Model
class TicTacToeTransformerSeq(nn.Module):
    def __init__(self):
        super(TicTacToeTransformerSeq, self).__init__()
        self.embedding = nn.Embedding(3, 64)  
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 9)  
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, board_dim, board_dim)
        x = self.embedding(x)  
        x = x.view(x.size(0), x.size(1), -1, x.size(-1))  # Adjusting shape to (batch_size, sequence_length, board_dim * board_dim, emb_dim)
        x = x.mean(dim=2)  # Mean or max pooling can be used here
        x = self.transformer(x)
        x = x.mean(dim=1)  # Aggregating over sequence length
        x = self.fc(x)
        return x


# 3. Preprocessing
def preprocess_experience(experiences):
    boards = [exp[0] for exp in experiences]
    boards_tensor = torch.tensor(boards, dtype=torch.long)
    return boards_tensor


def generate_sequences(buffer, sequence_length=5):
    sequential_buffer = []
    for i in range(len(buffer) - sequence_length + 1):
        sequence = buffer[i:i+sequence_length]
        sequential_buffer.append(sequence)
    return sequential_buffer

# Data and Model
replay_buffer = deque(maxlen=10000)  # Replay Buffer
generate_random_games(10000, replay_buffer)

model = TicTacToeTransformerSeq()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 4. Training
num_epochs = 100
batch_size = 32

for epoch in tqdm(range(num_epochs), desc="Training"):
    for _ in range(10):
        experiences = random.sample(replay_buffer, batch_size)  # Sample experiences from the replay buffer
        inputs = preprocess_experience(experiences)
        
        actions = torch.tensor([exp[1][0]*DIMENSION + exp[1][1] for exp in experiences], dtype=torch.long)
        rewards = torch.tensor([exp[2] for exp in experiences], dtype=torch.float)
        
        logits = model(inputs)
        
        loss = -torch.mean(torch.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze() * rewards)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

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




print("game1")
play_game(model)
print("game2")
play_game(model)
print("game3")
play_game(model)

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

print("Game 1: AI (Player 1) vs Random (Player 2)")
play_game_with_random_agent(model, 1)
print("\nGame 2: AI (Player 1) vs Random (Player 2)")
play_game_with_random_agent(model, 1)
print("\nGame 3: Random (Player 1) vs AI (Player 2)")
play_game_with_random_agent(model, 2)
print("\nGame 4: Random (Player 1) vs AI (Player 2)")
play_game_with_random_agent(model, 2)
