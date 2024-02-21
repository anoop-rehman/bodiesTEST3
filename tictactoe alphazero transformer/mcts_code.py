from constants import EMPTY_TABLE, DIMENSION
from copy import deepcopy
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class Board:
    def row_checker(self, state):
        for row in range(DIMENSION):
            total_multiplication = 1
            for column in range(DIMENSION):
                total_multiplication *= state[row][column]
            if total_multiplication == 2**DIMENSION: # Row full of twos
                return 2
            if total_multiplication == 1**DIMENSION: # Row full of ones
                return 1
        return -1
    
    def column_checker(self, state):
        for column in range(DIMENSION):
            total_multiplication = 1
            for row in range(DIMENSION):
                total_multiplication *= state[row][column]
            if total_multiplication == 2**DIMENSION: # Row full of twos
                return 2
            if total_multiplication == 1**DIMENSION: # Row full of ones
                return 1             
        return -1

    def diagonal_checker(self, state):
        for corner in range(1, 3):
            total_multiplication = 1
            if corner == 1:
                for i in range(DIMENSION):
                    total_multiplication *= state[i][i]
                if total_multiplication == 2**DIMENSION: # Row full of twos
                    return 2
                if total_multiplication == 1**DIMENSION: # Row full of ones
                    return 1
            if corner == 2:
                row = 0
                total_multiplication = 1
                for column in range(DIMENSION - 1, -1, -1):
                    total_multiplication *= state[row][column]
                    row += 1
                
                if total_multiplication == 2**DIMENSION: # Row full of twos
                    return 2
                if total_multiplication == 1**DIMENSION: # Row full of ones
                    return 1
        return -1
    
    def winning_state(self, state):
        if self.row_checker(state) != -1 or self.column_checker(state) != -1 or self.diagonal_checker(state) != -1:
            return True
        return False

    def full_board(self, state):
        zeroCounter = 0
        for row in range(DIMENSION):
            for column in range(DIMENSION):
                if state[row][column] == 0:
                    zeroCounter += 1  
        if zeroCounter == 0:
            return True
        return False
    
    def who_wins(self, state):
        if self.row_checker(state) == 1 or self.column_checker(state) == 1 or self.diagonal_checker(state) == 1:
            return 1
        if self.row_checker(state) == 2 or self.column_checker(state) == 2 or self.diagonal_checker(state) == 2:
            return -1
        if self.full_board((state)) == True:
            return 0
        return 2

    
    def who_actually_wins(self, state):
        if self.row_checker(state) == 1 or self.column_checker(state) == 1 or self.diagonal_checker(state) == 1:
            return 1
        if self.row_checker(state) == 2 or self.column_checker(state) == 2 or self.diagonal_checker(state) == 2:
            return 2
        
        return 0

    def print_formatting(self, state):
        for i in range(len(state)):
            print(state[i])

class ValueNet(torch.nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(DIMENSION * DIMENSION, 256)  # Assuming your state is flattened to DIMENSION * DIMENSION
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # outputs values between -1 and 1


class Node():
    def __init__(self, parent, state, move=None):
        self.parent = parent
        self.state = state
        self.player = None
        self.children = []  # Initialize to an empty list
        self.move = move

        self.value = 0
        self.visits = 0
    
    def choose_node(self, exploration_constant):
        best_ucb = float('-inf')
        best_node = None

        for child in self.children:
            if child.visits > 0:
                ucb = child.value/child.visits + exploration_constant * math.sqrt((math.log(self.visits))/child.visits)
            else:
                ucb = float('inf')

            if ucb > best_ucb:
                best_ucb = ucb
                best_node = child

        return best_node
    
    def create_children(self):  
        list_of_children = []

        for row in range(DIMENSION):
            for column in range(DIMENSION):
                if self.state[row][column] == 0:
                    temporary_state = deepcopy(self.state)
                    temporary_state[row][column] = 3 - self.player

                    move = (row, column)
                    temporary_node = Node(self, deepcopy(temporary_state), move)
                    temporary_node.player = 3 - self.player

                    list_of_children.append(temporary_node)
        
        self.children = list_of_children

value_net = ValueNet()

class MCTS:
    def __init__(self, model):
        self.board = Board()
        self.search_length = 100
        self.model = model
        self.value_net = ValueNet()
        optimizer = optim.Adam(list(model.parameters()) + list(self.value_net.parameters()), lr=0.01)
        self.model.optimizer = optimizer
        self.training_data = []
        self.value_data = []

    def search(self, state, player):
        original_state = deepcopy(state)
        starting_node = Node(None, state)
        starting_node.player = 3 - player
        starting_node.visits = 1
        starting_node.create_children()
        self.player_here = player

        if not starting_node.children:
            starting_node.create_children()

        for i in range(self.search_length):
            policy_values = self.get_policy_values(state)
            new_node = self.selection(starting_node, policy_values)
            
            value_estimate = self.simulation(new_node)
            self.backpropogation(new_node, value_estimate)
            
            current_state = new_node.state
            mcts_policy = self.get_mcts_policy(new_node)  # get MCTS policy for the current state
            self.training_data.append((current_state, mcts_policy, None))


        best_action_value = float("-inf")
        best_child = None
        for child in starting_node.children:
            if child.visits > 0:
                value = child.value / child.visits
                if value > best_action_value:
                    best_child = child
                    best_action_value = value
                
        return best_child  # Return the best child node


    def selection(self, node, policy_values=None):
        while self.board.who_wins(node.state) == 2:
            if not node.children:
                if node.visits == 0:
                    return node

                node.create_children()
                # After attempting to create children, if there are still no children
                # return the current node itself.
                if not node.children:
                    return node
            else:
                if policy_values is not None:
                    node = self.choose_node_with_policy(node, policy_values)
                else:
                    node = node.choose_node(2)  # using UCB without policy

        return node



    def simulation(self, node):
        # Convert the state to tensor and get the value estimate
        state_tensor = torch.tensor(node.state.flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value_estimate = self.value_net(state_tensor)
        return value_estimate.item()

    def backpropogation(self, node, value_estimate):
        while node:
            node.visits += 1
            node.value += value_estimate
            value_estimate = -value_estimate  # Switch value estimate for the opponent
            node = node.parent

    def choose_node_with_policy(self, node, policy_values):
        best_score = float('-inf')
        best_node = None
        for child, policy_value in zip(node.children, policy_values):
            if child.visits > 0:
                ucb = child.value / child.visits + 2 * math.sqrt((math.log(node.visits)) / child.visits)
                combined_score = ucb * policy_value
            else:
                combined_score = policy_value  # If unvisited, rely solely on policy network
            if combined_score > best_score:
                best_score = combined_score
                best_node = child
        if best_node is None:  # If no node was chosen, choose a random child
            best_node = random.choice(node.children)
        return best_node


    def get_policy_values(self, state):
        state_tensor = torch.tensor(state[np.newaxis, :], dtype=torch.long)
        if next(self.model.parameters()).is_cuda:
            state_tensor = state_tensor.cuda()
        policy_distribution = self.model(state_tensor)
        return F.softmax(policy_distribution, dim=-1).detach().cpu().numpy().flatten()

    def get_mcts_policy(self, starting_node):
        total_visits = sum(child.visits for child in starting_node.children)
        policy = [0] * (DIMENSION * DIMENSION)
        for child in starting_node.children:
            index = child.move[0] * DIMENSION + child.move[1]
            policy[index] = child.visits / total_visits
        return policy

    def compute_policy_loss(self, predicted_policy, mcts_policy):
        log_probs = F.log_softmax(predicted_policy, dim=-1)
        return -torch.sum(mcts_policy * log_probs)
    
    def train_networks(self, num_epochs):
        self.training_data = [sample for sample in self.training_data if sample[2] is not None]

        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            for state, mcts_policy, true_value in self.training_data:
                self.model.optimizer.zero_grad()
                
                # For policy
                state_tensor = torch.tensor(state.flatten(), dtype=torch.long).unsqueeze(0)
                mcts_policy_tensor = torch.tensor(mcts_policy, dtype=torch.float32).unsqueeze(0)
                predicted_policy = self.model(state_tensor)
                policy_loss = self.compute_policy_loss(predicted_policy, mcts_policy_tensor)
                
                # For value
                state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
                true_value_tensor = torch.tensor(true_value, dtype=torch.float32).unsqueeze(0)
                predicted_value = self.value_net(state_tensor)
                value_loss = F.mse_loss(predicted_value, true_value_tensor)
                
                loss = policy_loss + value_loss
                loss.backward()
                self.model.optimizer.step()
                
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(self.training_data)}")


    def self_play(self, num_games=100):
        for _ in tqdm(range(num_games)):
            state = np.zeros((DIMENSION, DIMENSION))
            game_history = []
            player = 1

            while self.board.who_wins(state) == 2:
                best_child_node = self.search(state, player)
                mcts_policy = self.get_mcts_policy(best_child_node)  # Pass node instead of state
                game_history.append((state, mcts_policy, None))  # 'None' is a placeholder for the reward.
                    
                state = best_child_node.state  # Extract the state from the best child node
                player = 3 - player  # Switch player

                    
            winner = self.board.who_actually_wins(state)
            # Assign rewards based on the game outcome
            for index, (s, p, r) in enumerate(game_history):
                if winner == 0:  # Draw
                    reward = 0
                else:
                    reward = winner if index % 2 == 0 else -winner
                game_history[index] = (s, p, reward)

            self.training_data += game_history



board = Board()

def random_agent(boardState):
    possible_moves = np.where(boardState == 0)
    num_possible_moves = possible_moves[0].shape[0]
    if num_possible_moves == 0:
        return None
    move_index = np.random.choice(num_possible_moves)
    return (possible_moves[0][move_index], possible_moves[1][move_index])

def play_mcts_vs_mcts(model, game_count=2):
    for _ in range(game_count):
        state = np.zeros((DIMENSION, DIMENSION))
        index = 0
        mcts = MCTS(model)

        while board.who_wins(state) == 2:
            if index % 2 == 0:
                player = 1
            if index % 2 == 1:
                player = 2
            index += 1

            best_child = mcts.search(state, player)
            state = best_child.state  # Extract the state from the best child node

            print("--")
            print(state)
            
def play_mcts_vs_random(model, game_count, starting_index):
    for _ in range(game_count):
        state = np.zeros((DIMENSION, DIMENSION))
        index = starting_index
        mcts = MCTS(model)

        while board.who_wins(state) == 2:
            if index % 2 == 0:  # MCTS turn
                player = index % 2 + 1
                best_child = mcts.search(state, player)
                state = best_child.state  # Extract the state from the best child node

            else:  # Random Agent turn
                player = index % 2 + 1
                move = random_agent(state)
                if move:
                    state[move[0]][move[1]] = player

            print("--")
            print(state)
            index += 1  # Increment index to alternate turns