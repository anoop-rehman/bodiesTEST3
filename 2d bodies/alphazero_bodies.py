import numpy as np
import torch
import torch.nn as nn
from soccer_env import SoccerEnv
from copy import deepcopy
import math
import time

GRID_WIDTH = 9
GRID_HEIGHT = 7
CELL_SIZE = 40

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)
    
# Value Network
class ValueNet(nn.Module):
    def __init__(self, embedding_dim=64, nhead=2, num_layers=2):
        super(ValueNet, self).__init__()
        self.embedding = nn.Embedding(64, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(embedding_dim * 6, 128)
        self.fc2 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        value = self.tanh(self.fc2(x))
        return value.squeeze(-1)

class BallTransformerSeq(nn.Module):
    def __init__(self, embedding_dim=64, nhead=2, num_layers=2):
        super(BallTransformerSeq, self).__init__()
        self.embedding = nn.Embedding(64, embedding_dim)  # 63 grid positions + ball possession
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim * 6, 63)  # 63 possible positions on the board (7x9)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Swap batch and sequence dimensions
        x = self.transformer(x)
        x = x.transpose(0, 1)  # Swap back to [batch, seq, feature]
        x = x.reshape(x.size(0), -1)  # Flatten all features
        x = self.fc(x)
        return x
    
class PlayersTransformerSeq(nn.Module):
    def __init__(self, embedding_dim=64, nhead=2, num_layers=2):
        super(PlayersTransformerSeq, self).__init__()
        self.embedding = nn.Embedding(64, embedding_dim)  # 63 grid positions + ball possession
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim * 6, 63)  # 63 possible positions on the board (7x9)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Swap batch and sequence dimensions
        x = self.transformer(x)
        x = x.transpose(0, 1)  # Swap back to [batch, seq, feature]
        x = x.reshape(x.size(0), -1)  # Flatten all features
        x = self.fc(x)
        return x

def preprocess_board_state_sequence(states):
    state_sequence = []
    for state in states:
        players_flat_positions = []
        for player, pos in state['player_positions'].items():
            players_flat_positions.append(pos[0] * GRID_HEIGHT + pos[1])
        ball_x = min(int(state['ball_position'][0] // CELL_SIZE), GRID_WIDTH - 1)
        ball_y = min(int(state['ball_position'][1] // CELL_SIZE), GRID_HEIGHT - 1)
        ball_flat_position = ball_x * GRID_HEIGHT + ball_y
        ball_possession = 0 if state['ball_possession'] is None else 1
        state_representation = players_flat_positions + [ball_flat_position, ball_possession]
        state_sequence.append(state_representation)
    return torch.tensor([state_sequence], dtype=torch.long).squeeze(0)

def action_to_index(action_tuple):
    action_mapping = {'MOVE_LEFT': 0, 'MOVE_RIGHT': 1, 'MOVE_UP': 2, 'MOVE_DOWN': 3}
    index = 0
    for i, action in enumerate(action_tuple):
        index += action_mapping[action] * (4 ** i)
    return index % 63


class Node():
    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.player = None
        self.children = None
        self.value = 0
        self.visits = 0
        self.action = None  # The move which led to this node

    def create_children(self):
        list_of_children = []
        actions = ['MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN']
        state_buffer = [self.state for _ in range(6)]
        preprocessed_state = preprocess_board_state_sequence(state_buffer)
        policy_logits = model.forward(preprocessed_state)

        for action_1 in actions:
            for action_2 in actions:
                for action_3 in actions:
                    for action_4 in actions:
                        state = deepcopy(self.state)
                        state = self.check_validity(action_1, state, "A1")
                        state = self.check_validity(action_2, state, "A2")
                        state = self.check_validity(action_3, state, "B1")
                        state = self.check_validity(action_4, state, "B2")
                        temporary_node = Node(self, deepcopy(state))
                        temporary_node.player = self.player
                        temporary_node.action = (action_1, action_2, action_3, action_4)
                        list_of_children.append(temporary_node)

        # Handle ball shooting logic
        if self.state['ball_possession'] == self.player:
            heatmap = ball_model.forward(preprocessed_state).squeeze().detach().numpy()
            
            # Potential ball directions
            ball_directions = ['SHOOT_UP', 'SHOOT_DOWN', 'SHOOT_LEFT', 'SHOOT_RIGHT']
            
            for direction in ball_directions:
                shoot_state = deepcopy(self.state)
                
                if direction == 'SHOOT_UP':
                    shoot_state['ball_position'] = (shoot_state['ball_position'][0], shoot_state['ball_position'][1] - 1)
                elif direction == 'SHOOT_DOWN':
                    shoot_state['ball_position'] = (shoot_state['ball_position'][0], shoot_state['ball_position'][1] + 1)
                elif direction == 'SHOOT_LEFT':
                    shoot_state['ball_position'] = (shoot_state['ball_position'][0] - 1, shoot_state['ball_position'][1])
                elif direction == 'SHOOT_RIGHT':
                    shoot_state['ball_position'] = (shoot_state['ball_position'][0] + 1, shoot_state['ball_position'][1])
                
                # Append node to children list
                shoot_node = Node(self, shoot_state)
                shoot_node.player = self.player
                shoot_node.action = direction
                list_of_children.append(shoot_node)

        self.children = list_of_children


    def check_validity(self, action, state, player):
        if action == 'MOVE_LEFT' and state['player_positions'][player][0] > 0:
            state['player_positions'][player][0] -= 1
        elif action == 'MOVE_RIGHT' and state['player_positions'][player][0] < GRID_WIDTH - 1:
            state['player_positions'][player][0] += 1
        elif action == 'MOVE_UP' and state['player_positions'][player][1] > 0:
            state['player_positions'][player][1] -= 1
        elif action == 'MOVE_DOWN' and state['player_positions'][player][1] < GRID_HEIGHT - 1:
            state['player_positions'][player][1] += 1
        return state

class MCTS():
    def __init__(self):
        self.search_length = 10

    def search(self, state):
        starting_node = Node(None, state)
        starting_node.player = None
        starting_node.create_children()
        for i in range(self.search_length):
            new_node = self.selection(starting_node)
            score = self.simulation(new_node)
            self.backpropagation(new_node, score)
        
        best_action_value = float("-inf")
        best_child = None
        for child in starting_node.children:
            value = child.value / (child.visits + 1)
            if value > best_action_value:
                best_child = child
                best_action_value = value
        return best_child.state

    def selection(self, node):
        while node.state['scores']['A'] == 0 and node.state['scores']['B'] == 0:
            if not node.children:
                if node.visits == 0:
                    return node
                node.create_children()
                return node.children[0]
            else:
                node = self.choose_node(node)
        return node

    def simulation(self, node):
        current_state = deepcopy(node.state)
        steps = 0
        max_steps = 500
        while not env.is_terminal(current_state) and steps < max_steps:
            actions = {}
            players = ['A1', 'A2', 'B1', 'B2']
            state_buffer = [current_state for _ in range(6)]
            preprocessed_state = preprocess_board_state_sequence(state_buffer)
            model_scores = model.forward(preprocessed_state)
            policy_scores = model_scores.squeeze().detach().numpy()
            last_sequence_scores = policy_scores[-1]
            possible_actions = ['MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN']
            for idx, player in enumerate(players):
                player_scores = last_sequence_scores[idx*4: (idx+1)*4]
                best_action_idx = np.argmax(player_scores)
                actions[player] = possible_actions[best_action_idx]
            current_state = env.apply_actions(current_state, actions)
            steps += 1
        value_net_output = value_net(preprocessed_state)
        reward = value_net_output[-1].item()
        return reward

    def backpropagation(self, node, score):
        while node:
            node.visits += 1
            node.value += score
            node = node.parent

    def choose_node(self, node, exploration_constant=1.0):
        current_state = node.state
        state_buffer = [current_state]
        temp_node = node
        while len(state_buffer) < 6 and temp_node.parent:
            state_buffer.insert(0, temp_node.parent.state)
            temp_node = temp_node.parent
        while len(state_buffer) < 6:
            state_buffer.insert(0, current_state)
        preprocessed_state = preprocess_board_state_sequence(state_buffer)
        model_scores = model.forward(preprocessed_state)
        model_scores = model_scores.squeeze().detach().numpy()
        last_sequence_scores = model_scores[-1]
        best_ucb = float('-inf')
        best_node = None
        for child in node.children:
            bias_index = action_to_index(child.action)
            bias = last_sequence_scores[bias_index]
            if child.visits > 0:
                if node.player in ['B1', 'B2']:
                    bias = -bias
                ucb = child.value / child.visits + exploration_constant * math.sqrt((math.log(node.visits)) / child.visits) + bias
            else:
                ucb = float('inf')
            if ucb > best_ucb:
                best_ucb = ucb
                best_node = child
        return best_node

def state_to_matrix(state):
    matrix = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    player_mapping = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4}
    for player, position in state['player_positions'].items():
        matrix[position[1], position[0]] = player_mapping[player]
    if state['ball_possession']:
        player_pos = state['player_positions'][state['ball_possession']]
        matrix[player_pos[1], player_pos[0]] = 5
    return matrix

value_net = ValueNet()
model = PlayersTransformerSeq()
model.load_state_dict(torch.load('model_players.pth'))
model.eval()

ball_model = BallTransformerSeq()
ball_model.load_state_dict(torch.load('model_ball.pth'))
ball_model.eval()

env = SoccerEnv()
state = env.get_state()

initial_matrix = state_to_matrix(state)
print("Initial state:")
print(initial_matrix)

mcts = MCTS()
new_state = mcts.search(state)

new_matrix = state_to_matrix(new_state)
print("\nState after MCTS:")
print(new_matrix)

# Assuming the rest of your code stays unchanged...

class Trainer:
    def __init__(self, env, mcts, model, ball_model, value_net, buffer, batch_size=64, epochs=5, learning_rate=0.001):
        self.env = env
        self.mcts = mcts
        self.model = model
        self.ball_model = ball_model
        self.value_net = value_net
        self.buffer = buffer
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(list(model.parameters()) + list(ball_model.parameters()) + list(value_net.parameters()), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def get_training_batch(self):
        experiences = self.buffer.sample(self.batch_size)
        states = []
        rewards = []
        for experience in experiences:
            states.append(experience[0])
            rewards.append(experience[4])
        return states, rewards

    def train(self):
        for _ in range(self.epochs):
            states, rewards = self.get_training_batch()
            state_tensors = torch.stack([preprocess_board_state_sequence(state) for state in states])

            model_output = self.model(state_tensors)
            ball_output = self.ball_model(state_tensors)
            value_output = self.value_net(state_tensors)

            # Assuming actions and ball positions are in 'rewards' list, modify accordingly
            model_loss = self.loss_fn(model_output, torch.tensor(rewards[:, 0]))
            ball_loss = self.loss_fn(ball_output, torch.tensor(rewards[:, 1]))
            value_loss = self.loss_fn(value_output, torch.tensor(rewards[:, 2]))

            total_loss = model_loss + ball_loss + value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

# Main execution
BUFFER_SIZE = 1000
BATCH_SIZE = 64
GAMES_PER_TRAINING = 50

env = SoccerEnv()
mcts = MCTS()
buffer = ReplayBuffer(BUFFER_SIZE)

trainer = Trainer(env, mcts, model, ball_model, value_net, buffer, batch_size=BATCH_SIZE)

games_played = 0
while True:
    print(games_played)
    env.play_game(mcts.search, buffer, num_games=1)
    games_played += 1

    if games_played % GAMES_PER_TRAINING == 0:
        trainer.train()
