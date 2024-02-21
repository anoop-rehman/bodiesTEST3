import torch
import random
from tqdm import tqdm
from model import TicTacToeTransformerSeq, preprocess_experience
from game_logic import generate_random_games
import torch.nn as nn

num_epochs = 100
batch_size = 32
DIMENSION = 3


model = TicTacToeTransformerSeq()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_model(model, replay_buffer, num_epochs=100, batch_size=32):
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
    
    return model
