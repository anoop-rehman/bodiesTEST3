import torch.nn as nn
import torch

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