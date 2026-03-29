import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        return self.output(attention_output)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return x + self.activation(self.norm(self.fc(x)))

class SmartStarTrackerDenoiser(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, num_heads=4):
        super().__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        self.encoder = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads)
        
        self.attention_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        self.decoder = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.2
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(2)
        ])
        
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size)
        )
        
        self.output_projection = nn.Linear(input_size, input_size)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        x = self.input_projection(x)
        
        encoder_output, _ = self.encoder(x)
        
        attended = self.attention(encoder_output)
        
        x = self.attention_projection(attended)
        
        decoder_output, _ = self.decoder(x)
        
        for block in self.residual_blocks:
            decoder_output = block(decoder_output)
        
        output = self.noise_predictor(decoder_output)
        
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = SmartStarTrackerDenoiser(input_size=7)
    print(f"Model parameters: {count_parameters(model):,}")
    
    test_input = torch.randn(4, 100, 7)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
