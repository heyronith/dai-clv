import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.data.online_retail_dataset import OnlineRetailDataset, collate_fn

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, lengths):
        # x shape: [batch, seq_len, input_dim]
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_x)
        # Use the last hidden state
        out = self.fc(hidden[-1])
        return out

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        res = self.residual(x)
        out = self.conv(x)
        # Handle padding for causal convolution
        out = out[:, :, :-self.conv.padding[0]] if self.conv.padding[0] > 0 else out
        return self.relu(out + res)

class TCNRegressor(nn.Module):
    def __init__(self, input_dim: int, num_channels: list = [64, 64], kernel_size: int = 2):
        super(TCNRegressor, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, padding))
            
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x, lengths):
        # x shape: [batch, seq_len, input_dim] -> [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # Global pooling over time (or just take the last valid step)
        # For simplicity, we'll take the mean across the active sequence
        pooled_out = []
        for i in range(x.size(0)):
            pooled_out.append(out[i, :, :lengths[i]].mean(dim=-1))
        out = torch.stack(pooled_out)
        return self.fc(out)

def train_deep_model(model, train_loader, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for seq, static, label, lengths in train_loader:
            optimizer.zero_grad()
            preds = model(seq, lengths)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

def get_deep_predictions(model, loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for seq, static, label, lengths in loader:
            preds = model(seq, lengths)
            all_preds.extend(preds.cpu().numpy().flatten())
    return np.array(all_preds)
