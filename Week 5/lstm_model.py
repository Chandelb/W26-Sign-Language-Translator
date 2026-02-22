import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from dataloader import get_dataloader

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class Video_LSTM(torch.nn.Module):
    def __init__(self, input_size=42, hidden_size=128, num_layers=2, dropout: float = 0, num_classes=5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size = hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc1 = torch.nn.Linear(hidden_size, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
    
    def forward(self, x, h0=None, c0=None):
        # Expect x: (B, T, input_size) because batch_first=True
        batch_size = x.size(0)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        logits = self.fc1(out[:, -1, :])  # last time step
        #logits = self.dropout(logits)
        logits = self.fc2(logits)
        return logits



