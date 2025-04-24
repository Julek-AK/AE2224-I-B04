import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.bidirectional = bidirectional  # <-- Store bidirectional flag
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, 1)  # Output a single continuous value

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.lstm(packed)
        
        if self.bidirectional:
            # hn shape: (num_layers * 2, batch_size, hidden_size)
            # For 1 layer → use hn[-2] (forward) and hn[-1] (backward)
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last_hidden = hn[-1]

        return self.fc(last_hidden)
