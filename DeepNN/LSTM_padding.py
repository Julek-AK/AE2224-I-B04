import DataReading as DR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from LSTM_parameters import LSTMRegressor

# Check if CUDA is available, if so, use GPU; otherwise, fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
sequences, lengths = DR.readData4("train")

# Pad to (batch, max_len, features)
targets = torch.tensor([seq[-1][2] for seq in sequences], dtype=torch.float32).unsqueeze(1)
padded = pad_sequence(sequences, batch_first=True)  # shape: [batch_size, max_seq_len, num_features]
lengths, perm_idx = torch.tensor(lengths).sort(descending=True)
padded = padded[perm_idx]
targets = targets[perm_idx]

# Move data to device (GPU/CPU)
padded = padded.to(device)
targets = targets.to(device)

# Model setup
input_size = padded.shape[2]  # e.g. 5 features
model = LSTMRegressor(input_size=input_size, hidden_size=64, num_layers=1, bidirectional=False).to(device)  # Move model to device
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(padded, lengths)
    loss = criterion(outputs, targets)

    # Backward + Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {loss.item():.4f}")

# Print final outputs and targets
print(outputs.cpu())  # Move outputs to CPU for printing
print(targets.cpu())  # Move targets to CPU for printing

predict = model()
