"""ChatGPTs solution to LSTMs"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Synthetic sine wave data for example
t = np.linspace(0, 100, 500)
data = np.sin(t) + 0.1 * np.random.randn(len(t))
data = torch.tensor(data, dtype=torch.float32)

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

seq_length = 20
X, y = create_sequences(data, seq_length)
X = X.unsqueeze(-1)  # Add input_size dim (1 feature)

# Define the LSTM model
class LSTMTimeSeries(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMTimeSeries, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Use last time step for prediction

# Train the model
model = LSTMTimeSeries()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(20):
    for xb, yb in loader:
        pred = model(xb)
        loss = criterion(pred.squeeze(), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Make predictions
model.eval()
preds = []
last_seq = X[-1].unsqueeze(0)  # Last sequence for prediction

with torch.no_grad():
    for _ in range(30):  # Predict 30 time steps ahead
        pred = model(last_seq)              # (1, 1)
        new_step = pred.unsqueeze(2)        # (1, 1, 1)
        preds.append(pred.item())
        last_seq = torch.cat([last_seq[:, 1:, :], new_step], dim=1)  # (1, seq_len, 1)

# Actual future values (ground truth)
true_future = data[-30:]

# Plotting
x_train = list(range(len(data) - 30))
x_pred = list(range(len(data) - 30, len(data)))
plt.figure(figsize=(12, 6))
plt.plot(x_train, data[:-30], label='Training Data')
plt.plot(x_pred, true_future, label='Actual Future', color='green')
plt.plot(x_pred, preds, label='Predicted Future', color='red', linestyle='--')
plt.title("LSTM Time Series Forecasting")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
