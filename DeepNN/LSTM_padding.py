import DataReading as DR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from LSTM_parameters import LSTMRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
import numpy as np
from sklearn.preprocessing import StandardScaler

# Check if CUDA is available, if so, use GPU; otherwise, fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
scaler = StandardScaler()
all_data = torch.cat(DR.readData4('train')[0]).numpy()
scaler.fit(all_data)  # Fit on all valid timesteps
# Load data
def data_create(data_type):
    sequences, lengths = DR.readData4(data_type)

    # Filter out sequences with target value == -30
    filtered = [(seq, length) for seq, length in zip(sequences, lengths) if seq[-1][2] != -30]

    sequences, lengths = zip(*filtered)
    lengths = [length for length in lengths if length > 1]  # Filter out empty sequences
    # Scale each sequence individually
    scaled_sequences = [torch.tensor(scaler.transform(seq[:-1]), dtype=torch.float32) for seq in sequences if len(seq) > 1]

    # Pad to (batch, max_len, features)
    targets = torch.tensor([seq[-1][2] for seq in sequences], dtype=torch.float32).unsqueeze(1)
    padded = pad_sequence(scaled_sequences, batch_first=True)  # shape: [batch_size, max_seq_len, num_features]
    lengths, perm_idx = torch.tensor(lengths).sort(descending=True)
    padded = padded[perm_idx]
    targets = targets[perm_idx]

    # Move data to device (GPU/CPU)
    padded = padded.to(device)
    targets = targets.to(device)
    
    return padded, targets, lengths
padded_train, targets_train, lengths_train = data_create("train")
padded_test, targets_test, lengths_test = data_create("test")
padded_validation, targets_validation, lengths_validation = data_create("validation")


# Model setup
input_size = padded_train.shape[2]  # e.g. 5 features
model = LSTMRegressor(input_size=input_size, hidden_size=50, num_layers=2, bidirectional=True,dropout=0.35).to(device)  # Move model to device
def lossfn(outputs, targets):
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    loss_raw = criterion1(outputs, targets) + criterion2(outputs, targets)
    weights = 1/(torch.abs(targets) + 1)  # Example: smaller values get higher weight
    return (loss_raw * weights).mean()

best_val_loss = float('inf')
patience = 5
patience_counter = 0

optimizer = optim.Adam(model.parameters(),lr=0.005,weight_decay=1e-4)
# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs_train = model(padded_train, lengths_train)
    loss = lossfn(outputs_train, targets_train)
    # Backward + Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs_validation = model(padded_validation, lengths_validation)
        val_loss = lossfn(outputs_validation, targets_validation)

    # Logging
    print(f"Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    # if val_loss.item() < best_val_loss:
    #     best_val_loss = val_loss.item()
    #     best_model_state = model.state_dict()  # Save best model
    #     patience_counter = 0
    # else:
    #     patience_counter += 1
    #     if patience_counter >= patience:
    #         print("Early stopping triggered!")
    #         model.load_state_dict(best_model_state)  # Restore best model
    #         #break
    
    

# Print final outputs and targets
print(outputs_train.cpu())  # Move outputs to CPU for printing
print(targets_train.cpu())  # Move targets to CPU for printing

model.eval()

# Disable gradient calculation during testing (saves memory and computations)
with torch.no_grad():
    # Forward pass on the test set
    outputs_test = model(padded_test, lengths_test)
    outputs_validation = model(padded_validation, lengths_validation)

# Print the results

outputs_test = outputs_test.cpu().squeeze().tolist()  # Move to CPU and convert to list for plotting
# outputs_test = [x+1 for x in outputs_test]  # Adjust the output values
targets_test = targets_test.cpu().squeeze().tolist()  # Move to CPU and convert to list for plotting
outputs_validation = outputs_validation.cpu().squeeze().tolist()  # Move to CPU and convert to list for plotting
# outputs_validation = [x+1 for x in outputs_validation]  # Adjust the output values
targets_validation = targets_validation.cpu().squeeze().tolist()  # Move to CPU and convert to list for plotting


def scoring(outputs, targets):
    outputs_bin = [1 if x > -6 else 0 for x in outputs]
    targets_bin = [1 if x > -6 else 0 for x in targets]
    f2_score = fbeta_score(targets_bin, outputs_bin, beta=2)

    square_differences = []
    for i in range(len(outputs)):
        if targets[i] > -6:
            if outputs[i] < -6:
                square_differences.append((-6.001 - targets[i])**2)
            else:
                square_differences.append((outputs[i] - targets[i])**2)
    MSE = np.sum(square_differences) / len(square_differences)
    steps = [i for i in range(len(outputs))]
    plt.title("Test Results")
    plt.scatter(steps, outputs, label='Predicted')  # squeeze() to remove the extra dimension
    plt.scatter(steps, targets, label='Target')  # squeeze() to remove the extra dimension
    plt.legend()
    plt.show()
    print(f"F2 Score test: {f2_score:.4f}")
    print(f"Mean Squared Error (MSE) test: {MSE:.4f}")
    print(f"Final Score test: {MSE/f2_score:.4f}")
print("Test set:")
scoring(outputs_test, targets_test)
print("Validation set:")
scoring(outputs_validation, targets_validation)