import DataReading as DR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from LSTM_parameters import LSTMRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import seaborn as sns


# Check if CUDA is available, if so, use GPU; otherwise, fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
scaler = StandardScaler()
all_data = torch.cat(DR.readData5('train')[0]).numpy()
scaler.fit(all_data)  # Fit on training data
# Load data
def data_create(data_type, min_steps=3):
    sequences, lengths = DR.readData5(data_type)

    filtered = [(seq, length) for seq, length in zip(sequences, lengths) if seq[-1][2] != -30]

    input_sequences = []
    input_lengths = []
    targets = []

    short_sequences = []
    short_targets = []

    for seq, length in filtered:
        # Find first index where time_to_tca < 2
        cutoff_idx = None
        for idx, step in enumerate(seq):
            if step[1] < 2.0:  # time_to_tca < 2 days
                cutoff_idx = idx
                break
        
        if cutoff_idx is None or cutoff_idx < min_steps:
            # If no such point is found, or not enough steps, consider it a "short sequence"
            short_sequences.append(seq)
            short_targets.append(seq[-1][2])
        else:
            # Keep sequence up to (but not including) the cutoff_idx
            truncated_seq = seq[:cutoff_idx]
            truncated_seq_tensor = torch.tensor(scaler.transform(truncated_seq), dtype=torch.float32)

            input_sequences.append(truncated_seq_tensor)
            input_lengths.append(len(truncated_seq_tensor))
            targets.append(seq[-1][2])

    if input_sequences:
        padded_inputs = pad_sequence(input_sequences, batch_first=True)
        lengths_tensor = torch.tensor(input_lengths)
        lengths_sorted, sorted_indices = lengths_tensor.sort(descending=True)
        
        padded_inputs = padded_inputs[sorted_indices]
        targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        targets_tensor = targets_tensor[sorted_indices]
    else:
        padded_inputs = torch.empty(0, 0, 0)
        lengths_sorted = torch.empty(0)
        targets_tensor = torch.empty(0)

    # Move to device
    padded_inputs = padded_inputs.to(device)
    targets_tensor = targets_tensor.to(device)

    if short_sequences:
        short_targets_tensor = torch.tensor(short_targets, device=device).unsqueeze(1)
    else:
        short_sequences = []
        short_targets_tensor = torch.empty(0, 1).to(device)

    return padded_inputs, targets_tensor, lengths_sorted, short_sequences, short_targets_tensor

# Load the datasets (train and test)
padded_train, targets_train, lengths_train = data_create("train")
padded_test, targets_test, lengths_test = data_create("test")

# Model setup
input_size = padded_train.shape[2]  # e.g. 5 features
model = LSTMRegressor(input_size=input_size, hidden_size=50, num_layers=3, bidirectional=True, dropout=0.4).to(device)  # Move model to device

# Define scoring functions
def compute_f2(outputs, targets):
    outputs_bin = (outputs.squeeze().cpu().numpy() > -6)
    targets_bin = (targets.squeeze().cpu().numpy() > -6)
    return fbeta_score(targets_bin, outputs_bin, beta=2)

def compute_mse(outputs, targets):
    outputs = outputs.cpu().squeeze()
    targets = targets.cpu().squeeze()
    square_differences = []
    clipped_outputs = [max(output, -6.001) for output in outputs]
    for i in range(len(outputs)):
        if targets[i] >= -6:
            square_differences.append((clipped_outputs[i] - targets[i])**2)
    return np.sum(square_differences) / len(square_differences)

# Define loss function
def lossfn(outputs, targets):
    binary_targets = (targets >= -7.5).float() * 0.8 + 0.1  # Adjusted binary targets with optimal threshold
    criterion1 = nn.L1Loss()  
    criterion2 = nn.BCEWithLogitsLoss()
    
    loss1 = criterion1(outputs, targets)
    loss2 = criterion2(outputs, binary_targets)

    # Compute multitask loss with uncertainty
    sigma1 = torch.exp(model.log_sigma1).clamp(min=1e-3, max=10)
    sigma2 = torch.exp(model.log_sigma2).clamp(min=1e-3, max=10)

    multitask_loss = (1 / (2 * sigma1 ** 2)) * loss1 + (1 / (2 * sigma2 ** 2)) * loss2 + torch.log(sigma1) + torch.log(sigma2)

    # Boundary penalty to disencourage predictions near the boundary
    margin = 2.0
    penalty_strength = 0.15

    boundary_targets = (torch.abs(targets + 6) < margin).float()
    boundary_penalty = (boundary_targets * (outputs + 6).abs()).mean()

    # Zero penalty to disencourage higher values
    alpha = 5.0
    zero_penalty = torch.exp(-alpha * outputs.abs()).mean()

    return multitask_loss + penalty_strength * boundary_penalty + 0.1 * zero_penalty

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001) 
num_epochs = 1000

train_losses = []
test_losses = []
test_f2_scores = []
test_mse_scores = []

# Training loop
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
        # compute test loss and metrics
        outputs_test = model(padded_test, lengths_test)
        test_loss = lossfn(outputs_test, targets_test)
        test_f2 = compute_f2(outputs_test, targets_test)
        test_mse = compute_mse(outputs_test, targets_test)

    # Print progress bar
    progress = int(30 * (epoch + 1) / num_epochs) 
    bar = '[' + '=' * progress + '-' * (30 - progress) + ']'
    percent = 100 * (epoch + 1) / num_epochs

    sys.stdout.write(f"\rEpoch {epoch+1}/{num_epochs} {bar} {percent:.1f}%")
    sys.stdout.flush()

    # Logging the losses and metrics
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    test_f2_scores.append(test_f2)
    test_mse_scores.append(test_mse)


#Scoring model + Plotting predictions
def scoring(outputs, targets):
    # Compute F2 and MSE scores
    f2_score = compute_f2(outputs, targets)
    mse_score = compute_mse(outputs, targets)

    # Plotting
    outputs = outputs.cpu().squeeze().tolist()  # Move to CPU and convert to list for plotting
    targets = targets.cpu().squeeze().tolist()  # Move to CPU and convert to list for plotting
    steps = [i for i in range(len(outputs))] # Create x-axis based on event index

    plt.figure(figsize=(10, 6))
    plt.title("Test Results", fontsize=16)
    plt.scatter(steps, outputs, color='blue', label='Predicted', alpha=0.7)
    plt.scatter(steps, targets, color='red', label='Target', alpha=0.7)

    # Plot lines connecting predictions and targets
    for i in range(len(outputs)):
        plt.plot([steps[i], steps[i]], [outputs[i], targets[i]], color='gray', linestyle='--', alpha=0.5)

    # Plot threshold line
    plt.axhline(y=-6, color='green', linestyle='-', label='Threshold (-6)', linewidth=2)

    # Labeling
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    print(f"F2 Score: {f2_score:.4f}")
    print(f"Mean Squared Error (MSE): {mse_score:.4f}")
    print(f"Final Score: {mse_score/f2_score:.4f}")

# Final evaluation on test set
print("Test set:")
scoring(outputs_test, targets_test)

# Diagnostics
def diagnostics(outputs, targets, train_losses, test_losses, test_f2_scores, test_mse_scores, dataset_name="Test Set"):
    outputs_np = outputs.cpu().squeeze().numpy() # Move to CPU and convert to array for plotting
    targets_np = targets.cpu().squeeze().numpy() # Move to CPU and convert to array for plotting

    # 1. Plot loss curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 2. Plot F2 score over epochs
    plt.subplot(1, 2, 2)
    plt.plot(test_f2_scores, label='Test F2 Score', color='green')
    plt.title('Test F2 Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F2 Score')
    plt.legend()
    plt.show()

    # 3. Plot MSE score over epochs
    plt.plot(test_mse_scores, label='Test MSE Score', color='orange')
    plt.title('Test MSE Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Score')
    plt.legend()
    plt.show()

    # 4. Histogram of predictions vs targets
    plt.figure(figsize=(8, 6))
    sns.histplot(outputs_np, color='blue', label='Predicted', kde=True)
    sns.histplot(targets_np, color='orange', label='Target', kde=True)
    plt.axvline(x=-6, color='red', linestyle='--', label='Threshold (-6)')
    plt.title(f'Prediction vs Target Distributions ({dataset_name})')
    plt.legend()
    plt.show()

    # 5. Confusion matrix
    bin_outputs = (outputs_np > -6).astype(int)
    bin_targets = (targets_np > -6).astype(int)
    cm = confusion_matrix(bin_targets, bin_outputs)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Risk', 'High Risk'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix ({dataset_name})')
    plt.show()

# Diagnostics for test set
print("Test set diagnostics:")
diagnostics(outputs_test, targets_test, train_losses, test_losses, test_f2_scores, test_mse_scores, dataset_name="Test Set")