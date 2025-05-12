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
def data_create(data_type):
    sequences, lengths = DR.readData5(data_type)

    # Filter out sequences with target value == -30
    filtered = [(seq, length) for seq, length in zip(sequences, lengths) if seq[-1][2] != -30]
    # separate long and short sequences
    MIN_LENGTH = 5
    long_sequences = [(seq, length) for seq, length in filtered if length >= MIN_LENGTH]
    short_sequences = [seq for seq, length in filtered if length < MIN_LENGTH]
    short_targets = torch.tensor([seq[-1][2] for seq in short_sequences], device=device).unsqueeze(1)  

    sequences, lengths = zip(*long_sequences)
    # Scale each sequence individually
    scaled_sequences = [torch.tensor(scaler.transform(seq[:-1]), dtype=torch.float32) for seq in sequences]
    # Pad to (batch, max_len, features)
    targets = torch.tensor([seq[-1][2] for seq in sequences], dtype=torch.float32).unsqueeze(1)
    padded = pad_sequence(scaled_sequences, batch_first=True)  # shape: [batch_size, max_seq_len, num_features]
    lengths, perm_idx = torch.tensor(lengths).sort(descending=True)
    padded = padded[perm_idx]
    targets = targets[perm_idx]

    # Move data to device (GPU/CPU)
    padded = padded.to(device)
    targets = targets.to(device)
    
    return padded, targets, lengths, short_sequences, short_targets

# Load the datasets (train and test)
padded_train, targets_train, lengths_train, short_sequences_train, short_targets_train = data_create("train")
padded_test, targets_test, lengths_test, short_sequences_test, short_targets_test = data_create('test')

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

# Default prediction function (from first place method)
def default_prediction(seq):
    r_min2 = seq[-1][2]
    if -6.04 <= r_min2 < -6:
        return -5.95
    elif -6.4 <= r_min2 < -6.04:
        return -5.6
    elif -7.3 <= r_min2 < -6.4:
        return -5
    elif -4 <= r_min2 < -3.5:
        return -4
    elif r_min2 >= -3.5:
        return -3.5
    else:
        return -6.0001

# Prediction function for both short and long sequences
def predict_with_default(model, sequences, lengths, short_sequences):
    preds = []

    # First, predict for short sequences using default_prediction
    for seq in short_sequences:
        preds.append(default_prediction(seq))

    # Then, predict for normal sequences using the model
    for seq, length in zip(sequences, lengths):
        seq = seq.unsqueeze(0).to(device)
        length_tensor = torch.tensor([length], dtype=torch.long, device="cpu")
        output = model(seq, length_tensor)
        preds.append(output.item())
    return torch.tensor(preds, device=device).unsqueeze(1)  # unsqueeze(1) to match target shape

model.eval()
with torch.no_grad():
    # Predict both short and long sequences
    outputs_test = predict_with_default(model, padded_test, lengths_test, short_sequences_test)
# Comine targets for short and long sequences
targets_test = torch.cat([short_targets_test, targets_test]).squeeze()

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