#Notes: orignigal risk is in log 10 scale
#Note: for now, each CDm is taken as an input and the CDM @ TCA as the target value
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import torch

def readData4(data_type):
    filename = "../DataSets/train_data.csv" if data_type == "validation" else f"../DataSets/{data_type}_data.csv"
    rawData = pd.read_csv(filename, usecols=[0,1,3,4,5,6,84]).to_numpy()
    OA_data = pd.read_csv(filename, usecols=[28,59]).to_numpy()

    # Compute OA (difference between the two columns)
    OA = np.array([CDM[0] - CDM[1] for CDM in OA_data])

    # Combine raw data and OA into one array
    CDMlist = np.column_stack((rawData, OA))

    # Group data by event index (first column)
    event_indices = CDMlist[:, 0].astype(int)
    unique_events = np.unique(event_indices)
    grouped_events = [torch.tensor(CDMlist[event_indices == eid], dtype=torch.float32) for eid in unique_events]

    # Sort by sequence length (descending)
    grouped_events.sort(key=lambda x: x.shape[0], reverse=True)
    lengths = [seq.shape[0] for seq in grouped_events]

    # Return based on type
    if data_type == "test":
        return grouped_events, lengths
    else:
        # Create labels for stratification (1 if last target > -6, else 0)
        labels = [(seq[-1][2] > -6).int().item() for seq in grouped_events]

        # Perform stratified split
        trainset, validationset = train_test_split(
            list(zip(grouped_events, lengths)),
            test_size=0.1,
            random_state=45,
            stratify=labels
        )

        dataset = validationset if data_type == "validation" else trainset
        sequences, lengths = zip(*dataset)
        return list(sequences), list(lengths)
    
def readData5(data_type):
    filename = f"../DataSets/{data_type}_data.csv"
    rawData = pd.read_csv(filename, usecols=[0,1,3,4,5,6,84]).to_numpy()
    OA_data = pd.read_csv(filename, usecols=[28,59]).to_numpy()

    # Compute OA (difference between the two columns)
    OA = np.array([CDM[0] - CDM[1] for CDM in OA_data])

    # Combine raw data and OA into one array
    CDMlist = np.column_stack((rawData, OA))

    # Group data by event index (first column)
    event_indices = CDMlist[:, 0].astype(int)
    unique_events = np.unique(event_indices)
    grouped_events = [torch.tensor(CDMlist[event_indices == eid], dtype=torch.float32) for eid in unique_events]

    # Sort by sequence length (descending)
    grouped_events.sort(key=lambda x: x.shape[0], reverse=True)
    lengths = [seq.shape[0] for seq in grouped_events]

    return grouped_events, lengths

filter1 = [seq for seq in readData4("train")[0] if len(seq)>=7]
filter2 = [seq for seq in filter1 if seq[-1][2] != -30]
print(len(filter2))  # Count sequences longer than 5 in test set
