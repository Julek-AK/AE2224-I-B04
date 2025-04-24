#Notes: orignigal risk is in log 10 scale
#Note: for now, each CDm is taken as an input and the CDM @ TCA as the target value
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import torch
#data import 

def readData(data_type):
    rawData = pd.read_csv("../DataSets/"+data_type+"_data.csv", usecols=[0,1,3])    
    rawData = rawData.to_numpy()

#data collection, split up in input and output
#input vector X, target vector Y 
    X = []
    t = []

    event_IDs = rawData[:, 0].astype(int)
    event_counts = np.bincount(event_IDs)
    event_IDs = np.unique(event_IDs)

    i  = 0
    j = 0

    for CDM in rawData:
        if j == event_counts[i]-1:
            for k in range(event_counts[i]-1):
                t.append(CDM[2])
            i += 1
            j = 0
        elif CDM[0] == event_IDs[i]:
            X.append(CDM[1:3])
            j +=1

    np.savetxt("../DataSets/ProcessedData.txt", np.column_stack((X,t)), delimiter=",")

    #Maybe implement normalsing here
    return X,t 



def readData3(data_type):
    if data_type == 'validation':
        rawData = pd.read_csv("../DataSets/train_data.csv", usecols=[0,1,3,84])
        rawData = rawData.to_numpy()
        OA_data = pd.read_csv("../DataSets/train_data.csv", usecols=[28,59])
        OA_data = OA_data.to_numpy()
    else:
        rawData = pd.read_csv("../DataSets/"+data_type+"_data.csv", usecols=[0,1,3,84])
        rawData = rawData.to_numpy()
        OA_data = pd.read_csv("../DataSets/"+data_type+"_data.csv", usecols=[28,59])
        OA_data = OA_data.to_numpy()
    
#data collection, split up in input and output
#input vector X, target vector Y 
    # OA = np.array([])
    # for CDM in OA_data:
    #     if abs(CDM[0] - CDM[1]) < abs(CDM[1]-CDM[0]):
    #         OA = np.array(CDM[0] - CDM[1])
    #     else:
    #         OA = np.array(180)
    OA = np.array([CDM[0] - CDM[1] for CDM in OA_data])
    CDMlist = np.column_stack((rawData, OA))
    np.savetxt("../DataSets/ProcessedData.txt", CDMlist, delimiter=",")
    #Maybe implement normalsing here
    event_indices = CDMlist[:, 0].astype(int)  # Convert index to integers
    unique_events = np.unique(event_indices)  # Get unique event IDs

    # Group data by event index
    grouped_events = [CDMlist[event_indices == event_id] for event_id in unique_events]

    if data_type == "test":
        return grouped_events  # List of arrays
    else:
        trainset, validationset = train_test_split(grouped_events, test_size=0.1, random_state=42, shuffle=False)
        return validationset if data_type == "validation" else trainset
readData3("Train")

def readData4(data_type):
    filename = "../DataSets/train_data.csv" if data_type == "validation" else f"../DataSets/{data_type}_data.csv"
    rawData = pd.read_csv(filename, usecols=[0,1,3,4,6,84]).to_numpy()
    OA_data = pd.read_csv(filename, usecols=[28,59]).to_numpy()

    # Compute OA (difference between the two columns)
    OA = np.array([CDM[0] - CDM[1] for CDM in OA_data])

    # Combine raw data and OA into one array
    CDMlist = np.column_stack((rawData, OA))
    np.savetxt("../DataSets/ProcessedData.txt", CDMlist, delimiter=",")

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
        trainset, validationset = train_test_split(list(zip(grouped_events, lengths)), test_size=0.1, random_state=42, shuffle=False)
        dataset = validationset if data_type == "validation" else trainset
        sequences, lengths = zip(*dataset)
        return list(sequences), list(lengths)