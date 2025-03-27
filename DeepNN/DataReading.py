#Notes: orignigal risk is in log 10 scale
#Note: for now, each CDm is taken as an input and the CDM @ TCA as the target value

import pandas as pd
import numpy as np 
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

def readData2(data_type):
    rawData = pd.read_csv("../DataSets/"+data_type+"_data.csv", usecols=[0,1,3,84])
    rawData = rawData.to_numpy()
    OA_data = pd.read_csv("../DataSets/"+data_type+"_data.csv", usecols=[28,59])
    OA_data = OA_data.to_numpy()

#data collection, split up in input and output
#input vector X, target vector Y 
    event = []

    i  = 0
    j = 0
    OA = np.array([])
    for CDM in OA_data:
        if abs(CDM[0] - CDM[1]) < abs(CDM[1]-CDM[0]):
            OA = np.array(CDM[0] - CDM[1])
        else:
            OA = np.array(180)
    OA = np.array([CDM[0] - CDM[1] for CDM in OA_data])
    for i in range(len(rawData)):
        for j, CDM in enumerate(rawData):
            event.append(np.append(CDM, OA[j]))

    #Maybe implement normalsing here
    return event
print(readData2("test"))