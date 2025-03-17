#Notes: orignigal risk is in log 10 scale
#Note: for now, each CDm is taken as an input and the CDM @ TCA as the target value

import pandas as pd
import numpy as np 

#data import 

def readData ():
    rawData = pd.read_csv("../DataSets/train_data.csv", usecols=[0,1,3])    
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
                t.append(CDM[1:3])
            i += 1
            j = 0
        elif CDM[0] == event_IDs[i]:
            X.append(CDM[1:3])
            j +=1

    return X,t 
print(readData())

        











    


    
           
    





