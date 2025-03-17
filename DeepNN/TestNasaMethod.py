#Notes: orignigal risk is in log 10 scale
#Note: for now, each CDm is taken as an input and the CDM @ TCA as the target value

import pandas as pd
import numpy as np 

#data import 
rawData = pd.read_csv("../DataSets/train_data.csv", usecols=[0,1,3])
rawData = rawData.to_numpy()

#data collection, split up in input and output
#input vector X, target vector Y 
X = []
t = []

event_IDs = rawData[:, 0].astype(int)
event_counts = np.bincount(event_IDs)
event_IDs = np.unique(event_IDs)

i = 0 
for CDM in rawData:
    if i == event_counts[i]:
        for j in range(event_counts[i]):
            t = np.append(t, CDM[1:3])
        i += 1

    if CDM[0] == event_IDs[i]:
        X = np.append(X, CDM[1:3])

print(len(X), len(Y))









    


    
           
    





