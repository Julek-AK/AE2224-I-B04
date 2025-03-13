#Notes: orignigal risk is in log 10 scale
#Note: for now, each CDm is taken as an input and the CDM TCA as the target value

import pandas as pd
import numpy as np 

#data import 
rawData = pd.read_csv("../DataSets/train_data.csv", usecols=[0,3])
rawData = rawData.to_numpy()

#data collection, split up in input and output
#input vector X, target vector Y 
X = np.empty(len(rawData)-13153)
Y = np.empty(len(rawData)-13153) 





