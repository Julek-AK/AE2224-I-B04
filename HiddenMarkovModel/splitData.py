import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def splitSet(dataName, cutPerc):

    dataSet =  pd.read_csv(rf"DataSets\{dataName}")

    # Split the data into training and validation sets
    trainDf, valDf = train_test_split(dataSet, test_size=cutPerc, random_state=10)

    # Print lengths to verify
    print(rf"Validation set length: {len(valDf)}")
    print(rf"Training set length: {len(trainDf)}")

    # Save the dataframes to CSV files
    valDf.to_csv("DataSets\HMM_validation_set.csv", index=False)
    trainDf.to_csv("DataSets\HMM_training_set.csv", index=False)

    return

def formatData(dataName, validation = False):
    observations = list()
    lengths = list()

    dataSet =  pd.read_csv(f"DataSets\{dataName}")

    observationList = dataSet['observations'].tolist()
    # print(observationList)

    for row in observationList:

        row = row.replace(",", "")
        row = row[1:-1]
        row = np.array(list(map(int, row.split()))).reshape(-1, 1)
        lengths.append(len(row))
        

        observations.append(row)
        
        #print(type(row))
    if validation: 
        outcomes = list()
        outcomeList = dataSet['outcome'].tolist()

        for row in outcomeList:

            row = row.replace(",", "")
            row = row[1:-1]
            row = np.array(list(map(int, row.split())))
            outcomes.append(row)

        return observations, outcomes
    
    else:
        return observations, lengths


# splitSet("HMM_train_data.csv", 0.1)

# formatData("HMM_training_set.csv")