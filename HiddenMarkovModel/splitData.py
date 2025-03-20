import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def splitSet(dataName, cutPerc):

    dataSet =  pd.read_csv(f"DataSets\{dataName}")

    # Split the data into training and validation sets
    trainDf, valDf = train_test_split(dataSet, test_size=cutPerc, random_state=42)

    # Print lengths to verify
    print(f"Validation set length: {len(valDf)}")
    print(f"Training set length: {len(trainDf)}")

    # Save the dataframes to CSV files
    valDf.to_csv("DataSets\HMM_validation_set.csv", index=False)
    trainDf.to_csv("DataSets\HMM_training_set.csv", index=False)

    return

def formatData(dataName):
    observations = list()
    lengths = list()

    dataSet =  pd.read_csv(f"DataSets\{dataName}")

    observationList = dataSet['observations'].tolist()
    # print(observationList)

    for row in observationList:

        row = row.replace(",", "")
        row = row[1:-1]
        # print(len(row))
        #print(list(map(int, row.split())))
        row = np.array(list(map(int, row.split()))).reshape(-1, 1)
        #print(f'row {row}')
        lengths.append(len(row))
        

        observations.append(row)
        
        #print(type(row))

    squishedObservations = np.concatenate(observations)


    return squishedObservations, lengths

splitSet("HMM_train_data.csv", 0.1)

formatData("HMM_training_set.csv")