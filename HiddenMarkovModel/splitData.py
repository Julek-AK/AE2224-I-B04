import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def splitSet(dataName, cutPerc):

    dataSet =  pd.read_csv(rf"DataSets\\{dataName}")

    # Split the data into training and validation sets
    trainDf, valDf = train_test_split(dataSet, test_size=cutPerc, random_state=42)

    # Print lengths to verify
    print(rf"Validation set length: {len(valDf)}")
    print(rf"Training set length: {len(trainDf)}")

    # Save the dataframes to CSV files
    valDf.to_csv("DataSets\\HMM_validation_set.csv", index=False)
    trainDf.to_csv("DataSets\\HMM_training_set.csv", index=False)

    return

def formatData(dataName, validation = False, test = False):
    '''
    Takes the input from the database and reformats it
    returns two arrays
    input: 
    - dataName: part of the DataSets file, string
    - validation: if true, outcomes are also returned

    returns: 
    - formatted data for HMM analysis
    - if validation is true, also the outcomes for each observation


    '''
    observations = list()
    lengths = list()

    dataSet =  pd.read_csv(f"DataSets\{dataName}")

    # We need the observations no matter if it's the validation or training set
    observationList = dataSet['observations'].tolist()
    

    # Formatting, as the data for some reason is a string, also converts it to the right shaped array
    for row in observationList:

        row = row.replace(",", "")
        row = row[1:-1]
        row = np.array(list(map(int, row.split()))).reshape(-1, 1)
        lengths.append(len(row))
        observations.append(row)
        
    # If it's a validation set we don't need the lengths but we do need the outcomes to test against
    if validation or test: 
        outcomes = list()
        outcomeList = dataSet['outcome'].tolist()

        for row in outcomeList:

            row = row.replace(",", "")
            row = row[1:-1]
            row = np.array(list(map(int, row.split())))
            outcomes.append(row)
        if test:
            eventIDs = dataSet['event_id'].tolist()
            return observations, outcomes, eventIDs

        return observations, outcomes
    
    else:
        return observations, lengths
