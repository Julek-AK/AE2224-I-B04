
# External Imports
import numpy as np
import os
import sys

# sys.path.append("../utils")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Internal Imports
from hmmLearnAlgorithm import idealPrediction, predictAndScore, predictNext, averagePredictions
from splitData import splitSet, formatData
from utils.scoring import benchmark



if __name__ == "__main__":

    lengths = []

    splitSet("HMM_train_data.csv", 0.1)
    observations, lengths = formatData("HMM_training_set.csv")
    squishedObservations = np.concatenate(observations)

    valObservations, valOutcomes = formatData("HMM_validation_set.csv", validation=True)

    model1 = idealPrediction(squishedObservations, lengths, 30)

    future, nonBinary = predictAndScore(model1, valObservations, valOutcomes, steps=3, score=False)

    print(future)
    print(nonBinary)
