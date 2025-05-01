import numpy as np
from hmmLearnAlgorithm import idealPrediction, predictAndScore, predictNext, averagePredictions
from splitData import splitSet, formatData
import pandas as pd
# import pickle
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.scoring import benchmark


lengths = []

# train set
splitSet("HMM_train_data.csv", 0.1)
observations, lengths = formatData("HMM_training_set.csv")
squishedObservations = np.concatenate(observations)

# validation set
valObservations, valOutcomes = formatData("HMM_validation_set.csv", validation=True)

# test set
testObservations, testOutcomes, testIDs = formatData("HMM_test_data_shifted.csv", test= True)

# train model with train set
model1 = idealPrediction(squishedObservations, lengths, 30)
prediction= predictAndScore(model1, testObservations, testOutcomes, steps = 5, score = False, binary = False)

predictionPD = pd.DataFrame({'predicted_risk': prediction, 'event_id': testIDs})
testOutcomePD = pd.DataFrame({'true_risk': testOutcomes, 'event_id': testIDs})

F_score, MSE_HR, L_score = benchmark(predictionPD)

# print(f"fscore: {F_score}")
# print(f"mse: {MSE_HR}")
# print(f"Lscore: {L_score}")