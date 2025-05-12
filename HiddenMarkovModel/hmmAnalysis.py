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
splitSet("HMM_SMOTE_data2.csv", 0.1)
observations, lengths = formatData("HMM_SMOTE_data2.csv")
squishedObservations = np.concatenate(observations)

# validation set
valObservations, valOutcomes, testIDs = formatData("HMM_validation_set.csv", test=True)

# test set
# testObservations, testOutcomes, testIDs = formatData("HMM_test_data_shifted.csv", test=True)

# train model with train set
model1 = idealPrediction(squishedObservations, lengths, 30)
prediction= predictAndScore(model1, valObservations, valOutcomes, steps = 1, score = False, binary = False)

predictionPD = pd.DataFrame({'predicted_risk': prediction, 'event_id': testIDs})
testOutcomePD = pd.DataFrame({'true_risk': valOutcomes, 'event_id': testIDs})

F_score, MSE_HR, L_score = benchmark(predictionPD, true_data='SMOTE_data2.csv')

# print(f"fscore: {F_score}")
# print(f"mse: {MSE_HR}")
# print(f"Lscore: {L_score}")