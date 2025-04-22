import numpy as np
from hmmLearnAlgorithm import idealPrediction, predictAndScore, predictNext, averagePredictions
from splitData import splitSet, formatData
import pickle
import os


lengths = []

# train set
splitSet("HMM_train_data.csv", 0.1)
observations, lengths = formatData("HMM_training_set.csv")
squishedObservations = np.concatenate(observations)

# validation set
valObservations, valOutcomes = formatData("HMM_validation_set.csv", validation=True)

# train model with train set
model1 = idealPrediction(squishedObservations, lengths, 30)
# with open("HiddenMarkovModel\\hmmModel1.pkl", "wb") as f: pickle.dump(model1, f)

#print(f'transmat {model1.transmat_}')

# check with validation model and get score
# predictions, nextScore, lastScore = predictAndScore(model1, valObservations, valOutcomes, steps = 3)
# print(f'next: {round(nextScore, 3)}%, last: {round(lastScore, 3)}%')
# file = open("hmmModel.pkl", 'rb')
# model1 = pickle.load(file)
# path = os.path.abspath("hmmModel1.pkl")

# print(f"Checking path: {path}")
# print(f"File exists: {os.path.exists(path)}")
# with open(path, "rb") as f:
#     model1 = pickle.load(f)
# model1 = pickle.load(open('hmmModel.pkl', 'rb'))
print(averagePredictions(model1, observations[0], 3, 3))
