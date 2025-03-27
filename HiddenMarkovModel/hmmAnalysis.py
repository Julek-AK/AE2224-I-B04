import numpy as np
from hmmLearnAlgorithm import idealPrediction, scores
from splitData import splitSet, formatData

lengths = []

# train set
splitSet("HMM_train_data.csv", 0.1)
observations, lengths = formatData("HMM_training_set.csv")
squishedObservations = np.concatenate(observations)

# validation set
valObservations, valOutcomes = formatData("HMM_validation_set.csv", validation=True)

# train model with train set
model1 = idealPrediction(squishedObservations, lengths)
print(f'transmat {model1.transmat_}')

# check with validation model and get score
nextScore, lastScore = scores(model1, valObservations, valOutcomes, steps = 3)
print(f'next: {round(nextScore, 3)}%, last: {round(lastScore, 3)}%')