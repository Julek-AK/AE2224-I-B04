import numpy as np
from hmmLearnAlgorithm import idealPrediction, scores
from splitData import splitSet, formatData

lengths = []

# train set
splitSet("HMM_train_data.csv", 0.1)
observations, lengths = formatData("HMM_training_set.csv")
squishedObservations = np.concatenate(observations)
test = np.array([1, 1, 1, 1, 1, 1]).reshape(-1, 1)

# validation set
valObservations, valOutcomes = formatData("HMM_validation_set.csv", validation=True)

model1 = idealPrediction(squishedObservations, lengths)
print(f'transmat {model1.transmat_}')

nextScore, lastScore = scores(model1, valObservations, valOutcomes)

print(f'next: {nextScore}, last: {lastScore}')


