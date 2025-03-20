import numpy as np
from hmmlearn import hmm
from hmmLearnAlgorithm import idealPrediction, predictNext
from splitData import splitSet, formatData

lengths = []

# train set
splitSet("HMM_train_data.csv", 0.1)
observations, lengths = formatData("HMM_training_set.csv")
test = np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1)

model1 = idealPrediction(observations, lengths)
print(f"Predicted: {model1.predict(test)}")



#print(f'observation predicting: {observations[1]}')
print(f'next prediction: {predictNext(model1, test, 6)}')