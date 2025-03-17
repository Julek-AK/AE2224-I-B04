import numpy as np
from hmmlearn import hmm

lengths = []

observations = [np.array([0, 1, 1, 0, 0, 0]).reshape(-1, 1),
                np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1),
                np.array([0, 0, 1, 1, 1, 0]).reshape(-1, 1),
                np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1),
                np.array([0, 0, 1, 0, 1, 1]).reshape(-1, 1),
                np.array([1, 1, 1, 1, 1, 1]).reshape(-1, 1)]

print(len(observations[0]))

for obs in observations:
    lengths.append(len(obs))

#gets correct format for hmmlearn
squishedObservations = np.concatenate(observations)


def idealPrediction(observations, lengths):
    models = list()
    scores = list()
    #tests a few random starts to find the global maximum
    # look at results, see if it makes sense or not. Sometimes is stuck in a local maximum and then this needs to be increased
    for idx in range (100):
        #iteration number can also be increased for a lower likelyhood of being stuck
        model = hmm.CategoricalHMM(n_components = 2, random_state=idx, n_iter=40)
        model.fit(observations, lengths)
        models.append(model)
        scores.append(model.score(observations, lengths))
        print(f'Converged: {model.monitor_.converged}\t\t'
                f'Score: {scores[-1]}')
        print(f'The best model had a score of {max(scores)}')
    return models[np.argmax(scores)]

model1 = idealPrediction(squishedObservations, lengths)


currentObservation = observations[0]

states = model1.predict(currentObservation)

print(states)
print(f'after: {model1.transmat_}')

def predictNext(model, observations):
    transMatrix = model.transmat_
    prediction = model.predict(observations)

    return np.argmax(transMatrix[prediction[-1]])

print(predictNext(model1, currentObservation))





