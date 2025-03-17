import numpy as np
from hmmlearn import hmm

lengths = []

observations = [np.array([0, 1, 1, 0, 0, 0]).reshape(-1, 1),
                np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1),
                np.array([1, 0, 0, 0, 0, 0]).reshape(-1, 1),
                np.array([0, 1, 1, 1, 1, 1]).reshape(-1, 1),
                np.array([0, 1, 0, 0, 0, 0]).reshape(-1, 1),
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
    for idx in range (10):
        model = hmm.CategoricalHMM(n_components = 2, random_state=idx, n_iter=20)
        model.fit(observations, lengths)
        models.append(model)
        scores.append(model.score(observations, lengths))
        print(f'Converged: {model.monitor_.converged}\t\t'
                f'Score: {scores[-1]}')
        print(f'The best model had a score of {max(scores)}')
    return models[np.argmax(scores)]

model = idealPrediction(squishedObservations, lengths)

states = model.predict(observations[5])

print(states)
print(f'after: {model.transmat_}')



