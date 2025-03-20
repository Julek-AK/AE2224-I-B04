import numpy as np
from hmmlearn import hmm

lengths = []

observations = [np.array([0, 1, 1, 0, 0, 0]).reshape(-1, 1),
                np.array([1, 0, 0, 0, 0, 0]).reshape(-1, 1),
                np.array([0, 0, 1, 1, 1, 0]).reshape(-1, 1),
                np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1),
                np.array([0, 0, 1, 0, 1, 1]).reshape(-1, 1),
                np.array([1, 0, 0, 1, 1, 1]).reshape(-1, 1),
                np.array([0, 0, 1, 0, 1, 0]).reshape(-1, 1),
                np.array([1, 0, 0, 1, 1, 1]).reshape(-1, 1),
                np.array([0, 0, 1, 0, 1, 1]).reshape(-1, 1)]

test = np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1)

print(len(observations[0]))

for obs in observations:
    lengths.append(len(obs))

# gets correct format for hmmlearn
squishedObservations = np.concatenate(observations)



# function finds the best model for observations (matrices)
# returns a model
def idealPrediction(obs, lens):
    models = list()
    scores = list()
    # tests a few random starts to find the global maximum
    # look at results, see if it makes sense or not. Sometimes is stuck in a local maximum and then this needs to be increased
    for idx in range (50):
        # iteration number can also be increased for a lower likelyhood of being stuck
        model = hmm.CategoricalHMM(n_components = 2, random_state=idx, n_iter=10)
        model.fit(obs, lens)
        models.append(model)
        # score shows how good the model is
        scores.append(model.score(obs, lens))
        # see if it's working
        print(f'Converged: {model.monitor_.converged}\t\t'
                f'Score: {scores[-1]}')
        print(f'The best model had a score of {max(scores)}')
    # returns the best performing model
    return models[np.argmax(scores)]

model1 = idealPrediction(squishedObservations, lengths)

states = model1.predict(test)

print(states)
print(f'after: {model1.transmat_}')


# function for predicting the next sequence of observations
#HAS TO BE REDONE
# https://github.com/hmmlearn/hmmlearn/issues/171
def predictNext(model, observations, steps = 1):
    
    transMatrix = model.transmat_
    # We only care about the last value in hidden
    prediction = model.predict(observations)[-1]
    nextSequence = []

    # print(f'Pred: {prediction}')
    # print(transMatrix)
    if int(steps) != steps:
        print("Imput value steps must be an integer")
        raise TypeError
    while steps >= 1:
        if steps == 1:
            if len(nextSequence) == 0:
                # choose the highest probability
                nextSequence.append(np.argmax(transMatrix[prediction]))
            else:
                nextSequence.append(np.argmax(transMatrix[nextSequence[-1]]))
            return nextSequence
        
        else:
            # predicts the next element randomly based on the transition probability
            nextSequence.append(np.random.choice(np.arange(0, 2, 1), p=[transMatrix[prediction][0], transMatrix[prediction][1]]))
            # print([transMatrix[prediction][0], transMatrix[prediction][1]])
            # print(nextSequence[-1])
        steps -= 1

#print(f'observation predicting: {observations[1]}')
print(f'next prediction: {predictNext(model1, test, 6)}')