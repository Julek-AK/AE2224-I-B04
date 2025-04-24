import numpy as np
from hmmlearn import hmm


def idealPrediction(obs, lens, n_iter):
    '''function finds the best model for observations (matrices),
    returns a model'''
    models = list()
    fitScores = list()

    # tests a few random starts to find the global maximum
    # look at results, see if it makes sense or not. Sometimes is stuck in a local maximum and then this needs to be changed
    for idx in range (50, 70):

        # iteration number can also be changed for a lower likelyhood of being stuck
        model = hmm.CategoricalHMM(n_components = 2, random_state=idx, n_iter=n_iter)
        model.fit(obs, lens)
        models.append(model)

        # score shows how good the model is
        fitScores.append(model.score(obs, lens))
        # see if it's working
        print(f'Converged {idx}: {model.monitor_.converged}\t\t'
                f'Score: {fitScores[-1]}')
    print(f'The best model had a score of {max(fitScores)}')
    # returns the best performing model
    return models[np.argmax(fitScores)]


# https://github.com/hmmlearn/hmmlearn/issues/171
def predictNext(model, observations, steps = 1):
    '''Function to predict a future sequence of observations
    returns the prediction array'''
    
    # Get the transition matrix from trained model
    transMatrix = model.transmat_
    # We only care about the last value in hidden
    prediction = model.predict(observations)[-1]

    nextSequence = []

    # Make sure input value is correct
    if int(steps) != steps:
        print("Imput value steps must be an integer")
        raise TypeError
    
    #prediction procedure
    while steps >= 1:
        if steps == 1 and len(nextSequence) == 0:
            # Just take most likely next value if we are only looking one in the future
            nextSequence.append(np.argmax(transMatrix[prediction]))
        
        else:
            # predicts the next element randomly based on the transition probability
            nextSequence.append(np.random.choice(np.arange(0, 2, 1), p=[transMatrix[prediction][0], transMatrix[prediction][1]]))
        steps -= 1
    return nextSequence

def predictAndScore(model, observations, outcomes, steps=1, score = True, verbose = False):
    scoreNext = 0
    scoreLast = 0
    predictNonBinary = []

    # gets prediction
    for i, observation in enumerate(observations):
        futurePrediction = averagePredictions(model, observation, steps = steps, avTimes = 3)
        if verbose:
            print(f"Predicted: {model.predict(observation)}")
            print(f'next prediction: {futurePrediction}')
        



        if score:
            #if predicted is correct, add point
            if futurePrediction[0] == outcomes[i][0]:
                scoreNext += 1

            if futurePrediction[-1] == outcomes[i][-1]:
                scoreLast += 1
        else:
            if outcomes[i][-1] == 0:
                predictNonBinary.append(-6.001)
            else: predictNonBinary.append(-5.34)
    
    if not score: return futurePrediction, predictNonBinary
    
    # average to get percentage correct
    scoreNext = scoreNext / len(observations) * 100
    scoreLast = scoreLast/ len(observations) * 100
    
    return futurePrediction, scoreNext, scoreLast

def averagePredictions(model, observation, steps = 1, avTimes = 1):
    predictions = []
    for i in range(avTimes):
        predictions.append(predictNext(model, observation, steps = steps))
    
    average = np.mean(np.array(predictions), axis = 0)
    roundedAverage = np.round(average).astype(int)

    return roundedAverage
