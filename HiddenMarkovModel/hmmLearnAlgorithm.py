import numpy as np
from hmmlearn import hmm


def idealPrediction(obs, lens, nIter):
    '''
    function finds the best model for observations (matrices),
    returns a model

    inputs: 
    - obs: array of observations, concatenate them into a single array shape (-1, 1)
    - lens: list of lengths of each observation before it was concatenated to one array
    - nIter: number of iterations to perform

    outputs:
    - best model HMMModel
    
    '''
    models = list()
    fitScores = list()

    # tests a few random starts to find the global maximum
    # look at results, see if it makes sense or not. Sometimes is stuck in a local maximum and then this needs to be changed
    for idx in range (50, 70):

        # iteration number can also be changed for a lower likelyhood of being stuck
        model = hmm.CategoricalHMM(n_components = 2, random_state=idx, n_iter=nIter)
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
def predictNext(model, observationList, steps = 1, binary = True):
    '''Function to predict a future sequence of observations
    returns the prediction array or binary value of last prediction
    
    inputs: 
    - model: trained hmm model
    - observationList: a list composed of elements of a single observation
    - steps: how far in the future to predict
    - binary: to return a list of predictions, if true, or a final float value

    returns:
    - array of predictions with steps into the future
    - single float if high or low risk
    '''
    
    # Get the transition matrix from trained model
    transMatrix = model.transmat_
    # We only care about the last value in hidden
    prediction = model.predict(observationList)[-1]

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

    if binary :
        return nextSequence
    else:
        return -6.001 if nextSequence[-1] == 0 else -5.34

def predictAndScore(model, observations, outcomes, steps=1, score = True, verbose = False, binary = False):
    '''
    Predicts future risk based on a list of observations and can also give a score
    inputs: 
    - model: trained HMM model
    - observations: sequence of high risk, low risk (array)
    - outcomes: high risk or low risk at TOC corresponding to each element in observations (array)
    - steps: how far in the future to predict (int)
    - score: use average scoring or not (bool)

    outputs:
    - if scoring is enabled: series of future binary predictions, score for one step and score for last step
    - if not, then just the future predictions either binary or "real"

    '''
    scoreNext = 0
    scoreLast = 0
    futurePredictions = []

    # gets prediction
    for i, observation in enumerate(observations):
        futurePrediction = predictNext(model, observation, steps = steps, binary=binary)
        if verbose:
            print(f"Predicted: {model.predict(observation)}")
            print(f'next prediction: {futurePrediction}')
        
        if score and binary:
            #if predicted is correct, add point
            if futurePrediction[0] == outcomes[i][0]:
                scoreNext += 1

            if futurePrediction[-1] == outcomes[i][-1]:
                scoreLast += 1

        futurePredictions.append(futurePrediction)
    
    # average to get percentage correct
    scoreNext = scoreNext / len(observations) * 100
    scoreLast = scoreLast/ len(observations) * 100

    if score: 
        return futurePredictions, scoreNext, scoreLast
    # if scoring is not needed, as it will be done later
    else: return futurePredictions

def averagePredictions(model, observation, steps=1, avTimes=1):
    '''
    Function that makes multiple future predictions and averages them out for one prediction output
    input:
    - model: trained HMM model
    - observation: list of a single observation to predict
    - steps: how many steps in the future the prediction should be
    - avTimes: how many predictions to average to get final prediction

    output:
    - array of rounded average of multiple prediction arrays
    '''
    predictions = []
    for i in range(avTimes):
        predictions.append(predictNext(model, observation, steps = steps))
    
    average = np.mean(np.array(predictions), axis = 0)
    roundedAverage = np.round(average).astype(int)

    return roundedAverage
