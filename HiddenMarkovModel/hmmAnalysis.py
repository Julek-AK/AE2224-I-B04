import pomegranate
from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM
import numpy as np

#The probability of a given hidden state (riskHighHidden, riskLowHidden) going to some output (0, 1) --> emission matrix
riskHighHidden = Categorical ([[0.5, 0.5]])
#So this means that the probability of hidden HR outputting hr or lr
riskLowHidden = Categorical ([[0.8, 0.2]])

#Create the model
model = DenseHMM(verbose = True)

#Add the initial distributions to it
model.add_distributions([riskHighHidden, riskLowHidden])

#Transition matrix using edges
model.add_edge(model.start, riskHighHidden, 0.6)  # Start in state A with probability 0.6
model.add_edge(model.start, riskLowHidden, 0.4)  # Start in state B with probability 0.4
model.add_edge(riskHighHidden, riskHighHidden, 0.7)  # Transition from A to A with probability 0.7
model.add_edge(riskHighHidden, riskLowHidden, 0.3)  # Transition from A to B with probability 0.3
model.add_edge(riskLowHidden, riskHighHidden, 0.4)  # Transition from B to A with probability 0.4
model.add_edge(riskLowHidden, riskLowHidden, 0.6) #Transition from B to B with probability 0.6
model.add_edge(riskHighHidden, model.end, 0.1)  # End in state A with probability 0.1
model.add_edge(riskLowHidden, model.end, 0.1)  # End in state B with probability 0.1


#Prepare training set
trainingSequences = [
    np.array([0, 1, 0, 1, 0]),
    np.array([1, 0, 1, 0, 1]),
    np.array([0, 0, 1, 1, 0])
]

X = [np.array([[x] for x in sequence]) for sequence in trainingSequences]

model.fit([X])

testSequence = np.array([0, 0, 0, 1])
testX = [np.array([[x] for x in testSequence])]

yHat = model.predict(testX)

print("sequence: {}".format(testSequence))
print("hmm pred: {}".format(yHat))
