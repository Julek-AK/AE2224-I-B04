"""LSTM tryout"""
#import DataReading as DR
import torch 
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt 


#Parameters
inputsize = 4 #The number of expected features in each input time step
hiddensize = 32 #Tunable, too high: overfitting, too low: underfitting
sequenceLength = 18 
numLayers = 1 #the amount of stacked LSTM 
LearningRate = 0.001
numEpochs = 20
#dataset = DR.readData3("train")

#Adapt and understand the chatGPT solution so that it works 
