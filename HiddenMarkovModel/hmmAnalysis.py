import numpy as np
from hmmLearnAlgorithm import idealPrediction, predictAndScore, predictNext, averagePredictions
from splitData import splitSet, formatData
import pandas as pd
from matplotlib import pyplot as plt

# import pickle
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.scoring import benchmark


lengths = []

# train set
splitSet("HMM_SMOTE_data3.csv", 0.1)
observations, lengths = formatData("HMM_SMOTE_data3.csv")
squishedObservations = np.concatenate(observations)

# validation set
valObservations, valOutcomes, testIDs = formatData("HMM_validation_set.csv", test=True)

# test set
# testObservations, testOutcomes, testIDs = formatData("HMM_test_data_shifted.csv", test=True)

# train model with train set
model1 = idealPrediction(squishedObservations, lengths, 30)
prediction= predictAndScore(model1, valObservations, valOutcomes, steps = 1, score = False, binary = False)

predictionPD = pd.DataFrame({'predicted_risk': prediction, 'event_id': testIDs})
testOutcomePD = pd.DataFrame({'true_risk': valOutcomes, 'event_id': testIDs})

F_score, MSE_HR, L_score = benchmark(predictionPD, true_data='SMOTE_data3.csv')

# print(f"fscore: {F_score}")
# print(f"mse: {MSE_HR}")
# print(f"Lscore: {L_score}")

def graph_data(model_prediction, true_data):
    # Prune excessive data
    model_prediction['event_id'] = model_prediction['event_id'].astype(str)
    true_data['event_id'] = true_data['event_id'].astype(str)

    common_events = np.intersect1d(
        model_prediction['event_id'].unique(),
        true_data['event_id'].unique())
    
    model_filtered = model_prediction[model_prediction['event_id'].isin(common_events)]
    true_filtered = true_data[true_data['event_id'].isin(common_events)]

    model_sorted = model_filtered.sort_values('event_id').reset_index(drop=True)
    true_sorted = true_filtered.sort_values('event_id').reset_index(drop=True)

    outputs = model_sorted['predicted_risk'].tolist()
    targets = true_data['risk'].tolist()
    # targets = [-6.001 if x[0] == 0 else -5.34 for x in targets]
    steps = [i for i in range(len(outputs))]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.title("Test Results", fontsize=16)
    plt.scatter(steps, outputs, color='blue', label='Predicted', alpha=0.7)
    plt.scatter(steps, targets, color='red', label='Target', alpha=0.7)

    for i in range(len(outputs)):
        plt.plot([steps[i], steps[i]], [outputs[i], targets[i]], color='gray', linestyle='--', alpha=0.5)

    plt.axhline(y=-6, color='green', linestyle='-', label='Threshold (-6)', linewidth=2)

    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    plt.show()

# graph_data(predictionPD, pd.read_csv("DataSets\\SMOTE_data.csv"))