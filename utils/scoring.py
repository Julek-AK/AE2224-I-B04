"""
Implements the ESA scoring method for benchmarking models

Currently, it uses the truncated test_data.csv, treating the CDM at 2 days as final, and the previous ones as 
those used for predictions
"""

# External imports
import numpy as np
from sklearn import metrics
import pandas as pd
import os

# Internal imports



def create_test_data(data_filename):
    # Generate test_data used for benchmarking
    assert os.path.exists(rf"Datasets\\{data_filename}"), "Generate test data before benchmarking!"

    test_data = pd.read_csv(rf"DataSets\\{data_filename}", usecols=[0,1,3])
    test_data.dropna(inplace=True)
    test_data.drop_duplicates(inplace=True)


    clean_test_data = pd.DataFrame(columns=['event_id', 'true_risk'])
    for event_id, df in test_data.groupby('event_id'):

        min_tca_time = df['time_to_tca'].min()
        row = df[df['time_to_tca'] == min_tca_time]
        risk = float(row['risk'].iloc[0])
        
        new_row = pd.DataFrame([{'event_id': event_id, 'true_risk': risk}])
        clean_test_data = pd.concat([clean_test_data, new_row], ignore_index=True)

    return clean_test_data



def benchmark(model_prediction, true_data="test_data_shifted.csv", beta=2):
    """
    model_prediction: dataframe with columns 'event_id' and 'predicted_risk'
    IMPROTANT this is NOT a binary classifier of high/low risk, but an actual numerical value

    test_data: dataframe with columns 'event_id' and 'true_risk'
    
    method used is following the ESA challenge
    https://link.springer.com/article/10.1007/s42064-021-0101-5
    """ 

    print("Benchmarking model...")

    # Sort and clean up
    model_prediction.sort_values(by='event_id', ascending=True, inplace=True)
    model_prediction.drop_duplicates(inplace=True)
    true_data = create_test_data(true_data)
    true_data.sort_values(by='event_id', ascending=True, inplace=True)
    true_data.drop_duplicates(inplace=True)

    # Correct inconsistent lengths
    if len(model_prediction) != len(true_data):
        print("Correcting data size mismatch by pruning events from true data")

        common_events = model_prediction['event_id'].unique()
        true_data = true_data[true_data['event_id'].isin(common_events)]

        true_data = true_data.reset_index(drop=True)
        model_prediction = model_prediction.reset_index(drop=True)
    print(f"Evaluating {len(model_prediction)} entries...")
    
    # Get the numpy arrays
    predicted_risk = model_prediction['predicted_risk'].to_numpy()
    true_risk = true_data['true_risk'].to_numpy()
    predicted_risk_binary = predicted_risk > -6
    true_risk_binary = true_risk > -6

    # Risk clipping
    predicted_risk[predicted_risk < -6] = -6.001

    # Compute metrics
    precision = metrics.precision_score(true_risk_binary, predicted_risk_binary)
    recall = metrics.recall_score(true_risk_binary, predicted_risk_binary)
    F_score = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

    N = np.count_nonzero(true_risk_binary)
    high_risk_squared_errors = true_risk_binary * np.square(predicted_risk - true_risk)
    MSE_HR = np.sum(high_risk_squared_errors) / N
    L_score = MSE_HR/F_score

    print(f"F-score: {F_score:.3g}")
    print(f"High risk MSE: {MSE_HR:.3g}")
    print(f"Final Loss Score: {L_score:.3g}")

    return F_score, MSE_HR, L_score





if __name__ == "__main__":
    # Naive baseline
    filename = "train_data.csv"
    test_data = pd.read_csv(rf"DataSets\{filename}", usecols=[0, 1, 3])
    test_data.dropna(inplace=True)
    test_data.drop_duplicates(inplace=True)

    clean_test_data = pd.DataFrame(columns=['event_id', 'true_risk'])
    for event_id, df in test_data.groupby('event_id'):

        min_tca_time = df['time_to_tca'].min()
        df2 = df[df['time_to_tca'] != min_tca_time]

        try:
            second_min_tca_time = df2['time_to_tca'].min()
            row = df2[df2['time_to_tca'] == second_min_tca_time]
        except IndexError:
            row = df[df['time_to_tca'] == min_tca_time]
        risk = float(df['risk'].iloc[0])

        new_row = pd.DataFrame([{'event_id': event_id, 'true_risk': risk}])
        clean_test_data = pd.concat([clean_test_data, new_row], ignore_index=True)

    naive_baseline = clean_test_data.rename(columns={'event_id': 'event_id', 'true_risk': 'predicted_risk'})

    benchmark(naive_baseline, true_data=f"{filename}")