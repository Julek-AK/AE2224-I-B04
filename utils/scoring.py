"""
Implements the ESA scoring method for benchmarking models

Currently, it uses the truncated test_data.csv, treating the CDM at 2 days as final, and the previous ones as 
those used for predictions
"""

# External imports
import numpy as np
from sklearn import metrics
import pandas as pd

# Internal imports



# Generate test_data used for benchmarking
test_data = pd.read_csv(r"DataSets\test_data.csv", usecols=[0,1,3])
test_data.dropna(inplace=True)
test_data.drop_duplicates(inplace=True)

# Correct for missing CDMs from 2 days up to TCA
test_data['time_to_tca'] -= 2

clean_test_data = pd.DataFrame(columns=['event_id', 'true_risk'])
for event_id, df in test_data.groupby('event_id'):

    min_tca_time = df['time_to_tca'].min()
    row = df[df['time_to_tca'] == min_tca_time]
    risk = float(row['risk'].iloc[0])
    
    new_row = pd.DataFrame([{'event_id': event_id, 'true_risk': risk}])
    clean_test_data = pd.concat([clean_test_data, new_row], ignore_index=True)



def benchmark(model_prediction, true_data=clean_test_data, beta=2):
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
    true_data.sort_values(by='event_id', ascending=True, inplace=True)
    true_data.drop_duplicates(inplace=True)

    # Consistency checks
    if len(model_prediction) != len(true_data):
        print("Inconsistent dataset size")
        return
    
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
    high_risk_squared_errors = true_risk_binary * np.pow((predicted_risk - true_risk), 2)
    MSE_HR = np.sum(high_risk_squared_errors) / N
    L_score = MSE_HR/F_score

    print(f"F-score: {F_score:.3g}")
    print(f"High risk MSE: {MSE_HR:.3g}")
    print(f"Final Loss Score: {L_score:.3g}")

    return F_score, MSE_HR, L_score





if __name__ == '__main__':
    # Naive baseline
    test_data = pd.read_csv(r"DataSets\test_data.csv", usecols=[0,1,3])
    test_data.dropna(inplace=True)
    test_data.drop_duplicates(inplace=True)

    # Correct for missing CDMs from 2 days up to TCA
    # test_data['time_to_tca'] -= 2

    clean_test_data = pd.DataFrame(columns=['event_id', 'true_risk'])
    for event_id, df in test_data.groupby('event_id'):

        min_tca_time = df['time_to_tca'].min()
        df2 = df[df['time_to_tca'] != min_tca_time]

        try:
            second_min_tca_time = df2['time_to_tca'].min()
            row = df2[df2['time_to_tca'] == second_min_tca_time]
            risk = float(row['risk'].iloc[0])
        except IndexError:
            row = df[df['time_to_tca'] == min_tca_time]
            risk = float(df['risk'].iloc[0])

        new_row = pd.DataFrame([{'event_id': event_id, 'true_risk': risk}])
        clean_test_data = pd.concat([clean_test_data, new_row], ignore_index=True)

    naive_baseline = clean_test_data.rename(columns={'event_id': 'event_id', 'true_risk': 'predicted_risk'})

    benchmark(naive_baseline)