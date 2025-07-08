"""
Removes outliers, non-functional values and in general cleans the dataset.
Converts CDM data into a string of 0 and 1 representing a sequence of whether the CDM
labels the event collision risk as high or low.
This will be later used to train the HMM
"""

# External Imports
import pandas as pd
import numpy as np

# Internal Imports
...


def cleanup(csv):
    """
    Removes all NaN values, as well as events with physically nonsensical parameters
    Based on kesslerlib, https://github.com/kesslerlib/kessler 
    """

    original_length = len(csv)
    print(f"Starting with {original_length} entries.")
    print("Removing invalid values")

    # Remove NaN values
    csv.dropna(inplace=True)

    # Remove outliers
    # outlier_features = ['t_sigma_r', 't_sigma_t', 't_sigma_n', 't_sigma_rdot', 't_sigma_tdot', 't_sigma_ndot']
    csv = csv[csv['t_sigma_r'] <= 20]
    csv = csv[csv['c_sigma_r'] <= 1000]
    csv = csv[csv['t_sigma_t'] <= 2000]
    csv = csv[csv['c_sigma_t'] <= 100000]
    csv = csv[csv['t_sigma_n'] <= 10]
    csv = csv[csv['c_sigma_n'] <= 450]

    final_length = len(csv)
    print(f"Remaining entries: {final_length}, which is {final_length/original_length:.3g} of the original.")
    return csv


def generate_hmm_data(filename, risk_threshold=-6, smote=False, verbose=False):
    """
    Processes the raw dataset into a format accepted by hidden markov models
    the resulting dataframe has columns:

    event_id: same identifier as in original data

    observations: a tuple of 15 values (3 per day, up until 2 days before TCA), 0 refers to low risk, and 1 to high risk of collision
    if event_id has too many CDMs, the oldest ones are removed, if there are too few CDMs, new datapoints are padded until 15

    outcome: a tuple of two values. The first one is the risk state in the next CDM after the 2 day cut-off. The second value is the final
    risk state in the CDM closest to TCA
    """

    print(f"Processing file {filename}")

    csv = pd.read_csv(rf"DataSets\{filename}")
    if not smote: csv = cleanup(csv)  # Since SMOTE data has been cleaned beforehand

    # Additional cleanups
    csv = csv[csv['time_to_tca'] > 0]
    csv = csv.iloc[:, :4]
    csv = csv.drop_duplicates()
    
    if verbose: print(f"There are {csv['event_id'].nunique()} events to process")

    # Prepare output DataFrame
    data = pd.DataFrame(columns=['event_id', 'observations', 'outcome'])
    data['observations'] = data['observations'].astype('object')

    # Generate a data line for each event
    for event_id, df in csv.groupby('event_id'):
        if verbose: print(f"Processing event {event_id}")

        df.sort_values(by='time_to_tca', ascending=False, inplace=True)

        # Remove events with all collision risks of -30
        if all(df['risk'] == -30):
            if verbose: print("negligible risk")
            continue

        # Converts the risk entries into a binary determinator of high/low risk
        df['risk'] = df['risk'].apply(lambda x: 0 if x < risk_threshold else 1)

        # Split into observations and predictions based on competition requirement
        observations = df[df['time_to_tca'] > 2]
        predictions = df[df['time_to_tca'] < 2]

        # Remove events with lacking data, depending on whether it's train or test
        n_obsv = len(observations)
        n_pred = len(predictions)
        if n_obsv == 0 or n_pred == 0:
            if verbose: print("insufficient CDMs")
            continue
   
        # Prepare risk sequence based on number of observation CDMs
        if n_obsv > 15:
            risks = observations['risk'].tolist()
            risk_sequence = risks[-15:]

        elif n_obsv == 15:
            risk_sequence = observations['risk'].tolist()

        elif n_obsv < 15:
            risks = observations['risk'].tolist()
            times = observations['time_to_tca'].tolist()

            # Distribute known datapoints amongst the 15 required CDMs
            risk_sequence = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
            for risk, time in zip(risks, times):
                i = int((6 - np.floor(time))*3)
                while i < 15:
                    if risk_sequence[i] is None:
                        risk_sequence[i] = risk
                        break
                    else:
                        i+=1

            # Pad right
            for i in range(1, 15):
                if risk_sequence[i] is None:
                    risk_sequence[i] = risk_sequence[i-1]

            # Pad left
            for i in range(2, 16):
                if risk_sequence[-i] is None:
                    risk_sequence[-i] = risk_sequence[-i+1]
            
        risk_sequence = tuple(risk_sequence)

        # Prepare the data used for predicting

        predictions = predictions['risk'].tolist()
        first_prediction = predictions[0]
        final_state = predictions[-1]
        prediction = (first_prediction, final_state)

        # Add to the dataset
        new_row = pd.DataFrame([{'event_id': event_id, 'observations': risk_sequence, 'outcome': prediction}])
        data = pd.concat([data, new_row], ignore_index=True)

    data.to_csv(f"./DataSets/HMM_{filename}", index=False)
    print(f"Data saved as HMM_{filename}")


if __name__ == '__main__':
    # generate_hmm_data("train_data.csv" )
    # generate_hmm_data("train_data.csv", verbose=False)
    generate_hmm_data("SMOTE_data3.csv", smote=True, verbose=True)
