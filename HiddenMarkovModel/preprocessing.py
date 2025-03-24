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
    csv = csv.dropna()

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


def generate_hmm_data(filename, risk_threshold=-6, traindata= True, verbose=False):
    """
    Processes the raw dataset into a format accepted by hidden markov models
    the resulting dataframe has columns:

    event_id: same identifier as in original data

    observations: a tuple of 15 values (3 per day, up until 2 days before TCA), 0 refers to low risk, and 1 to high risk of collision
    if event_id has too many CDMs, the oldest ones are removed, if there are too few CDMs, new datapoints are padded until 15

    outcome: a tuple of two values. The first one is the risk state in the next CDM after the 2 day cut-off. The second value is the final
    risk state in the CDM closest to TCA
    """

    # TODO make work for test data
    if filename == "test_data.csv":
        raise NotImplementedError("test data doesn't contain predictions, this is yet to be implemented")

    csv = pd.read_csv(rf"DataSets\{filename}")
    csv = cleanup(csv)

    # Additional cleanups
    csv = csv[csv['time_to_tca'] > 0]

    # Cut un-needed columns and duplicate rows
    csv = csv.iloc[:, :4]
    csv = csv.drop_duplicates()
    
    n_events = csv['event_id'].nunique()
    if verbose:
        print(f"There are {n_events} events to process")

    # Prepare output DataFrame
    data = pd.DataFrame(columns=['event_id', 'observations', 'outcome'])

    # Generate a data line for each event
    for event_id, df in csv.groupby('event_id'):
        if verbose:
            print(f"Processing event {event_id}")

        df.sort_values(by='time_to_tca', ascending=False, inplace=True)

        # Remove events with final collision risk of -30
        if all(df['risk'] == -30):
            print("worked!")
            continue

        # Converts the risk entries into binary determinator of high/low risk
        df['risk'] = df['risk'].apply(lambda x: 0 if x < risk_threshold else 1)

        # Split into observations and predictions based on competition requirement
        observations = df[df['time_to_tca'] > 2]
        predictions = df[df['time_to_tca'] < 2]

        # Remove events with lacking data
        n_obsv = len(observations)
        n_pred = len(predictions)
        if n_obsv == 0 or n_pred == 0:
            if verbose:
                print("insufficient CDMs")
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
                i = (6 - int(np.floor(time)))*3
                while i < 15:
                    if risk_sequence[i] is None:
                        risk_sequence[i] = risk
                        break
                    else:
                        i+=1

            # Pad-left
            for i in range(1, 15):
                if risk_sequence[i] is None:
                    risk_sequence[i] = risk_sequence[i-1]

            # Pad-right
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

    data.to_csv(f"./DataSets/HMM_{filename}")
    print(f"Data saved as HMM_{filename}")
            

if __name__ == '__main__':
    generate_hmm_data("train_data.csv")
    # generate_hmm_data("test_data.csv", verbose=True)
