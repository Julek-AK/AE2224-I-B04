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


def generate_hmm_data(verbose=False):
    csv = pd.read_csv(r"DataSets\train_data.csv")

    csv = cleanup(csv)

    # Additional cleanups
    csv = csv[csv['time_to_tca'] > 0]

    # Cut un-needed columns and duplicate rows
    csv = csv.iloc[:, :4]
    csv = csv.drop_duplicates()
    
    # Converts the risk entries into binary determinator of high/low risk
    threshold = -6
    csv['risk'] = csv['risk'].apply(lambda x: 0 if x < threshold else 1)

    # Print result
    # print(csv)

    n_events = csv['event_id'].nunique()
    if verbose:
        print(f"There are {n_events} events to process")

    # max_n_cdms = csv['event_id'].value_counts().max()
    # print(f"Longest cdm string: {max_n_cdms}")
    # print(csv[csv["event_id"] == csv["event_id"].value_counts().idxmax()])

    # min_n_cdms = csv['event_id'].value_counts().min()
    # print(f"Shortest cdm string: {min_n_cdms}")
    # print(csv[csv["event_id"] == csv["event_id"].value_counts().idxmin()])

    # Preview specific event
    # print(csv[csv['event_id'] == 12779])

    # Prepare output DataFrame
    data = pd.DataFrame(columns=['event_id', 'observations', 'outcome'])


    for event_id, df in csv.groupby('event_id'):
        if verbose:
            print(f"Processing event {event_id}")

        df.sort_values(by='time_to_tca', ascending=False, inplace=True)
        
        observations = df[df['time_to_tca'] > 2]
        predictions = df[df['time_to_tca'] < 2]

        n_obsv = len(observations)
        n_pred = len(predictions)

        # Remove events with lacking data
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

    data.to_csv("./DataSets/HMM_train_data.csv")
    print("Data saved as HMM_train_data.csv")
            

if __name__ == '__main__':
    generate_hmm_data()
