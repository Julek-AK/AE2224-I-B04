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


def convert_to_binary_risk(row, risk_threshhold=-7):
    if row['risk'] > risk_threshhold:
        return 1
    else:
        return 0




if __name__ == '__main__':
    csv = pd.read_csv(r"DataSets\train_data.csv")

    csv = cleanup(csv)

    # Additional removals
    csv = csv[csv['risk'] != -30]

    # Cut un-needed columns
    csv = csv.iloc[:, :4]
    
    threshold = -6





