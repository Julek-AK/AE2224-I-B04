"""
Converts CDM data into a string of 0 and 1 representing a sequence of whether the CDM
labels the event collision risk as high or low.
This will be later used to train the HMM

"""

# External Imports
import pandas as pd
import numpy as np

# Internal Imports
...

csv = pd.read_csv(r"DataSets\train_data.csv")
csv = csv.iloc[:, 0:4]
print(csv)


# Iterate over each event
n_events = csv["event_id"].max()
print(n_events)