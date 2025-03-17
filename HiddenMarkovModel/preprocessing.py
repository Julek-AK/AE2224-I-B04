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

# Setting up threshold for high/low risk events
threshold = -6

# Converts the risk entries into 0 and 1 regarding the risk
csv["risk"] = csv["risk"].apply(lambda x: 0 if x < threshold else 1)

# Print result
print(csv)

grouped_df = csv.groupby("event_id")["risk"].apply(list).reset_index()

print(grouped_df)