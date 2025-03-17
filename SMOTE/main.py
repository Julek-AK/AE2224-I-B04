import pandas as pd
import scipy
import numpy as np


def pandas_data_frame_creation ():
    train_df = pd.read_csv("DataSets/train_data.csv")
    test_df = pd.read_csv("DataSets/test_data.csv")
    return train_df, test_df

def filter_by_risk(df, risk):
    event_ids_to_remove = df.groupby('event_id')['risk'].max() < risk
    valid_event_ids = event_ids_to_remove[event_ids_to_remove == False].index
    return df[df['event_id'].isin(valid_event_ids)]

def sort_by_mission_id(df):
        return df.sort_values(by = ['event_id', 'time_to_tca'], ascending=[1, 0])

train_df, test_df = pandas_data_frame_creation()

filtered_train_df = filter_by_risk(train_df, -4.0)

sorted_train_df = sort_by_mission_id(filtered_train_df)

print(sorted_train_df.head(55))