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

def clean_data(df):
    """Removes any rows containing NaN values."""
    return df.dropna()

train_df, test_df = pandas_data_frame_creation()

filtered_train_df = filter_by_risk(train_df, -4.0)

sorted_train_df = sort_by_mission_id(filtered_train_df)
cleaned_data = clean_data(sorted_train_df)

print(cleaned_data[['event_id', 't_sigma_r', 'risk']].head(50))

#print(cleaned_data.head(50))