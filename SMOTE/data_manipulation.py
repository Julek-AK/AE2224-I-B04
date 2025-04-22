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
    df = df[df['t_sigma_r'] <= 20]
    df = df[df['c_sigma_r'] <= 1000]
    df = df[df['t_sigma_t'] <= 2000]
    df = df[df['c_sigma_t'] <= 100000]
    df = df[df['t_sigma_n'] <= 10]
    df = df[df['c_sigma_n'] <= 450]
    df = df.dropna()
    return df

def create_event_dict(df):
    event_dict = {}
    for event_id, group in df.groupby('event_id'):
        event_array = group[['risk', 'time_to_tca']].to_numpy()  
        event_dict[event_id] = event_array
    return event_dict


train_df, test_df = pandas_data_frame_creation()
filtered_train_df = filter_by_risk(train_df, -4.0)
sorted_train_df = sort_by_mission_id(filtered_train_df)
cleaned_data = clean_data(sorted_train_df)


print(cleaned_data.shape)
event_dict = create_event_dict(cleaned_data)

def event_with_extreme_cdms(event_dict):
    max_event = None
    max_cdms = -1  # start with a very low number
    min_event = None
    min_cdms = float('inf')  # start with a very high number
    
    for event_id, cdm_array in event_dict.items():
        num_cdms = cdm_array.shape[0]
        if num_cdms > max_cdms:
            max_cdms = num_cdms
            max_event = event_id
        if num_cdms < min_cdms:
            min_cdms = num_cdms
            min_event = event_id
    return max_event, max_cdms, min_event, min_cdms

'''
max_event, max_cdms, min_event, min_cdms = event_with_extreme_cdms(event_dict)
print(f"Event {max_event} has the maximum number of CDMs: {max_cdms}")
print(f"Event {min_event} has the minimum number of CDMs: {min_cdms}")
'''
