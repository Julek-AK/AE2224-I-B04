import pandas as pd
import scipy
import numpy as np


def pandas_data_frame_creation ():
    train_df = pd.read_csv("DataSets/train_data.csv")
    test_df = pd.read_csv("DataSets/test_data.csv")
    return train_df, test_df

def sort_by_mission_id():
        return train_df.sort_values(by = 'mission_id', ascending=1)

train_df, test_df = pandas_data_frame_creation()
sorted_train_df = sort_by_mission_id()
print(sorted_train_df.head(50))