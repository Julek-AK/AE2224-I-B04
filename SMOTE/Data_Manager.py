import pandas as pd
import numpy as np

class Data_Manager:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.train_df = None
        self.test_df = None

    def load_data(self):
        self.train.df = pd.read_csv(self.train_file):
    
    def filter_by_risk(self,risk_threshold):
        event_ids_to_keep = self.train_df.groupby('event_id')['risk'].max() <= risk_threshold
        valid_event_ids = event_ids_to_keep[event_ids_to_keep].index
        self.train_df = self.train_df[self.train_df['event_id'].isin(valid_event_ids)]
        
    def sort_by_event_id_time_to_tca(self):
        self.train_df = self.train_df.sort_values(by = ['event_id', 'time_to_tca'], ascending =[1,0])
    def clean_data(self):
        """Applies filtering and removes NaN values."""
        conditions = [
            ('t_sigma_r', 20),
            ('c_sigma_r', 1000),
            ('t_sigma_t', 2000),
            ('c_sigma_t', 100000),
            ('t_sigma_n', 10),
            ('c_sigma_n', 450)
        ]
        
        for col, threshold in conditions:
            self.train_df = self.train_df[self.train_df[col] <= threshold]

        self.train_df = self.train_df.dropna()
        print("Data cleaned successfully!")

    def create_event_dict(self):
        """Creates a dictionary where keys are event IDs and values are 2D numpy arrays [Risk, Time_to_TCA]."""
        event_dict = {
            event_id: group[['risk', 'time_to_tca']].to_numpy()
            for event_id, group in self.train_df.groupby('event_id')
        }
        return event_dict