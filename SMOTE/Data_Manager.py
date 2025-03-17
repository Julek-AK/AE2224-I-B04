import pandas as pd
import numpy as np

class Data_Manager:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.train_df = None
        self.test_df = None

    def load_data(self):
        self.train.df = pd.read_csv(self.train_file)
        