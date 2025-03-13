import pandas as pd
import scipy
import numpy as np
df = pd.read_csv("DataSets/train_data.csv")
print(len(df))
data_np = df.to_numpy()
no_rows = data_np.shape[0]
#print(no_rows)

#print(df.columns.to_list())
