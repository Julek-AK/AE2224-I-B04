import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from imbalanced_ts.over_sampling import SMOTE_TS



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

def build_event_sequences(df: pd.DataFrame,
                          threshold: float = -6.0
                         ) -> tuple[list[np.ndarray], np.ndarray, list]:
    X_seq, y, ids = [], [], []
    for eid, grp in df.groupby('event_id'):
        arr = grp[['risk','time_to_tca']].to_numpy()
        label = 'high_risk' if (arr[:,0] <= threshold).any() else 'low_risk'
        X_seq.append(arr)
        y.append(label)
        ids.append(eid)
    return X_seq, np.array(y), ids

def apply_sequence_smote(X_seq: list[np.ndarray],
                         y: np.ndarray,
                         k_neighbors: int = 3,
                         random_state: int = 42
                        ) -> tuple[list[np.ndarray], np.ndarray]:
    sm = SMOTE_TS(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = sm.fit_resample(X_seq, y)
    return X_res, y_res

def flatten_sequences_to_df(X_res: list[np.ndarray],
                            y_res: np.ndarray,
                            original_ids: list,
                            threshold: float = -6.0
                           ) -> pd.DataFrame:
    orig_map = {
        tuple(arr.flatten()): eid
        for eid, arr in zip(original_ids, X_res[:len(original_ids)])
    }

    rows = []
    syn_count = 0
    for arr, label in zip(X_res, y_res):
        key = tuple(arr.flatten())
        if key in orig_map:
            eid = orig_map[key]
        else:
            syn_count += 1
            eid = f"synthetic_{syn_count}"
        for risk_val, ttc_val in arr:
            rows.append({
                'event_id':    eid,
                'risk':        risk_val,
                'time_to_tca': ttc_val,
                'risk_label':  label
            })
    return pd.DataFrame(rows)

df3 = (
    train_df
        .pipe(sort_by_mission_id)
        .pipe(clean_data)
)
print(df3.shape, df3.head(20))