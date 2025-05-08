import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pandas_data_frame_creation ():
    train_df = pd.read_csv("DataSets/train_data.csv")
    return train_df

def clean_data(df):
    df = df[df['t_sigma_r'] <= 20]
    df = df[df['c_sigma_r'] <= 1000]
    df = df[df['t_sigma_t'] <= 2000]
    df = df[df['c_sigma_t'] <= 100000]
    df = df[df['t_sigma_n'] <= 10]
    df = df[df['c_sigma_n'] <= 450]
    df = df.dropna()
    return df

def sort_by_mission_id(df):
        return df.sort_values(by = ['event_id', 'time_to_tca'], ascending=[1, 0])

def label_events_by_risk(df: pd.DataFrame, threshold: float = -6.0) -> pd.DataFrame:

    """
    Labels each CDM row in df as 'high_risk' or 'low_risk' according to
    whether its event's maximum log10(risk) ≥ threshold.

    Parameters:
      df        – DataFrame with at least ['event_id','risk'] columns
      threshold – log10-risk cutoff (default: -6.0)

    Returns:
      A copy of df with a new column 'risk_label'.
    """
    df = df.copy()

    # 1) Compute each event’s maximum log-risk, broadcast to its rows
    df['event_max_risk'] = (
        df.groupby('event_id')['risk']
          .transform('max')
    )

    # 2) Assign: high_risk if any CDM’s risk ≥ threshold
    df['risk_label'] = np.where(
        df['event_max_risk'] >= threshold,
        'high_risk',
        'low_risk'
    )

    # 3) Drop helper
    #df.drop(columns=['event_max_risk'], inplace=True)

    return df

def data_frame_statistics(df):
    total_cdms = len(df)
    total_num_events = df['event_id'].nunique()
    event_labels = df.groupby('event_id')['risk_label'].first()
    no_high_risk_events = event_labels.value_counts()
    low_risk_events = total_num_events - no_high_risk_events
    return total_cdms, total_num_events, no_high_risk_events, low_risk_events

if __name__ == '__main__':
    train_df = pandas_data_frame_creation()
    cleaned_data = clean_data(train_df)
    new_labeled_data = label_events_by_risk(cleaned_data)
    sorted_data = sort_by_mission_id(new_labeled_data)
    total_cdms, total_num_events, no_high_risk_events, low_risk_events = data_frame_statistics(sorted_data)

    print(f'Total CDMS after cleaning = {total_cdms}\n')
    print(f'Total number of events = {total_num_events}\n')
    print(no_high_risk_events)