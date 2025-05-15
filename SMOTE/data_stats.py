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

def raw_data_stats(df):
    total_cdms = len(df)
    total_num_events = df['event_id'].nunique()
    return total_cdms, total_num_events


def smote_data_stats():
    df = pd.read_csv("DataSets/SMOTE_data_12_05.csv")
    TOTAL_CDMS_WITH_SMOTE =  len(df)
    total_num_events = df['event_id'].nunique()
    is_synthetic = df['event_id'].astype(str).str.startswith("synthetic")
    total_synthetic_cdms = is_synthetic.sum()
    total_synthetic_events = df.loc[is_synthetic, 'event_id'].nunique()
    return TOTAL_CDMS_WITH_SMOTE, total_num_events, total_synthetic_cdms, total_synthetic_events

    
def count_risk_events(df):
    # Group by event_id, and classify each event based on whether any row is high risk
    event_risk = df.groupby('event_id')['risk_label'].apply(
        lambda group: 'high_risk' if (group == 'high_risk').any() else 'low_risk'
    )

    # Count the number of high and low risk events
    risk_counts = event_risk.value_counts()

    # Extract counts with fallback to 0 if the category is missing
    no_high_risk_events = risk_counts.get('high_risk', 0)
    no_low_risk_events = risk_counts.get('low_risk', 0)

    return no_high_risk_events, no_low_risk_events



if __name__ == '__main__':
    #Create raw data_frame
    raw_data = pandas_data_frame_creation()
    #label raw data_frame as high or low risk
    labeled_data_raw_data = label_events_by_risk(raw_data)
    #get the total no of events and total cdms of the raw data
    total_raw_cdms, total_raw_events =  raw_data_stats(labeled_data_raw_data)
    #Find and count how many events are high or low risk
    NUMBER_OF_HIGH_RISK_EVENTS_RAW_DATA, NUMBER_OF_LOW_RISK_EVENTS_RAW_DATA = count_risk_events(labeled_data_raw_data)
 
    #Print raw data statisctics
    print(f'Raw number of CDMs = {total_raw_cdms}')
    print(f'Total number of RAW events = {total_raw_events}')
    print(f'Number of low risk events in the raw data frame = {NUMBER_OF_LOW_RISK_EVENTS_RAW_DATA}')
    print(f'Number of high risk events in the data frame = {NUMBER_OF_HIGH_RISK_EVENTS_RAW_DATA}\n ')

    TOTAL_CDMS_AFTER_SMOTE, TOTAL_EVENTS_AFTER_SMOTE, TOTAL_SYNTETHIC_CDMS, TOTAL_SYNTHETIC_EVENTS = smote_data_stats()







    #Print SYNTETHIC data statistics
    print(f'Number of CDMS after SMOTE = {TOTAL_CDMS_AFTER_SMOTE}')
    print(f'NUmber of EVENTS after SMOTE =  {TOTAL_EVENTS_AFTER_SMOTE}')
    print(f'Number of Synthetic CDMs = {TOTAL_SYNTETHIC_CDMS}')
    print(f'Number of Synthetic EVENTS = {TOTAL_SYNTHETIC_EVENTS}')