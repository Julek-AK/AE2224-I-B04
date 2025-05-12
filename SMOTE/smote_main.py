import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm


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

def build_event_sequences(df: pd.DataFrame,
                          threshold: float = -6.0
                         ) -> tuple[list[np.ndarray], np.ndarray, list]:
    X_seq, y, ids = [], [], []
    for eid, grp in df.groupby('event_id'):
        arr = grp[['risk','time_to_tca']].to_numpy()
        label = 'high_risk' if (arr[:,0] >= threshold).any() else 'low_risk'
        X_seq.append(arr)
        y.append(label)
        ids.append(eid)
    return X_seq, np.array(y), ids

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

def dtw_path(s: np.ndarray, t: np.ndarray):
    """
    Compute the DTW alignment path between two sequences s and t.
    Each is an array of shape (n_timesteps, n_features).
    Returns a list of index pairs (i, j) that align s[i] with t[j].
    """
    n, m = len(s), len(t)
    # Initialize cost matrix with infinities
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    # Fill cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = np.linalg.norm(s[i - 1] - t[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    # Backtrack to build the warping path
    path: List[Tuple[int, int]] = []
    i, j = n, m
    while (i > 0) or (j > 0):
        path.append((i - 1, j - 1))
        # Determine next step
        moves = []
        if i > 0 and j > 0:
            moves.append((cost[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            moves.append((cost[i - 1, j], i - 1, j))
        if j > 0:
            moves.append((cost[i, j - 1], i, j - 1))
        _, i, j = min(moves, key=lambda x: x[0])
    path.reverse()
    return path

def custom_sequence_smote(
    X_seq: List[np.ndarray],
    y: np.ndarray,
    k_neighbors: int = 3,
    sampling_strategy: float = 0.5,
    random_state: int = 42
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Sequence-SMOTE with progress bars for long-running loops.
    """
    np.random.seed(random_state)
    high_idx = np.where(y == 'high_risk')[0]
    low_count = int(np.sum(y == 'low_risk'))
    high_count = len(high_idx)

    if sampling_strategy <= 0 or sampling_strategy >= 1:
        raise ValueError("sampling_strategy must be in (0, 1)")
    target_high = int(np.round(low_count * sampling_strategy / (1.0 - sampling_strategy)))
    n_synth = max(0, target_high - high_count)
    if n_synth == 0 or high_count < 2:
        return X_seq, y

    k = min(k_neighbors, high_count - 1)

    # Precompute DTW distances with progress bar
    dist = np.zeros((high_count, high_count))
    for i in tqdm(range(high_count), desc="Computing DTW distance rows"):
        for j in range(i + 1, high_count):
            path = dtw_path(X_seq[high_idx[i]], X_seq[high_idx[j]])
            dsum = sum(np.linalg.norm(X_seq[high_idx[i]][p] - X_seq[high_idx[j]][q]) for p, q in path)
            dist[i, j] = dist[j, i] = dsum

    # Generate synthetic sequences with progress bar
    synth_seqs: List[np.ndarray] = []
    per_original = int(np.ceil(n_synth / high_count))
    total_iterations = high_count * per_original
    pbar = tqdm(total=total_iterations, desc="Generating synthetic sequences")
    for idx_pos, seq_idx in enumerate(high_idx):
        neighbors = high_idx[np.argsort(dist[idx_pos])[1 : k + 1]]
        for _ in range(per_original):
            if len(synth_seqs) >= n_synth:
                break
            nbr_idx = np.random.choice(neighbors)
            s1, s2 = X_seq[seq_idx], X_seq[nbr_idx]
            path = dtw_path(s1, s2)
            aligned1 = np.array([s1[i] for i, _ in path])
            aligned2 = np.array([s2[j] for _, j in path])
            lam = np.random.rand()
            synth = aligned1 + lam * (aligned2 - aligned1)
            synth_seqs.append(synth)
            pbar.update(1)
        if len(synth_seqs) >= n_synth:
            break
    pbar.close()

    X_res = X_seq + synth_seqs
    y_res = np.concatenate([y, np.array(['high_risk'] * len(synth_seqs))])
    return X_res, y_res

def plot_sequence_length_histogram(df_raw, df_bal, synthetic_prefix="synthetic_"):
    """
    Plot histogram of CDM counts per event for raw vs. synthetic events.
    """
    # Raw sequence lengths
    raw_counts = df_raw.groupby('event_id').size()
    # Synthetic sequence lengths
    syn_ids = [eid for eid in df_bal['event_id'].unique() if str(eid).startswith(synthetic_prefix)]
    syn_counts = df_bal[df_bal['event_id'].isin(syn_ids)].groupby('event_id').size()

    plt.figure()
    plt.hist(raw_counts, bins=30, alpha=0.7, label='Raw events')
    plt.hist(syn_counts, bins=30, alpha=0.7, label='Synthetic events')
    plt.xlabel('Number of CDMs per event')
    plt.ylabel('Count of events')
    plt.title('Distribution of Sequence Lengths')
    plt.legend()
    plt.show()

def plot_example_trajectories(df_raw, df_bal, n_examples=3, synthetic_prefix="synthetic_"):
    """
    Overlay example risk vs time-to-TCA curves for a few real and synthetic high-risk events.
    """
    # Select example real high-risk event IDs
    real_ids = df_raw[df_raw['risk_label'] == 'high_risk']['event_id'].unique()[:n_examples]
    # Select example synthetic high-risk event IDs
    syn_ids = [eid for eid in df_bal['event_id'].unique()
               if str(eid).startswith(synthetic_prefix)
               and df_bal[df_bal['event_id'] == eid]['risk_label'].iat[0] == 'high_risk'][:n_examples]

    plt.figure()
    # Plot real events
    for eid in real_ids:
        grp = df_raw[df_raw['event_id'] == eid].sort_values('time_to_tca')
        plt.plot(grp['time_to_tca'], grp['risk'], linestyle='-', label=f'Real {eid}')
    # Plot synthetic events
    for eid in syn_ids:
        grp = df_bal[df_bal['event_id'] == eid].sort_values('time_to_tca')
        plt.plot(grp['time_to_tca'], grp['risk'], linestyle='--', label=f'Synth {eid}')

    plt.xlabel('Time to TCA (days)')
    plt.ylabel('Log10 Risk')
    plt.title('Example High-Risk Event Trajectories')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 1) Load & preprocess
    train_df, _ = pandas_data_frame_creation()
    df_raw = (
        train_df
            .pipe(sort_by_mission_id)
            .pipe(clean_data)
        )


    df_raw = label_events_by_risk(df_raw, threshold=-6.0)


    # 3) Build event‐level sequences and labels
    
    X_seq, y, ids = build_event_sequences(df_raw, threshold=-6.0)
    
    #print(X_seq)
    print(f"Built {len(X_seq)} event sequences based on existing events and filtered data:")
    print("  High-risk events:", (y == 'high_risk').sum())
    print("  Low-risk  events:", (y == 'low_risk').sum())


    # 4) Apply our from-scratch SMOTE-TS
    X_res, y_res = custom_sequence_smote(
        X_seq,
        y,
        k_neighbors=3,
        sampling_strategy=0.4,
        random_state=42
    )

    print(f"After SMOTE-TS, total events: {len(X_res)}")
    print("  High-risk events:", (y_res == 'high_risk').sum())
    print("  Low-risk  events:", (y_res == 'low_risk').sum())
    print("Smote Finished RUNNING!")


    # 5) Flatten back into a CDM‐level DataFrame
    df_balanced = flatten_sequences_to_df(
        X_res,
        y_res,
        original_ids=ids,
        threshold=-6.0
    )
    df_balanced.to_csv(f"./DataSets/SMOTE_data.csv", index=False)

    # 6) Quick sanity‐check on the flattened DataFrame
    print("Balanced CDM‐level DataFrame:")
    print("  Total rows (CDMs):", len(df_balanced))
    print("  Unique events:", df_balanced['event_id'].nunique())
    print("  Event label counts:")
    


