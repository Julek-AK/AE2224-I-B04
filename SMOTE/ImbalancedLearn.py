from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt

def prepare_data_for_smote(clustered_events):
    """
    Convert clustered data into a format suitable for SMOTE (oversampling).
    This function assumes you want to balance the clusters based on class frequency.
    """
    X = []
    y = []

    # Flatten the clustered events into a 2D array (samples x features)
    for cluster_id, cluster_data in clustered_events.items():
        for series in cluster_data:
            # Flatten the event (risk values)
            X.append(series)
            y.append(cluster_id)  # Assign the cluster label to this event

    X = np.array(X)
    y = np.array(y)

    return X, y

def apply_smote(X, y):
    """
    Apply SMOTE to balance the dataset by oversampling the minority class.
    """
    smote = SMOTE(sampling_strategy="auto", random_state=42)  # "auto" balances based on class distribution
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled

# Assuming you have clustered_events from your clustering function
X, y = prepare_data_for_smote(clustered_events)

# Apply SMOTE to oversample the minority class
X_resampled, y_resampled = apply_smote(X, y)
