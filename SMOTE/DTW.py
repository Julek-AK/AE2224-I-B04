import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tslearn.clustering import TimeSeriesKMeans

# Example: High-Risk CDMs (Pc evolutions)
high_risk_events = np.array([
    [0.50, 0.48, 0.45, 0.40, 0.35],  # Event 1
    [0.55, 0.53, 0.50, 0.45, 0.40],  # Event 2
    [0.52, 0.49, 0.47, 0.42, 0.38],  # Event 3
    [0.54, 0.51, 0.48, 0.43, 0.39],  # Event 4
    [0.80, 0.78, 0.75, 0.70, 0.65],  # Event 5 (Different trend)
    [0.85, 0.83, 0.80, 0.76, 0.72],  # Event 6 (Different trend)
])

# Define number of clusters (adjust based on dataset)
num_clusters = 2

# Apply DTW-based K-Means clustering
kmeans_dtw = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", random_state=42)
cluster_labels = kmeans_dtw.fit_predict(high_risk_events)

# Print cluster assignments
for i, label in enumerate(cluster_labels):
    print(f"Event {i+1} assigned to Cluster {label}")

# Organize events by cluster
clustered_events = {i: [] for i in range(num_clusters)}
for i, label in enumerate(cluster_labels):
    clustered_events[label].append(high_risk_events[i])

# Convert lists to NumPy arrays
clustered_events = {k: np.array(v) for k, v in clustered_events.items()}

