import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
import warnings
# Suppress FutureWarnings from sklearn
warnings.simplefilter(action='ignore', category=FutureWarning)
from kneed import KneeLocator


def find_optimal_k(data, k_min=1, k_max=10):
    """Uses the Elbow Method to determine the best k for DTW clustering without KneeLocator."""
    distortions = []
    k_values = list(range(k_min, k_max + 1))

    for k in k_values:
        model = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=42, n_init=10)
        model.fit(data)
        distortions.append(model.inertia_)  # Sum of squared DTW distances

    # Ensure all k-values appear on the X-axis
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, distortions, marker='o')
    plt.xticks(k_values)  # Fix the X-axis
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Distortion (DTW Sum of Squared Distances)")
    plt.title(f"Elbow Method for DTW Clustering (k_max={k_max})")
    plt.grid()
    plt.show(block=True)

    # **Fix: If distortions are constant, default to k=3**
    if len(set(distortions)) == 1:
        print("Warning: All distortions are identical. Defaulting to k=3.")
        return 3

    # **Fix: Manual heuristic for finding elbow (instead of KneeLocator)**
    drop_rates = np.diff(distortions) / distortions[:-1]  # Relative decrease
    elbow_index = np.argmax(drop_rates < 0.1) + 1  # Find first small drop

    optimal_k = k_values[elbow_index] if elbow_index < len(k_values) else k_max
    print(f"Optimal number of clusters: {optimal_k}")

    return optimal_k


def cluster_high_risk_events(data, num_clusters):
    """Clusters high-risk events using DTW-based K-Means."""
    kmeans_dtw = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", random_state=42)
    cluster_labels = kmeans_dtw.fit_predict(data)

    # Print cluster assignments
    for i, label in enumerate(cluster_labels):
        print(f"Event {i+1} assigned to Cluster {label}")

    # Organize events by cluster
    clustered_events = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(cluster_labels):
        clustered_events[label].append(data[i])

    # Convert lists to NumPy arrays
    clustered_events = {k: np.array(v) for k, v in clustered_events.items()}
    
    return clustered_events

# Example high-risk events dataset
"""""
high_risk_events = np.array([
    # Cluster 0: Flat sequences
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  
    [0.6, 0.6, 0.6, 0.6, 0.6, 0.6],  
    [0.4, 0.4, 0.4, 0.4, 0.4, 0.4],  

    # Cluster 1: Steady decrease
    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],  
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],  
    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],  

    # Cluster 2: Oscillating values
    [1.0, 0.5, 1.0, 0.5, 1.0, 0.5],  
    [0.9, 0.4, 0.9, 0.4, 0.9, 0.4],  
    [0.8, 0.3, 0.8, 0.3, 0.8, 0.3],  
])
"""
high_risk_events = np.array([
    # Cluster 0: All ones (constant high risk)
    [1, 1, 1, 1, 1, 1],  
    [1, 1, 1, 1, 1, 1],  
    [1, 1, 1, 1, 1, 1],  

    # Cluster 1: All zeros (constant low risk)
    [0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0],  

    # Cluster 2: Immediate Drop (sudden failure)
    [1, 0.8, 0.5, 0.2, 0.1, 0.0],  
    [1, 0.85, 0.6, 0.3, 0.15, 0.0],  
    [1, 0.9, 0.7, 0.4, 0.2, 0.0],  

    # Cluster 3: Alternating (oscillating risk)
    [1, 0, 1, 0, 1, 0],  
    [0, 1, 0, 1, 0, 1],  
    [1, 0, 1, 0, 1, 0],  
])
print("New dataset shape:", high_risk_events.shape)

# Step 1: Run elbow method to get best k
optimal_k = find_optimal_k(high_risk_events)

# Step 2: Cluster events using the optimal k
clustered_events = cluster_high_risk_events(high_risk_events, optimal_k) #4 is optimal_k

