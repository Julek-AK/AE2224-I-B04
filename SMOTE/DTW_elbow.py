import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
import warnings
# Suppress FutureWarnings from sklearn
warnings.simplefilter(action='ignore', category=FutureWarning)
from kneed import KneeLocator

def find_optimal_k(
    data, 
    k_min=1, 
    k_max=10, 
    drop_threshold=0.05, 
    log_scale=False
):
    """
    1) Computes DTW-based distortion for k in [k_min..k_max].
    2) Uses a threshold on the absolute drop to pick the first 'flattening' k.
    3) Plots two subplots:
       - Distortion vs. k (with optional log scale and Y-axis cutoff).
       - Absolute drop vs. k (with a horizontal threshold line).
    4) Highlights the chosen k in both subplots.
    """
    k_values = list(range(k_min, k_max + 1))
    distortions = []

    # 1) Compute distortions for each k
    for k in k_values:
        model = TimeSeriesKMeans(
            n_clusters=k, metric="dtw", random_state=42, n_init=10
        )
        model.fit(data)
        distortions.append(model.inertia_)

    distortions = np.array(distortions)
    
    # 2) Calculate absolute drops: 
    #    abs_drop[i] = distortions[i] - distortions[i+1]
    abs_drops = []
    for i in range(len(distortions) - 1):
        abs_drop = distortions[i] - distortions[i + 1]
        abs_drops.append(abs_drop)
    abs_drops = np.array(abs_drops)

    # 3) Find the FIRST index where the absolute drop is below drop_threshold
    below_thresh_indices = np.where(abs_drops < drop_threshold)[0]
    if len(below_thresh_indices) > 0:
        # elbow_index is the first time the drop < threshold
        elbow_index = below_thresh_indices[0] + 1
        optimal_k = k_values[elbow_index]
    else:
        # If never below threshold, pick the largest k
        elbow_index = len(k_values) - 1
        optimal_k = k_values[elbow_index]

    # OPTIONAL: Impose a minimum k if you want at least 3, for example:
    # if optimal_k < 3:
    #     optimal_k = 3

    # 4) Plot the results in two subplots
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    
    # Top subplot: Distortion vs. k
    axes[0].plot(k_values, distortions, 'o-', label="Distortion")
    if log_scale:
        axes[0].set_yscale("log")
    axes[0].set_ylabel("Distortion (DTW SSE)")
    axes[0].set_title(f"Elbow Method (Threshold={drop_threshold}), Distortion vs. k")
    axes[0].grid(True)
    axes[0].set_ylim(0, 2)  # Limit the Y-axis for the distortion plot
    
    # Highlight chosen k on distortion plot
    # Find the index of chosen k in k_values:
    chosen_k_index = k_values.index(optimal_k)
    axes[0].plot(
        optimal_k, 
        distortions[chosen_k_index], 
        'ro', 
        markersize=10, 
        label=f"Chosen k = {optimal_k}"
    )
    axes[0].legend()

    # Bottom subplot: Absolute drop vs. k (k_values[1:] since no drop for k=last)
    # We'll plot them at x = k_values[1..], i.e. the "to" cluster count
    drop_x = k_values[1:]
    axes[1].plot(drop_x, abs_drops, 's-', label="Absolute drop")
    # Horizontal threshold line
    axes[1].axhline(drop_threshold, color='r', linestyle='--', label="Threshold")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Absolute Drop")
    axes[1].grid(True)
    axes[1].set_title("Absolute Drop from k to k+1")

    # Highlight the chosen k on the drop plot
    # The chosen k corresponds to the drop from (k-1) to k, 
    # so it is at index (k-1) in abs_drops if k>1
    if optimal_k > k_values[0]:
        highlight_index = optimal_k - k_values[0] - 1  # offset from k_min
        if 0 <= highlight_index < len(abs_drops):
            axes[1].plot(
                drop_x[highlight_index], 
                abs_drops[highlight_index], 
                'ro', 
                markersize=10, 
                label=f"Chosen k = {optimal_k}"
            )
    axes[1].legend()

    # Plot absolute drop for each k
    for k, abs_drop in zip(k_values[1:], abs_drops):
        axes[1].text(k, abs_drop, f"{abs_drop:.2f}", fontsize=9, ha='center')

    plt.tight_layout()
    plt.show()

    # Print debug info
    print(f"Chosen elbow k: {optimal_k}")

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

""""
different test set
high_risk_events = np.array([
    # Category 0: constant near -30
    [-30, -30, -30, -30, -30, -30],
    [-29, -30, -29, -30, -29, -30],
    [-30, -30, -29, -30, -30, -29],

    # Category 1: ramp from -30 to -10
    [-30, -26, -22, -18, -14, -10],
    [-29, -25, -21, -17, -13, -10],
    [-30, -27, -24, -18, -15, -10],

    # Category 2: ramp from -25 to -2
    [-25, -21, -16, -11, -7,  -2],
    [-26, -20, -15, -10, -6,  -2],
    [-25, -22, -17, -12, -8,  -2],

    # Category 3: oscillation between -30 and -10
    [-30, -10, -30, -10, -30, -10],
    [-29, -10, -29, -10, -29, -10],
    [-30, -11, -30, -11, -30, -11],
])
"""

print("New dataset shape:", high_risk_events.shape)

# Step 1: Run elbow method to get best k
optimal_k = find_optimal_k(high_risk_events)

# Step 2: Cluster events using the optimal k
clustered_events = cluster_high_risk_events(high_risk_events, optimal_k)
print(clustered_events)

