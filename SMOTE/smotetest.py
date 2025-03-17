import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging
import matplotlib.pyplot as plt

# Define clusters
cluster_0 = np.array([
    [0.01, 0.02, 0.04, 0.08, 0.12],  # Event 1 (Low Risk)
    [0.02, 0.03, 0.05, 0.10, 0.15]   # Event 2 (Low Risk)
])

cluster_1 = np.array([
    [0.50, 0.48, 0.45, 0.40, 0.35],  # Event 3 (High Risk)
    [0.55, 0.53, 0.50, 0.45, 0.40]   # Event 4 (High Risk)
])

# Compute DBA for each cluster
barycenter_0 = dtw_barycenter_averaging(cluster_0)
barycenter_1 = dtw_barycenter_averaging(cluster_1)

# Plot results
plt.figure(figsize=(8, 5))

# Plot cluster members
for series in cluster_0:
    plt.plot(series, linestyle='dashed', color='blue', alpha=0.5)
plt.plot(barycenter_0, linestyle='solid', color='blue', label="DBA Cluster 0")

for series in cluster_1:
    plt.plot(series, linestyle='dashed', color='red', alpha=0.5)
plt.plot(barycenter_1, linestyle='solid', color='red', label="DBA Cluster 1")

plt.title("DTW Barycenter Averaging (DBA)")
plt.xlabel("Time Steps")
plt.ylabel("Risk Probability")
plt.legend()
plt.grid(True)
plt.show()
