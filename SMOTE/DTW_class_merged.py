import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from kneed import KneeLocator

class TimeSeriesClustering:
    """
    A class to perform DTW-based K-Means clustering on time series da
    
    
    ta,
    find the optimal number of clusters via the elbow method, and improve
    cluster centroids using DTW Barycenter Averaging (DBA).
    """
    def __init__(self, data):
        """
        Initializes the clustering instance.
        
        Parameters:
            data (np.ndarray): A NumPy array of time series data. Each row 
                               should represent a time series.
        """
        self.data = data

    def find_optimal_k(self, k_min=1, k_max=10, drop_threshold=0.05, log_scale=False):
        """
        Computes DTW-based distortions for a range of k-values, plots the
        distortion and its drop across k, and returns the chosen optimal k.
        
        Parameters:
            k_min (int): Minimum number of clusters to try.
            k_max (int): Maximum number of clusters to try.
            drop_threshold (float): Absolute drop threshold to determine the elbow.
            log_scale (bool): Whether to display the distortion plot in log scale.
            
        Returns:
            optimal_k (int): The chosen number of clusters based on the elbow method.
        """
        k_values = list(range(k_min, k_max + 1))
        distortions = []

        # Compute distortions for each k using DTW-based K-Means.
        for k in k_values:
            model = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=42, n_init=10)
            model.fit(self.data)
            distortions.append(model.inertia_)
        distortions = np.array(distortions)
        
        # Calculate absolute drop in distortion from one k to the next.
        abs_drops = np.array([distortions[i] - distortions[i+1] for i in range(len(distortions) - 1)])
        
        # Determine the optimal k based on the first drop below the threshold.
        below_thresh_indices = np.where(abs_drops < drop_threshold)[0]
        if len(below_thresh_indices) > 0:
            elbow_index = below_thresh_indices[0] + 1
            optimal_k = k_values[elbow_index]
        else:
            elbow_index = len(k_values) - 1
            optimal_k = k_values[elbow_index]
        
        # Plotting the elbow method results.
        fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
        
        # Top plot: Distortion vs. k.
        axes[0].plot(k_values, distortions, 'o-', label="Distortion")
        if log_scale:
            axes[0].set_yscale("log")
        axes[0].set_ylabel("Distortion (DTW SSE)")
        axes[0].set_title(f"Elbow Method (Threshold={drop_threshold}): Distortion vs. k")
        axes[0].grid(True)
        axes[0].set_ylim(0, 2)
        chosen_k_index = k_values.index(optimal_k)
        axes[0].plot(optimal_k, distortions[chosen_k_index], 'ro', markersize=10, label=f"Chosen k = {optimal_k}")
        axes[0].legend()
        
        # Bottom plot: Absolute drop vs. k.
        drop_x = k_values[1:]
        axes[1].plot(drop_x, abs_drops, 's-', label="Absolute drop")
        axes[1].axhline(drop_threshold, color='r', linestyle='--', label="Threshold")
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Absolute Drop")
        axes[1].grid(True)
        axes[1].set_title("Absolute Drop from k to k+1")
        if optimal_k > k_values[0]:
            highlight_index = optimal_k - k_values[0] - 1  # offset in abs_drops
            if 0 <= highlight_index < len(abs_drops):
                axes[1].plot(drop_x[highlight_index], abs_drops[highlight_index], 'ro', markersize=10, label=f"Chosen k = {optimal_k}")
        axes[1].legend()
        for k, abs_drop in zip(k_values[1:], abs_drops):
            axes[1].text(k, abs_drop, f"{abs_drop:.2f}", fontsize=9, ha='center')
        
        plt.tight_layout()
        plt.show()
        print(f"Chosen elbow k according to the algorithm: {optimal_k}")
        
        return optimal_k

    def cluster_high_risk_events(self, num_clusters):
        """
        Clusters the time series data into the specified number of clusters
        using DTW-based K-Means.
        
        Parameters:
            num_clusters (int): The number of clusters to form.
            
        Returns:
            kmeans_model: The fitted K-Means model.
            clustered_events (dict): Dictionary mapping each cluster label to the
                                     corresponding events as a NumPy array.
        """
        kmeans_model = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", random_state=42)
        cluster_labels = kmeans_model.fit_predict(self.data)
        
        # Print cluster assignments.
        for i, label in enumerate(cluster_labels):
            print(f"Event {i+1} assigned to Cluster {label}")
        
        # Organize events by cluster.
        clustered_events = {i: [] for i in range(num_clusters)}
        for i, label in enumerate(cluster_labels):
            clustered_events[label].append(self.data[i])
        clustered_events = {k: np.array(v) for k, v in clustered_events.items()}
        
        return kmeans_model, clustered_events
    """
    def apply_dba_to_centroids(self, kmeans_model):


        Applies DTW Barycenter Averaging (DBA) to refine the centroids
        obtained from the K-Means model.
        
        Parameters:
            kmeans_model: The K-Means model whose centroids are to be improved.
            
        Returns:
            improved_centroids (np.ndarray): The DBA-refined centroids.

        # Use DBA to refine the centroids.
        dba = TimeSeriesDBA(n_iter=10)
        dba.fit(self.data)
        improved_centroids = dba.cluster_centers_
        return improved_centroids
        """
    def run_pipeline(self, k_min=1, k_max=10, drop_threshold=0.05, log_scale=False):
        """
        Runs the complete clustering pipeline:
          1. Determines the optimal number of clusters using the elbow method.
          2. Clusters the data.
          3. Applies DBA to improve the centroids.
          4. Plots the improved centroids.
          
        Returns:
            optimal_k (int): Chosen number of clusters.
            kmeans_model: The fitted K-Means model.
            clustered_events (dict): The events organized by cluster.
            improved_centroids (np.ndarray): The refined centroids.
        """
        optimal_k = self.find_optimal_k(k_min=k_min, k_max=k_max, drop_threshold=drop_threshold, log_scale=log_scale)
        k_value = int(input("Please input the optimal k value: ")) #this is bcs I want the user to select it based on the graph
        kmeans_model, clustered_events = self.cluster_high_risk_events(k_value)
        #improved_centroids = self.apply_dba_to_centroids(kmeans_model)
        """"
        # Plot improved centroids.
        for i, centroid in enumerate(improved_centroids):
            plt.plot(centroid.ravel(), label=f'Improved Centroid {i}')
        plt.legend()
        plt.title("DBA Improved Centroids")
        plt.show()
        """
        return optimal_k, kmeans_model, clustered_events #,improved_centroids

# ------------------ Example usage ------------------

if __name__ == "__main__":
    # Example high-risk events dataset (9 events, 6 time steps each).
    high_risk_events = np.array([
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        [1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
        [0.9, 0.4, 0.9, 0.4, 0.9, 0.4],
        [0.8, 0.3, 0.8, 0.3, 0.8, 0.3],
    ])
    
    # Create an instance of TimeSeriesClustering.
    clustering = TimeSeriesClustering(high_risk_events)
    
    # Run the full pipeline.
    clustering.run_pipeline()
