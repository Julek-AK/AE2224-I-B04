import pandas as pd
import scipy
import numpy as np
from formatting_and_interpolation import Interpolate_
from DTW_class_merged import TimeSeriesClustering
import matplotlib.pyplot as plt

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


if __name__ == "__main__":

    train_df, test_df = pandas_data_frame_creation()
    filtered_train_df = filter_by_risk(train_df, -5.0)
    sorted_train_df = sort_by_mission_id(filtered_train_df)
    cleaned_data = clean_data(sorted_train_df)

    print(cleaned_data.shape)

    event_dict = create_event_dict(cleaned_data)

    def event_with_extreme_cdms(event_dict):
        max_event = None
        max_cdms = -1  # start with a very low number
        min_event = None
        min_cdms = float('inf')  # start with a very high number
        
        for event_id, cdm_array in event_dict.items():
            num_cdms = cdm_array.shape[0]
            if num_cdms > max_cdms:
                max_cdms = num_cdms
                max_event = event_id
            if num_cdms < min_cdms:
                min_cdms = num_cdms
                min_event = event_id
        return max_event, max_cdms, min_event, min_cdms

    '''
    max_event, max_cdms, min_event, min_cdms = event_with_extreme_cdms(event_dict)
    print(f"Event {max_event} has the maximum number of CDMs: {max_cdms}")
    print(f"Event {min_event} has the minimum number of CDMs: {min_cdms}")
    '''
    interpolated_event_dict = Interpolate_(event_dict,3)

    dtw = TimeSeriesClustering(interpolated_event_dict)
    optimal_k, kmeans_model, clustered_events, cluster_labels = dtw.run_pipeline()


    #print(interpolated_event_dict)
    print(type(clustered_events))
    print(clustered_events)

    print("Cluster Labels:")
    print(type(cluster_labels))
    print(cluster_labels)


    labels = cluster_labels

    # Create a figure for subplots
    num_clusters = len(np.unique(labels))  # Determine number of unique clusters
    cols = 3  # Number of columns in the subplot grid
    rows = (num_clusters // cols) + (num_clusters % cols > 0)  # Calculate number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Adjust the figsize as necessary

    # Flatten axs for easier iteration
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    # Get unique clusters from labels
    unique_labels = np.unique(labels)

    # Create a dictionary to map labels to colors
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # Using a colormap for consistent colors
    label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}

    # Scale the axes equally for all subplots
    tca_min, tca_max = float('inf'), float('-inf')
    pc_min, pc_max = float('inf'), float('-inf')

    # First, determine the axis limits across all events
    for i, (event_id, data_points) in enumerate(event_dict.items()):
        pc_values = [point[0] for point in data_points]
        tca_values = [point[1] for point in data_points]
        tca_min, tca_max = min(tca_min, min(tca_values)), max(tca_max, max(tca_values))
        pc_min, pc_max = min(pc_min, min(pc_values)), max(pc_max, max(pc_values))

    # Plot each event in the corresponding cluster's subplot
    for i, (event_id, data_points) in enumerate(event_dict.items()):
        label = labels[i]  # Get the label for the event
        cluster_index = unique_labels.tolist().index(label)  # Get the index for the cluster
        
        # Extract PC and TCA values
        pc_values = [point[0] for point in data_points]
        tca_values = [point[1] for point in data_points]
        
        # Plot the event points in the appropriate subplot
        axs[cluster_index].scatter(tca_values, pc_values, color=label_to_color[label], label=f"Event {event_id}")
        axs[cluster_index].plot(tca_values, pc_values, linestyle='-', alpha=0.5, color=label_to_color[label])
        
        # Set labels and title for each subplot
        axs[cluster_index].set_xlabel('Time to Closest Approach (TCA)')
        axs[cluster_index].set_ylabel('Probability of Collision (PC)')
        axs[cluster_index].set_title(f'Cluster {label}')

    # Adjust the overall plot for the combined plot with colors for clusters
    fig_combined, ax_combined = plt.subplots(figsize=(8, 6))

    # Plot all events on the combined plot, colored by cluster label
    for i, (event_id, data_points) in enumerate(event_dict.items()):
        label = labels[i]  # Get the label for the event
        color = label_to_color[label]  # Get the color for the label
        
        # Extract PC and TCA values
        pc_values = [point[0] for point in data_points]
        tca_values = [point[1] for point in data_points]
        
        # Plot the event points on the combined plot
        ax_combined.scatter(tca_values, pc_values, color=color, label=f"Event {event_id}")
        ax_combined.plot(tca_values, pc_values, linestyle='-', alpha=0.5, color=color)

    # Set labels and title for the combined plot
    ax_combined.set_xlabel('Time to Closest Approach (TCA)')
    ax_combined.set_ylabel('Probability of Collision (PC)')
    ax_combined.set_title('All Events Colored by Cluster')

    # Set the same limits for all subplots to keep them scaled equally
    for ax in axs:
        ax.set_xlim(tca_min, tca_max)
        ax.set_ylim(pc_min, pc_max)

    # Show all plots
    plt.tight_layout()
    plt.show()