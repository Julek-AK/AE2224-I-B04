# visualize_events.py
import matplotlib.pyplot as plt

def plot_cdm_counts(event_dict, plot_type="histogram"):
    """
    Plots the number of CDMs per event from an event dictionary.
    
    Parameters:
        event_dict (dict): Dictionary where keys are event IDs and values are 2D numpy arrays.
        plot_type (str): 'histogram' for a histogram or 'bar' for a bar chart.
    """
    # Extract the number of CDMs for each event.
    cdm_counts = [cdm_array.shape[0] for cdm_array in event_dict.values()]
    
    if plot_type == "histogram":
        plt.figure(figsize=(10, 6))
        plt.hist(cdm_counts, bins=20, edgecolor='black')
        plt.xlabel("Number of CDMs per Event")
        plt.ylabel("Frequency")
        plt.title("Distribution of CDM Counts per Event")
        plt.show()
    elif plot_type == "bar":
        event_ids = list(event_dict.keys())
        plt.figure(figsize=(12, 6))
        plt.bar(event_ids, cdm_counts)
        plt.xlabel("Event ID")
        plt.ylabel("Number of CDMs")
        plt.title("CDM Count per Event")
        plt.show()
    else:
        print("Unknown plot type. Use 'histogram' or 'bar'.")

import random
import matplotlib.pyplot as plt

def plot_scatter_cdm(event_dict, plot_type='scatter', num_events=5):
    """
    Plots a scatter plot of CDM positions for a randomly selected subset of events from an event dictionary.
    
    Parameters:
        event_dict (dict): Dictionary where keys are event IDs and values are 2D numpy arrays.
                           Each array should have at least two columns corresponding to x and y coordinates.
        plot_type (str): Currently only supports 'scatter'.
        num_events (int): The number of events to randomly sample and plot. Defaults to 10.
    """
    if plot_type != 'scatter':
        print("Unknown plot type. Use 'scatter'.")
        return
    
    plt.figure(figsize=(10, 7))
    
    # Randomly sample event keys
    all_event_ids = list(event_dict.keys())
    if len(all_event_ids) > num_events:
        sampled_event_ids = random.sample(all_event_ids, num_events)
    else:
        sampled_event_ids = all_event_ids

    # Loop through each sampled event and plot its CDM positions
    for event_id in sampled_event_ids:
        cdm_array = event_dict[event_id]
        # Ensure there are at least two dimensions for a scatter plot
        if cdm_array.shape[1] < 2:
            print(f"Event {event_id} does not have enough dimensions for a scatter plot (requires at least 2).")
            continue
        
        x = cdm_array[:, 0]
        y = cdm_array[:, 1]
        plt.scatter(y, x, label=f"Event {event_id}", alpha=0.7)
    
    plt.xlabel("TIME TO TCA")
    plt.ylabel("RISK")
    plt.title("Scatter Plot of CDM Positions for Randomly Selected Events")
    plt.legend()
    plt.show()
