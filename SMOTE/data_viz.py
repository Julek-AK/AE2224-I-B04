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

def plot_scatter_cdm(event_dict, plot_type='scatter'):
    """
    Plots a scatter plot of CDM positions for each event from an event dictionary.
    
    Parameters:
        event_dict (dict): Dictionary where keys are event IDs and values are 2D numpy arrays.
                           Each array should have at least two columns corresponding to x and y coordinates.
        plot_type (str): Currently only supports 'scatter'.
    """
    if plot_type != 'scatter':
        print("Unknown plot type. Use 'scatter'.")
        return

    plt.figure(figsize=(10, 7))
    
    # Loop through each event and plot its CDM positions
    for event_id, cdm_array in event_dict.items():
        # Ensure there are at least two dimensions for a scatter plot
        if cdm_array.shape[1] < 2:
            print(f"Event {event_id} does not have enough dimensions for a scatter plot (requires at least 2).")
            continue
        
        x = cdm_array[:, 0]
        y = cdm_array[:, 1]
        plt.scatter(x, y, label=f"Event {event_id}", alpha=0.7)
    
    plt.xlabel("Risk")
    plt.ylabel("Time to TCA")
    plt.title("placeholdeeer")
    plt.legend()
    plt.show()
