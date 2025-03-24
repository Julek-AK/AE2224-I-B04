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
