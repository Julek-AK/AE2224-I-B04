from Data_Manager import Data_Manager
from data_viz import plot_cdm_counts

def main():
    dm = Data_Manager('DataSets/train_data.csv' , 'DataSets/test_data.csv')
    train_array, test_array, event_dict = dm.run_pre_processing(risk_threshold=-6.0)
    
    print("Preprocessing complete.")
    print("Train Array shape:", train_array.shape)
    print("Test Array shape:", test_array.shape)
    plot_cdm_counts(event_dict, plot_type="bar")
    num_events = len(event_dict)
    print(f"Number of events: {num_events}")

if __name__ == '__main__':
    main()
