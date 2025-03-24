from Data_Manager import Data_Manager

def main():
    dm = Data_Manager('train_data.csv' , 'test_data.csv')
    train_array, test_array = dm.run_pre_processing(risk_threshold=-4.0)
    print("Preprocessing complete.")
    print("Train Array shape:", train_array.shape)
    print("Test Array shape:", test_array.shape)
    print("Train Array shape:", train_array.shape)
    print("Test Array shape:", test_array.shape)
if __name__ == '__main__':
    main()
