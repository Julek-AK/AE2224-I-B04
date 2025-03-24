from Data_Manager import Data_Manager

def main():
    dm = Data_Manager('DataSets/train_data.csv' , 'DataSets/test_data.csv')
    dm.load_data()  # Now dm.train_df and dm.test_df are set as DataFrames



    print("Train DF shape:", dm.train_df.shape)
    print("Test DF shape:", dm.test_df.shape)
    
    
    
    
    train_array, test_array = dm.run_pre_processing(risk_threshold=-4.0)
    
    
    
    print("Preprocessing complete.")
    print("Train Array shape:", train_array.shape)
    print("Test Array shape:", test_array.shape)

if __name__ == '__main__':
    main()
