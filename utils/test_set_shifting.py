"""
Shifts all CDMs in the test set by two days, allowing for actual benchmarking and testing to be performed

This code will generate a new .csv file in DataSets, called test_data_shifted.csv
with the exact same formatting as the original, so that it can be used seamlessly
"""

# External Imports
import pandas as pd



def main():
    test_data = pd.read_csv("DataSets\\test_data.csv")
    test_data['time_to_tca'] -= 2
    
    test_data.to_csv("DataSets\\test_data_shifted.csv", index=False)
    print("Shifted test data generated!")


if __name__ == "__main__":
    main()

