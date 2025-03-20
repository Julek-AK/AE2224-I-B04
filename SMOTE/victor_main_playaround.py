from Data_Manager import Data_Manager

df = Data_Manager(train_file = "DataSets/train_data.csv", test_file="DataSets/test_data.csv")
caca = df.load_data()
print(caca.head())