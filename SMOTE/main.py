from Data_Manager import Data_Manager

def main():
    df = Data_Manager("DataSets/train_data.csv", "DataSets/test_data.csv")
    df.load_data()
    df.clean_data()
    df.filter_by_risk(-4.0)
    df.sort_by_event_id_time_to_tca()
    df.create_event_dict()
    df.df_to_numpy()
    print(df)

if __name__ == '__main__':
    main()
