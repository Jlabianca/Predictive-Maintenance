import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data():
    # Define column names
    column_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)] + ['NA_1', 'NA_2']

    # Load train data
    train_df = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=column_names)
    
    # Load test data
    test_df = pd.read_csv('test_FD001.txt', sep='\s+', header=None, names=column_names)
    
    # Load RUL data
    rul_df = pd.read_csv('RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])
    
    # Drop unnecessary columns
    train_df.drop(columns=['NA_1', 'NA_2'], inplace=True)
    test_df.drop(columns=['NA_1', 'NA_2'], inplace=True)
    
    # Normalize the data
    scaler = MinMaxScaler()
    train_df.iloc[:, 2:] = scaler.fit_transform(train_df.iloc[:, 2:])
    test_df.iloc[:, 2:] = scaler.transform(test_df.iloc[:, 2:])
    
    return train_df, test_df, rul_df

if __name__ == '__main__':
    train_df, test_df, rul_df = preprocess_data()
    train_df.to_csv('processed_train.csv', index=False)
    test_df.to_csv('processed_test.csv', index=False)
    rul_df.to_csv('processed_rul.csv', index=False)
