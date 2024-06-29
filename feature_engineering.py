import pandas as pd

def engineer_features(df):
    # Generate RUL
    df['RUL'] = df.groupby('id')['cycle'].transform(max) - df['cycle']
    
    # Generate rolling mean features
    for sensor in [f'sensor{i}' for i in range(1, 22)]:
        df[f'{sensor}_rolling_mean'] = df.groupby('id')[sensor].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    
    return df

if __name__ == '__main__':
    train_df = pd.read_csv('processed_train.csv')
    test_df = pd.read_csv('processed_test.csv')

    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    train_df.to_csv('features_train.csv', index=False)
    test_df.to_csv('features_test.csv', index=False)
