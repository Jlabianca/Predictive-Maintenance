import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def evaluate_model(test_df, rul_df):
    X_test = test_df.drop(columns=['id', 'cycle', 'RUL'])
    y_test = test_df['RUL']
    
    model_path = 'models/random_forest_model.pkl'
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    return y_pred

if __name__ == '__main__':
    test_df = pd.read_csv('features_test.csv')
    rul_df = pd.read_csv('processed_rul.csv')
    evaluate_model(test_df, rul_df)
