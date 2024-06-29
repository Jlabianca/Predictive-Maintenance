import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import gzip
import shutil

def compress_file(file_path):
    with open(file_path, 'rb') as f_in:
        with gzip.open(f"{file_path}.gz", 'wb', compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(file_path)  # Remove the uncompressed file
    os.rename(f"{file_path}.gz", file_path)  # Rename the compressed file to the original name

def build_and_train_model(train_df, test_df):
    X = train_df.drop(columns=['id', 'cycle', 'RUL'])
    y = train_df['RUL']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reduce model complexity further
    model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    # Ensure the models directory exists
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")
    
    model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Compress the model file
    compress_file(model_path)
    print(f"Model compressed to {model_path}")

if __name__ == '__main__':
    train_df = pd.read_csv('features_train.csv')
    test_df = pd.read_csv('features_test.csv')
    build_and_train_model(train_df, test_df)
