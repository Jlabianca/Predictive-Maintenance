from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from model_building import build_and_train_model
from model_evaluation import evaluate_model
from visualization import plot_results
import pandas as pd
import joblib

def main():
    # Step 1: Data Preprocessing
    train_df, test_df, rul_df = preprocess_data()
    
    # Step 2: Feature Engineering
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # Save the engineered features for consistency
    train_df.to_csv('features_train.csv', index=False)
    test_df.to_csv('features_test.csv', index=False)
    
    # Step 3: Model Building
    build_and_train_model(train_df, test_df)
    
    # Step 4: Model Evaluation
    y_pred = evaluate_model(test_df, rul_df)
    
    # Step 5: Visualization
    plot_results(test_df, y_pred)

if __name__ == '__main__':
    main()
