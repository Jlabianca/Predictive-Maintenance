# Project Overview
## Objective

This project requires designing an algorithm that predicts whether an engine failure is likely to occur in the near future based on NASA’s Turbofan Engine Degradation Simulation Data Set (CMAPSS). It involves helping mechanics identify when an engine needs to be replaced by predicting machinery breakdowns. Therefore, it can reduce unplanned downtime, maintenance costs, and maximize engine safety and reliability.

## Key Concepts and Workflow

Predictive Maintenance
Predictive maintenance means the prediction of equipment failures, before they happen, and the scheduling of timely maintenance activities in order to avoid these failures. Assuming that we perform maintenance on a machine before it indeed fails, and using sensor data coupled with machine learning algorithms in order to monitor equipment’s condition and prediction of its Remaining Useful Life (RUL).

## Machine Learning Approach
This project trains a machine learning algorithm to predict the remaining useful life of a turbofan engine using a supervised learning approach. The algorithm consists of the following four steps:
1. Data Preprocessing: Load, clean, and normalize the data.
2. Feature Engineering: Create additional features that help improve model performance.
3. Model Building: Train a machine learning model to predict the RUL.
4. Model Evaluation: Assess the performance of the model.
5. Visualization: Visualize the model’s predictions to interpret its accuracy and reliability.


```bash
predictive_maintenance/
│
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   ├── processed_train.csv
│   ├── processed_test.csv
│   ├── processed_rul.csv
│   ├── features_train.csv
│   ├── features_test.csv
│
├── models/
│   └── random_forest_model.pkl
│
├── scripts/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   └── visualization.py
│
└── main.py
```

## Conclusion

Given the current scenario of increasing air traffic and different weather events that either intensify the pressure on the engines or damage the turbofan itself, this project shows how a predictive maintenance model can be built using machine learning, by following a process that requires data preprocessing, feature engineering, model building, evaluation, and visualization.
In this manner, we can anticipate the RUL (Remaining Useful Life) of the engines and plan the maintenance in advance, avoiding unplanned downtime and high costs.

## Key Points:

1. Data Preprocessing: Cleaning and normalizing the data for consistent model input.
2. Feature Engineering: Creating new features to enhance model accuracy.
3. Model Building: Training a Random Forest Regressor to predict RUL.
4. Model Evaluation: Using MSE to assess model performance.
5. Visualization: Plotting true vs. predicted RUL for interpretation.

We use Git LFS to manage the large model file, and it works well enough that we can work in a team and keep individual versions without having to spend excessive time downloading and uploading all the data. This workflow can be generally applied in predictive maintenance applications in other industries.
