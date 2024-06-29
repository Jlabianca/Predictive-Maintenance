import pandas as pd
import matplotlib.pyplot as plt

def plot_results(test_df, y_pred):
    # Ensure that the test DataFrame and predictions are not empty
    if test_df.empty or len(y_pred) == 0:
        print("No data available to plot.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(test_df['cycle'], test_df['RUL'], label='True RUL', alpha=0.6)
    plt.scatter(test_df['cycle'], y_pred, label='Predicted RUL', alpha=0.6)
    
    plt.xlabel('Cycle')
    plt.ylabel('RUL')
    plt.title('True vs Predicted RUL over Cycles')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # This part is optional and for standalone testing of the plotting function
    # Load test data
    test_df = pd.read_csv('features_test.csv')
    
    # Simulate predictions for illustration purposes
    # Replace with actual predictions from your model
    y_pred = [200] * len(test_df)  # Dummy data
    
    # Plot results
    plot_results(test_df, y_pred)
