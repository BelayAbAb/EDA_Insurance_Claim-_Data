import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define path to the processed data
processed_data_path = r'C:\Users\User\Desktop\10Acadamy\Week-3\Data\MachineLearningRating_v3\processed_data'

def load_data(file_path):
    """Load data from a .npy file and check its validity."""
    try:
        data = np.load(file_path, allow_pickle=True)
        # Check if data is empty
        if data.size == 0:
            raise ValueError("Loaded array is empty.")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_feature_importance(importances, feature_names, output_path):
    """Plot and save feature importance."""
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format='jpg')
    plt.close()

def plot_model_performance(y_test, y_pred, output_path):
    """Plot and save model performance: Actual vs Predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('XGBoost: Actual vs. Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, format='jpg')
    plt.close()

def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test):
    """Train and evaluate the XGBoost model."""
    try:
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"XGBoost Mean Squared Error: {mse}")
        print(f"XGBoost R^2 Score: {r2}")

        # Plot feature importance
        if X_train.shape[1] > 0:
            feature_importances = model.feature_importances_
            feature_names = [f'Feature {i}' for i in range(X_train.shape[1])]
            plot_feature_importance(feature_importances, feature_names,
                                    r'C:\Users\User\Desktop\10Acadamy\Week-3\Data\MachineLearningRating_v3\xgboost_feature_importance.jpg')

        # Plot model performance
        plot_model_performance(y_test, y_pred,
                               r'C:\Users\User\Desktop\10Acadamy\Week-3\Data\MachineLearningRating_v3\xgboost_performance.jpg')

    except Exception as e:
        print(f"Error during XGBoost training or evaluation: {e}")

def main():
    # Load datasets
    X_train = load_data(os.path.join(processed_data_path, 'X_train.npy'))
    X_test = load_data(os.path.join(processed_data_path, 'X_test.npy'))
    y_train = load_data(os.path.join(processed_data_path, 'y_train.npy'))
    y_test = load_data(os.path.join(processed_data_path, 'y_test.npy'))

    # Check if data is loaded correctly
    print(f"X_train type: {type(X_train)}, shape: {X_train.shape if X_train is not None else 'None'}")
    print(f"X_test type: {type(X_test)}, shape: {X_test.shape if X_test is not None else 'None'}")
    print(f"y_train type: {type(y_train)}, shape: {y_train.shape if y_train is not None else 'None'}")
    print(f"y_test type: {type(y_test)}, shape: {y_test.shape if y_test is not None else 'None'}")

    # Validate shapes and print samples
    if X_train is not None and X_train.ndim > 0:
        print(f"Sample of X_train: {X_train[:5]}")
    else:
        print("X_train is empty or not correctly loaded.")
    if X_test is not None and X_test.ndim > 0:
        print(f"Sample of X_test: {X_test[:5]}")
    else:
        print("X_test is empty or not correctly loaded.")
    if y_train is not None and y_train.size > 0:
        print(f"Sample of y_train: {y_train[:5]}")
    else:
        print("y_train is empty or not correctly loaded.")
    if y_test is not None and y_test.size > 0:
        print(f"Sample of y_test: {y_test[:5]}")
    else:
        print("y_test is empty or not correctly loaded.")

    # Convert arrays to float type and check content
    try:
        if X_train is not None:
            X_train = np.array(X_train, dtype=float)
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)

        if X_test is not None:
            X_test = np.array(X_test, dtype=float)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)

        if y_train is not None:
            y_train = np.array(y_train, dtype=float)

        if y_test is not None:
            y_test = np.array(y_test, dtype=float)

        # Train and evaluate XGBoost model
        if X_train is not None and y_train is not None:
            train_and_evaluate_xgboost(X_train, X_test, y_train, y_test)

    except Exception as e:
        print(f"Error during data conversion or model evaluation: {e}")

if __name__ == "__main__":
    main()

