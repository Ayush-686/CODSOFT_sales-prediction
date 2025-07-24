import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

def predict_sales():
    print("=== Starting Sales Prediction ===")

    # Attempt to load dataset
    dataset = None
    filenames = ['advertising.csv', 'Advertising.csv']

    for name in filenames:
        try:
            dataset = pd.read_csv(name)
            print(f"Dataset loaded: '{name}' with shape {dataset.shape}")
            break
        except FileNotFoundError:
            continue
        except Exception as err:
            print(f"Format error with '{name}': {err}")

    if dataset is None:
        print("Searching for advertising dataset in current directory...")
        for fname in os.listdir('.'):
            if "advertising" in fname.lower() and fname.endswith(".csv"):
                try:
                    dataset = pd.read_csv(fname)
                    print(f"Auto-detected and loaded: '{fname}' with shape {dataset.shape}")
                    break
                except Exception as e:
                    print(f"Could not load '{fname}': {e}")

    if dataset is None:
        print("ERROR: Advertising dataset not found. Upload the file and try again.")
        return

    print("\n--- First 5 Rows ---")
    print(dataset.head())

    print("\n--- Checking for Missing Values ---")
    print(dataset.isnull().sum())

    # Standardize column names
    dataset.columns = [col.lower().replace('.', '_').replace(' ', '_') for col in dataset.columns]

    features = ['tv', 'radio', 'newspaper']
    target = 'sales'

    # Check if required columns exist
    if not all(col in dataset.columns for col in features + [target]):
        print("Missing required columns in dataset:")
        print("Available columns:", dataset.columns.tolist())
        return

    # Drop rows with missing values
    before_drop = dataset.shape[0]
    dataset.dropna(subset=features + [target], inplace=True)
    after_drop = dataset.shape[0]
    if before_drop != after_drop:
        print(f"Removed {before_drop - after_drop} rows with missing values.")

    X = dataset[features]
    y = dataset[target]

    print(f"\nFeature Matrix Shape: {X.shape}")
    print("Target Vector Shape:", y.shape)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\nData split into training and testing sets.")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model trained.")

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    print("\n--- Model Evaluation ---")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE:  {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R²:   {r2_score(y_test, y_pred):.2f}")

    # Coefficients
    print("\n--- Model Coefficients ---")
    for feature, coef in zip(features, model.coef_):
        print(f"{feature.capitalize()}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    # Sample predictions
    sample_inputs = pd.DataFrame([
        [200, 30, 15],
        [10, 40, 50],
        [150, 5, 5]
    ], columns=features)

    print("\n--- Sample Predictions ---")
    sample_preds = model.predict(sample_inputs)
    for idx, (inp, pred) in enumerate(zip(sample_inputs.values, sample_preds), start=1):
        print(f"Campaign {idx}: TV={inp[0]}k, Radio={inp[1]}k, Newspaper={inp[2]}k => Predicted Sales: {pred:.2f}k")

    # Detailed predictions
    results = pd.DataFrame({
        'Actual_Sales': y_test.reset_index(drop=True),
        'Predicted_Sales': y_pred,
        'TV_Budget': X_test['tv'].reset_index(drop=True),
        'Radio_Budget': X_test['radio'].reset_index(drop=True),
        'Newspaper_Budget': X_test['newspaper'].reset_index(drop=True)
    })

    print("\n--- First 10 Test Predictions ---")
    print(results.head(10).to_string(index=False))

    print("\n--- Last 10 Test Predictions ---")
    print(results.tail(10).to_string(index=False))

    print("\n=== Sales Prediction Complete ===")

if __name__ == "__main__":
    predict_sales()
