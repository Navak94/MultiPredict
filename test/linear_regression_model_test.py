import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json


def load_companies_from_json():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and look for companies.json
    json_path = os.path.join(script_dir, '..', 'companies.json')
    print("Looking for JSON at:", json_path)  # Debugging line
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data['companies']

# --- User Settings ---
stocks = load_companies_from_json()  # Load from JSON instead of hardcoding
forecast_month = '2025-01'  # Month to predict
training_windows = [90, 120, 150, 200]  # Days to train before the month
poly_degrees = [2, 3]  # Polynomial degrees to test

# --- Helper Functions ---
def download_stock_data(stock, start_date, end_date):
    print(f"Downloading {stock} data from {start_date} to {end_date}")
    data = yf.download(stock, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])  # Explicitly convert to datetime
    data.set_index('Date', inplace=True)         # Set Date as index
    return data[['Open']]



def prepare_data(df):
    # Make sure index is datetime
    df.index = pd.to_datetime(df.index)
    X = df.index.map(lambda x: x.toordinal()).values.reshape(-1, 1)
    y = df['Open'].values
    return X, y


def plot_results(dates, actual, predictions, title):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual, label='Actual', color='blue')
    plt.plot(dates, predictions, label='Predicted', color='green')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    #plt.show()

# --- Main Loop ---
for stock in stocks:
    print(f"\n--- Processing {stock} ---")
    
    # Define date ranges
    forecast_start = pd.to_datetime(forecast_month)
    forecast_end = forecast_start + pd.DateOffset(days=31)  # Predict full month
    
    # Download stock data (120 extra days for safety)
    data_start = forecast_start - pd.DateOffset(days=max(training_windows) + 31)
    df = download_stock_data(stock, data_start.strftime('%Y-%m-%d'), forecast_end.strftime('%Y-%m-%d'))
    
    # Store results for comparison
    results = {}
    
    # Loop through configurations
    for window in training_windows:
        for degree in poly_degrees:
            config_name = f"{window}_days_degree_{degree}"
            print(f"\nTesting configuration: {config_name}")
            
            # Prepare Training Data
            train_start = forecast_start - pd.DateOffset(days=window)
            train_end = forecast_start - pd.DateOffset(days=1)  # Train up to the day before the month starts
            train_data = df[(df.index >= train_start) & (df.index <= train_end)]
            test_data = df[(df.index >= forecast_start) & (df.index < forecast_end)]
            
            X_train, y_train = prepare_data(train_data)
            X_test, y_test = prepare_data(test_data)
            
            # Polynomial Features
            poly_features = PolynomialFeatures(degree=degree)
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.transform(X_test)

            # Linear Regression Model
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            
            # Make predictions
            predictions = model.predict(X_test_poly)
            
            # Store predictions for this configuration
            results[config_name] = {
                'dates': test_data.index,
                'actual': y_test,
                'predicted': predictions,
                'MAE': mean_absolute_error(y_test, predictions),
                'RMSE': np.sqrt(mean_squared_error(y_test, predictions))
            }
            
            # Plot predictions vs actual for this configuration
            plot_results(test_data.index, y_test, predictions, f'{stock} - {config_name}')
    
    # --- Compare Configurations ---
    print(f"\n--- Performance Comparison for {stock} ---")
    for config, metrics in results.items():
        print(f"{config}: MAE = {metrics['MAE']:.2f}, RMSE = {metrics['RMSE']:.2f}")
    

    # --- Plot and Save All Configurations Together ---
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df['Open'], label='Actual', color='black')
    for config, metrics in results.items():
        plt.plot(metrics['dates'], metrics['predicted'], label=f'{config}')
    plt.title(f'{stock} - Actual vs Predictions for {forecast_month}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Save the final comparison plot
    output_dir = 'comparison_plots_Linear_Regression'
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    output_file = f'{output_dir}/{stock}_{forecast_month}_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved as {output_file}")

