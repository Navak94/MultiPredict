import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import yfinance as yf
import datetime
import os
import csv
from datetime import datetime, timedelta
import json
import time

print("yfinance version:", yf.__version__)

# Create output directory
os.makedirs('stock_predictions', exist_ok=True)

# Load configuration from JSON file
def load_config_json(path="companies.json"):
    with open(path, 'r') as file:
        config = json.load(file)
    return config['companies']

# Function to append data to CSV
def append_to_csv(Company, today_val, predicted_val, trend_val, filename="stock_analysis_linear_regression.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Company", "Today Value", "Predicted Value", "Trend Value"])  # Headers
        writer.writerow([Company, today_val, predicted_val, trend_val])

# Retry logic for downloading data
def download_data(Company, start_time_past, next_year, retries=3):
    for attempt in range(retries):
        try:
            data = yf.download(Company, 
                               start=start_time_past.strftime("%Y-%m-%d"), 
                               end=next_year.strftime("%Y-%m-%d"), 
                               progress=False, 
                               threads=False, 
                               auto_adjust=False)
            if data.empty:
                print(f"No data for {Company}. Retrying... ({attempt+1}/{retries})")
                time.sleep(2)
                continue
            return data
        except Exception as e:
            print(f"Error fetching data for {Company}: {e}. Retrying... ({attempt+1}/{retries})")
            time.sleep(2)
    print(f"Failed to download data for {Company} after {retries} attempts.")
    return None

# Main loop
training_data_days_list = [120]
companies = load_config_json()
total_invest = 0
total_return = 0

for training_data_days in training_data_days_list:
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    current_date = datetime.strptime(formatted_date, "%Y-%m-%d")
    next_day_date = current_date + timedelta(days=1)
    past_date_beginning_data = current_date - timedelta(days=training_data_days)
    future_date_beginning_data = current_date + timedelta(days=30)
    next_year = current_date + timedelta(days=365)
    
    start_time_past = past_date_beginning_data
    end_time_past = formatted_date
    start_time_future = next_day_date
    end_time_future = future_date_beginning_data
    
    for Company in companies:
        directory = "stock_predictions/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        csv_file = f"{directory}{Company}_data.csv"
        
        # Download data with retry logic
        data = download_data(Company, start_time_past, next_year)
        if data is None:
            continue
        
        # Save data to CSV
        data.reset_index().to_csv(csv_file, index=False)

        
        # Read the CSV file
        df = pd.read_csv(csv_file, usecols=['Date', 'Open'])
        
        # Ensure correct data types
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
        
        # Check if there's any valid data
        if df['Open'].dropna().empty:
            print(f"No valid 'Open' prices for {Company}. Skipping...")
            continue
        
        # Extract the last valid value
        today_val = float(df['Open'].dropna().values[-1])
        
        # Prepare Data
        # Drop rows where Date is NaT
        df = df.dropna(subset=['Open'])

        # Drop rows where index (Date) is NaT
        df = df[~df.index.isna()]

        # Now safely convert dates to ordinal
        X = df.index.map(lambda x: x.toordinal()).values.reshape(-1, 1)

        y = df['Open'].dropna().values
        
        if len(y) == 0:
            print(f"No data for {Company}. Skipping...")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training
        poly_features = PolynomialFeatures(degree=3)
        X_train_poly = poly_features.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Predict future dates
        future_dates = pd.date_range(start=start_time_future, end=end_time_future, freq='D')
        future_dates_numeric = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
        future_dates_poly = poly_features.transform(future_dates_numeric)
        future_predictions = model.predict(future_dates_poly)
        
        # Calculate trend
        predicted_val = float(future_predictions[-1])
        predicted_val_first = float(future_predictions[0])
        trend_val = predicted_val - predicted_val_first
        
        # Determine trend term
        term = "long term" if training_data_days == 120 else "short term"
        
        if trend_val >= 0:
            append_to_csv(Company, today_val, predicted_val, "Upward Trend")
            if trend_val > 1.5:
                #print(f'{Company} {term} HIGH upward Trend!')
                total_invest += today_val
                total_return += predicted_val
        else:
            append_to_csv(Company, today_val, predicted_val, "Downward Trend")
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Open'], color='blue', label='Original Data')
        plt.plot(future_dates, future_predictions, color='green', label='Future Predictions')
        plt.title(Company)
        plt.xlabel('Date')
        plt.ylabel('Open')
        plt.legend()
        plt.grid(True)
        
        plot_folder_save = f'{directory}/{Company}/'
        if not os.path.exists(plot_folder_save):
            os.makedirs(plot_folder_save)
            
        plot_filename = f'{plot_folder_save}{Company}_{training_data_days}_training_days_data_Linear_Regression.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

# Final Output
percent_return = round(100 * (total_return / total_invest) - 100, 2) if total_invest > 0 else 0
total_invest = round(total_invest, 2)
total_return = round(total_return, 2)
print(f"Total Investment: ${total_invest}")
print(f"Total Return: ${total_return}")
print(f"Percentage Return: {percent_return}%")
