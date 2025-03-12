import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
import os
import json
import time
import random
from datetime import datetime

def load_companies_from_json():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, '..', 'companies.json')
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data['companies']

# --- User Settings ---
stocks = load_companies_from_json()
start_year = 2015
end_year = datetime.now().year

monthly_trends = []
failed_downloads = []
skipped_cases = []

# --- Helper Functions ---
def download_stock_data(stock, start_date, end_date):
    """Downloads stock data with retries and rate-limit handling."""
    print(f"Downloading {stock} data from {start_date} to {end_date}")

    attempts = 3
    for attempt in range(attempts):
        try:
            data = yf.download(stock, start=start_date, end=end_date)
            if not data.empty:
                time.sleep(random.uniform(0.1, 0.5))
                return add_technical_indicators(data)
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {stock}: {e}")
            time.sleep(30)

    print(f"Failed to download {stock} after {attempts} attempts.")
    failed_downloads.append({'Stock': stock, 'Start Date': start_date, 'End Date': end_date})
    return None  

def add_technical_indicators(df):
    """Adds Moving Averages, RSI, Bollinger Bands, and Volume Change."""
    df['10_MA'] = df['Open'].rolling(window=10).mean()
    df['50_MA'] = df['Open'].rolling(window=50).mean()
    df['200_MA'] = df['Open'].rolling(window=200).mean()
    df['RSI'] = 100 - (100 / (1 + df['Open'].diff().clip(lower=0).rolling(14).mean() /
                                 df['Open'].diff().clip(upper=0).abs().rolling(14).mean()))
    df['Bollinger_Upper'] = df['Open'].rolling(20).mean() + 2 * df['Open'].rolling(20).std()
    df['Bollinger_Lower'] = df['Open'].rolling(20).mean() - 2 * df['Open'].rolling(20).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    df.dropna(inplace=True)  # Remove NaNs from rolling calculations
    return df

def prepare_data(df):
    df.index = pd.to_datetime(df.index)
    X = df.index.map(lambda x: x.toordinal()).values.reshape(-1, 1)
    y = df['Open'].values
    return X, y

def calculate_monthly_trend(prices):
    """Returns 1 if the month's trend is UP, else 0."""
    return 1 if prices[-1] > prices[0] else 0

def get_dynamic_training_windows(volatility):
    """Adjusts training window size based on market volatility."""
    if volatility > 0.05:  # High volatility
        return [60, 120]  # Use longer training periods
    else:  # Low volatility
        return [15, 30]  # Focus on recent trends

# --- Loop Through Years and Months ---
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        if year == datetime.now().year and month > datetime.now().month:
            continue  

        forecast_month = f'{year}-{month:02d}'

        for stock in stocks:
            print(f"\n--- Processing {stock} for {forecast_month} ---")

            forecast_start = pd.to_datetime(forecast_month)
            forecast_end = forecast_start + pd.DateOffset(days=31)

            df = download_stock_data(stock, (forecast_start - pd.DateOffset(days=400)).strftime('%Y-%m-%d'),
                                     forecast_end.strftime('%Y-%m-%d'))
            if df is None:
                print(f"Skipping {stock} due to download failure.")
                continue

            volatility = df['Open'].pct_change().std().item()  # ✅ Convert Series to scalar
            training_windows = get_dynamic_training_windows(volatility)

            actual_prices = df[(df.index >= forecast_start) & (df.index < forecast_end)]['Open'].values
            if len(actual_prices) < 2:
                print(f"Not enough data for {stock} in {forecast_month}. Skipping...")
                continue

            actual_trend = calculate_monthly_trend(actual_prices)

            monthly_trend_row = {'Stock': stock, 'Year': year, 'Month': month, 'Actual Trend': actual_trend}

            for window in training_windows:
                train_start = forecast_start - pd.DateOffset(days=window)
                train_end = forecast_start - pd.DateOffset(days=1)
                train_data = df[(df.index >= train_start) & (df.index <= train_end)]
                test_data = df[(df.index >= forecast_start) & (df.index < forecast_end)]

                X_train, y_train = prepare_data(train_data)
                X_test, _ = prepare_data(test_data)  # No need for y_test in logistic regression

                if len(y_train) < 2:
                    print(f"Skipping {stock} - Not enough training data.")
                    continue

                y_train_trend = np.array([1 if y_train[i] > y_train[i-1] else 0 for i in range(1, len(y_train))])
                X_train = X_train[1:]  # Align X_train with y_train_trend

                if len(np.unique(y_train_trend)) < 2:
                    print(f"Skipping {stock} for {forecast_month} - Only one class in training data.")
                    continue

                # Polynomial Features
                poly_features = PolynomialFeatures(degree=2, include_bias=False)
                X_train_poly = poly_features.fit_transform(X_train)
                X_test_poly = poly_features.transform(X_test)

                # Feature Selection
                selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'), threshold="mean")
                selector.fit(X_train_poly, y_train_trend)

                if selector.transform(X_train_poly).shape[1] == 0:
                    print(f"⚠️ No features selected for {stock} in {forecast_month}, using all features.")
                else:
                    X_train_poly = selector.transform(X_train_poly)
                    X_test_poly = selector.transform(X_test_poly)

                # Train Logistic Regression
                model = LogisticRegression(max_iter=1000, solver='saga', penalty='l1', class_weight='balanced')
                model.fit(X_train_poly, y_train_trend)

                predicted_trend = model.predict(X_test_poly)[-1]  # ✅ Fixed to use `X_test_poly`
                confidence = model.predict_proba(X_test_poly)[-1][predicted_trend]

                monthly_trend_row[f"{window}_days"] = "Up" if predicted_trend == 1 else "Down"
                monthly_trend_row[f"{window}_days_Confidence"] = confidence
                monthly_trend_row[f"{window}_days_Correct"] = 1 if predicted_trend == actual_trend else 0  # ✅ New column

            monthly_trends.append(monthly_trend_row)

# --- Save Monthly Trends to CSV ---
output_dir = 'logistic_regression_trend_analysis'
os.makedirs(output_dir, exist_ok=True)
monthly_trends_df = pd.DataFrame(monthly_trends)
output_file = f'{output_dir}/monthly_trends_logistic_regression.csv'
monthly_trends_df.to_csv(output_file, index=False)
print(f"\n✅ Monthly trends saved to: {output_file}")
