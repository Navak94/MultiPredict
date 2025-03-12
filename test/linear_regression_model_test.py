import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta

def load_companies_from_json():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, '..', 'companies.json')
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data['companies']

# --- User Settings ---
stocks = load_companies_from_json()
start_year = 2023
end_year = datetime.now().year
training_windows = [30,60, 90, 120,150]
poly_degrees = [2]
prediction_windows = [3,7]  # 1-week, 2-weeks, 3-weeks
monthly_trends = []

def download_stock_data(stock, start_date, end_date):
    data = yf.download(stock, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data[['Open']]

def prepare_data(df):
    df.index = pd.to_datetime(df.index)
    X = df.index.map(lambda x: x.toordinal()).values.reshape(-1, 1)
    y = df['Open'].values
    return X, y

def calculate_trend(prices):
    return 'Up' if prices[-1] > prices[0] else 'Down'

for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        if year == datetime.now().year and month > datetime.now().month:
            continue
        
        forecast_start = pd.to_datetime(f'{year}-{month:02d}-01')
        forecast_end = forecast_start + pd.DateOffset(days=31)
        
        for stock in stocks:
            print(f"\n--- Processing {stock} for {year}-{month:02d} ---")
            
            data_start = forecast_start - pd.DateOffset(days=max(training_windows) + 31)
            df = download_stock_data(stock, data_start.strftime('%Y-%m-%d'), forecast_end.strftime('%Y-%m-%d'))
            
            monthly_trend_row = {'Stock': stock, 'Year': year, 'Month': month}
            actual_prices = df[(df.index >= forecast_start) & (df.index < forecast_end)]['Open'].values
            if len(actual_prices) < 2:
                continue
            
            monthly_trend_row['Actual Trend'] = calculate_trend(actual_prices)
            
            for window in training_windows:
                for degree in poly_degrees:
                    for pred_window in prediction_windows:
                        
                        config_name = f"{window}_days_degree_{degree}_{pred_window}_day_pred"
                        
                        train_start = forecast_start - pd.DateOffset(days=window)
                        train_end = forecast_start - pd.DateOffset(days=1)
                        pred_end = forecast_start + timedelta(days=pred_window)
                        
                        train_data = df[(df.index >= train_start) & (df.index <= train_end)]
                        test_data = df[(df.index >= forecast_start) & (df.index < pred_end)]
                        
                        if train_data.empty or test_data.empty:
                            continue
                        
                        X_train, y_train = prepare_data(train_data)
                        X_test, y_test = prepare_data(test_data)
                        
                        poly_features = PolynomialFeatures(degree=degree)
                        X_train_poly = poly_features.fit_transform(X_train)
                        X_test_poly = poly_features.transform(X_test)
                        
                        model = LinearRegression()
                        model.fit(X_train_poly, y_train)
                        predictions = model.predict(X_test_poly)
                        
                        predicted_trend = calculate_trend(predictions)
                        monthly_trend_row[config_name] = predicted_trend
                        monthly_trend_row[f"{config_name}_Accuracy"] = 1 if predicted_trend == monthly_trend_row['Actual Trend'] else 0
                        
            monthly_trends.append(monthly_trend_row)

output_dir = 'monthly_trend_analysis'
os.makedirs(output_dir, exist_ok=True)
monthly_trends_df = pd.DataFrame(monthly_trends)
output_file = f'{output_dir}/weekly_trends_analysis.csv'
monthly_trends_df.to_csv(output_file, index=False)
print(f"\nWeekly trends saved in CSV: {output_file}")
