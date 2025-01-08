import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import os
import getpass
import csv
import json
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load the stock list from the config file
def load_config_json(path="companies.json"):
    with open(path, 'r') as file:
        config = json.load(file)
    return config['companies']

# Load stock data
companies = load_config_json()
training_data_days_list = [120]
total_invest = 0
total_return = 0

def create_neural_net_model():
    model = Sequential([
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)  # Single neuron for regression output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

for training_data_days in training_data_days_list:
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    current_date = datetime.datetime.strptime(formatted_date, "%Y-%m-%d")
    next_day_date = current_date + datetime.timedelta(days=1)
    past_date_beginning_data = current_date - datetime.timedelta(days=training_data_days)
    next_year = current_date + datetime.timedelta(days=365)

    for Company in companies:
        directory = f"C:\\Users\\{getpass.getuser()}\\Desktop\\pyStockData\\"
        if not os.path.exists(directory):
            os.makedirs(directory)

        csv_file = f"{directory}{Company}_data.csv"

        # Fetch data from Yahoo Finance
        try:
            data = yf.download(Company, start=past_date_beginning_data, end=next_year)
            data.to_csv(csv_file)
        except Exception as e:
            print(f"Error fetching data for {Company}: {e}")
            continue

        # Read CSV and preprocess data
        df = pd.read_csv(csv_file, usecols=['Date', 'Open'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Prepare data for training
        X = df.index.map(lambda x: x.toordinal()).values.reshape(-1, 1)
        y = df['Open'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Build and train the neural network
        model = create_neural_net_model()
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

        # Generate future predictions iteratively (one day at a time)
        future_dates_numeric = np.array([date.toordinal() for date in pd.date_range(start=next_day_date, periods=30)]).reshape(-1, 1)
        future_dates_scaled = scaler.transform(future_dates_numeric)
        future_predictions_adjusted = []
        last_input = future_dates_scaled[0].reshape(1, -1)  # Start with the first future date

        for i in range(30):
            predicted_value = model.predict(last_input)[0][0]  # Predict next day price
            future_predictions_adjusted.append(predicted_value)
            last_input = last_input + 1  # Move to the next day numerically

        future_predictions_adjusted = np.array(future_predictions_adjusted).reshape(-1, 1)

        # Plot results with adjusted predictions
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Open'], label='Original Data')
        plt.plot(pd.date_range(start=next_day_date, periods=30), future_predictions_adjusted, label='Neural Network Sequential Predictions', color='orange')
        plt.title(f'{Company} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Open Price')
        plt.legend()
        plt.grid(True)

        plot_folder_save = f'{directory}{Company}\\'
        if not os.path.exists(plot_folder_save):
            os.makedirs(plot_folder_save)
        plt.savefig(f'{plot_folder_save}{Company}_NN_{training_data_days}_training_days.png', dpi=300)
        plt.close()

print("Neural Network Stock Analysis Completed")
