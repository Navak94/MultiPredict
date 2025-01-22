import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime
import os
import csv
import json
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Initialize CSV file with headers (if it doesn't exist)
def initialize_csv(filename="lstm_stock_trends.csv"):
    if not os.path.exists(filename):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Company", "Today Value", "Predicted Value", "Trend"])
        print(f"\n✅ CSV initialized: {filename}")

# Function to append results to CSV
def append_to_csv(Company, today_val, predicted_val, trend_val, filename="lstm_stock_trends.csv"):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([Company, today_val, predicted_val, trend_val])
    print(f"✅ Appended {Company} data to {filename}")

# Load the stock list from the config file
def load_config_json(path="companies.json"):
    with open(path, 'r') as file:
        config = json.load(file)
    return config['companies']

# Create the LSTM model
def create_lstm_model():
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(30, 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)  # Single neuron for regression output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to create sequences for LSTM training
def create_sequences(data, sequence_length=30):
    sequences, labels = [], []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Prepare data and train the model
companies = load_config_json()
sequence_length = 30
training_days = 120

# Initialize the CSV before running
initialize_csv()

for Company in companies:
    try:
        # Fetch stock data
        data = yf.download(Company, start=(datetime.datetime.now() - datetime.timedelta(days=training_days)), end=datetime.datetime.now())
        df = data['Close'].values.reshape(-1, 1)

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_normalized = scaler.fit_transform(df)

        # Create sequences
        X, y = create_sequences(df_normalized, sequence_length)
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

        # Split data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Initialize and train the LSTM model
        model = create_lstm_model()
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

        # Forecasting the next 30 days
        last_sequence = X[-1].reshape(1, sequence_length, 1)
        future_predictions = []

        for _ in range(30):
            next_predicted_value = model.predict(last_sequence)[0][0]
            future_predictions.append(next_predicted_value)
            next_value = np.array([[next_predicted_value]])
            last_sequence = np.append(last_sequence[:, 1:, :], next_value.reshape(1, 1, 1), axis=1)

        # Rescale predictions back to original values
        future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        last_120_days = scaler.inverse_transform(df_normalized[-120:])

        # Determine the trend based on the predicted values
        today_value = last_120_days[-1][0]
        predicted_value = future_predictions_rescaled[-1][0]
        trend_val = "Very High Upward Trend" if predicted_value > today_value * 1.2 else "Upward Trend" if predicted_value > today_value else "Downward Trend"

        # Append the results to the CSV
        append_to_csv(Company, today_value, predicted_value, trend_val)

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(data.index[-120:], last_120_days, label='Last 120 Days')
        plt.plot(data.index[-1] + pd.to_timedelta(np.arange(1, 31), unit='D'), 
                 future_predictions_rescaled, label='LSTM Predictions', color='orange')
        plt.title(f'{Company} Stock Price Prediction with TensorFlow LSTM')
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.legend()
        plt.savefig(f'stock_predictions/{Company}/{Company}_lstm_tensorflow_prediction.png', dpi=300)
        plt.close()

        print(f"✅ Completed LSTM prediction for {Company}")
    except Exception as e:
        print(f"❌ Error processing {Company}: {e}")

print("✅ LSTM Stock Prediction Completed")
