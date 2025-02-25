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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Define how many years of historical data to fetch
HISTORICAL_YEARS = 10  # Try to get up to 10 years of data
SEQUENCE_LENGTH = 30  # Past days used for prediction
MIN_REQUIRED_DAYS = 500  # Minimum data points required (2 years of daily data)

# Initialize CSV file with headers
def initialize_csv(filename="lstm_stock_trends.csv"):
    if not os.path.exists(filename):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Company", "Today Value", "Predicted Value", "Trend"])
        print(f"\n✅ CSV initialized: {filename}")

# Append prediction results to CSV
def append_to_csv(Company, today_val, predicted_val, trend_val, filename="lstm_stock_trends.csv"):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([Company, today_val, predicted_val, trend_val])
    print(f"✅ Appended {Company} data to {filename}")

# Load stock list from config
def load_config_json(path="companies.json"):
    with open(path, 'r') as file:
        config = json.load(file)
    return config['companies']

# Create the LSTM model with regularization
def create_lstm_model():
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, 1)),  # Handle variable-length sequences
        LSTM(256, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(1)  # Single neuron for regression output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# Create sequences for LSTM
def create_sequences(data, sequence_length=30):
    sequences, labels = [], []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Fetch and process stock data
def fetch_stock_data(company):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=HISTORICAL_YEARS * 365)

    # Download stock data
    data = yf.download(company, start=start_date, end=end_date)

    if data.empty or len(data) < MIN_REQUIRED_DAYS:
        raise ValueError(f"Not enough data for {company}")

    # Extract closing prices
    df = data['Close'].values.reshape(-1, 1)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_normalized = scaler.fit_transform(df)

    return df_normalized, scaler, data.index

# Train and predict
def train_and_predict(company):
    try:
        # Fetch stock data
        df_normalized, scaler, dates = fetch_stock_data(company)

        # Create sequences
        X, y = create_sequences(df_normalized, SEQUENCE_LENGTH)
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

        # Split into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train model
        model = create_lstm_model()
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1, 
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

        # Forecast the next 30 days
        last_sequence = X[-1].reshape(1, SEQUENCE_LENGTH, 1)
        future_predictions = []

        for _ in range(30):
            next_predicted_value = model.predict(last_sequence)[0][0]
            future_predictions.append(next_predicted_value)
            next_value = np.array([[next_predicted_value]])
            last_sequence = np.append(last_sequence[:, 1:, :], next_value.reshape(1, 1, 1), axis=1)

        # Rescale predictions back to original values
        future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        last_120_days = scaler.inverse_transform(df_normalized[-120:])

        # Determine trend
        today_value = last_120_days[-1][0]
        predicted_value = future_predictions_rescaled[-1][0]
        trend_val = "Very High Upward Trend" if predicted_value > today_value * 1.2 else "Upward Trend" if predicted_value > today_value else "Downward Trend"

        # Append results
        append_to_csv(company, today_value, predicted_value, trend_val)

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(dates[-120:], last_120_days, label='Last 120 Days')
        plt.plot(dates[-1] + pd.to_timedelta(np.arange(1, 31), unit='D'), 
                 future_predictions_rescaled, label='LSTM Predictions', color='orange')
        plt.title(f'{company} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.legend()

        output_dir = f'stock_predictions/{company}/'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}{company}_lstm_tensorflow_prediction.png', dpi=300)
        plt.close()

        print(f"✅ Completed LSTM prediction for {company}")

    except Exception as e:
        print(f"❌ Error processing {company}: {e}")

# Main execution
if __name__ == "__main__":
    initialize_csv()
    companies = load_config_json()

    for company in companies:
        train_and_predict(company)

    print("✅ LSTM Stock Prediction Completed")
