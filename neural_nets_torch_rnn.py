import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent Tkinter issues
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Load the stock list from the config file
def load_config_json(path="companies.json"):
    with open(path, 'r') as file:
        config = json.load(file)
    return config['companies']

# RNN model definition with Dropout
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last output for prediction
        return out

# Function to create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return sequences

# Hyperparameters
sequence_length = 30
hidden_size = 256
epochs = 150
learning_rate = 0.0005
clip_value = 1.0
batch_size = 32

# GPU Throttling Settings
if not torch.cuda.is_available():
    raise RuntimeError("CUDA device not available. Ensure you have the correct PyTorch installation and drivers.")
device = torch.device('cuda:0')
torch.set_float32_matmul_precision('medium')
torch.cuda.set_per_process_memory_fraction(0.5, device=device)
torch.set_num_threads(4)

print(f"Using device: {device}")
print(f"GPU in use: {torch.cuda.get_device_name(0)}")

companies = load_config_json()

for Company in companies:
    data = yf.download(Company, start=(datetime.datetime.now() - datetime.timedelta(days=540)), end=datetime.datetime.now())
    if data.empty:
        print(f"No data found for {Company}, skipping.")
        continue

    # Corrected data scaling to avoid incorrect price representation
    df = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_normalized = scaler.fit_transform(df)
    df_normalized = torch.tensor(df_normalized, dtype=torch.float32).to(device)

    sequences = create_sequences(df_normalized, sequence_length)
    X = torch.stack([s[0] for s in sequences]).to(device)
    X = X.view(X.size(0), sequence_length, 1)
    y = torch.tensor([s[1] for s in sequences]).to(device)
    y = y.view(-1, 1)

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = StockLSTM(input_size=1, hidden_size=256).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X).view(-1, 1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            epoch_loss += loss.item()

    model.eval()
    last_sequence = X[-1].unsqueeze(0).clone().to(device)
    future_predictions = []

    future_dates = pd.date_range(start=datetime.datetime.now() + datetime.timedelta(days=1), periods=30)
    for _ in range(30):
        with torch.no_grad():
            next_prediction = model(last_sequence).item()
            next_input = torch.tensor([[next_prediction]], dtype=torch.float32).to(device)
            last_sequence = torch.cat((last_sequence[:, 1:], next_input.unsqueeze(0)), dim=1)
            future_predictions.append(next_prediction)

    # Properly rescale both past data and predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    last_120_days = scaler.inverse_transform(df[-120:])

    print(f"Tomorrow's Predicted Value for {Company}: {future_predictions[0]:.2f}")
    print(f"Predicted Value for {Company} 30 days from tomorrow: {future_predictions[-1]:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-120:], last_120_days, label="Last 120 Days")
    plt.plot(future_dates, future_predictions, label="RNN Predictions", color='orange')
    plt.title(f'{Company} Stock Price Prediction with PyTorch RNN')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    os.makedirs('stock_predictions', exist_ok=True)
    plt.savefig(f'stock_predictions/{Company}_stock_prediction.png', dpi=300)
    plt.close()
