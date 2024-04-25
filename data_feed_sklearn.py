import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import yfinance as yf
import datetime
import os
import getpass
import csv


Company = "RTX"


# Define directory path
directory = f"C:\\Users\\{getpass.getuser()}\\Desktop\\pyStockData\\"

# Create directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Define CSV file path
csv_file = f"{directory}{Company}_data.csv"

# Open the CSV file for writing with explicit encoding
with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"])  # Write header row

    try:
        # Fetch data from yfinance for the specified company
        data = yf.download(Company, start="2023-01-01", end="2024-12-31")

        # Write data to CSV file
        for index, row in data.iterrows():
            dividend = row.get('Dividends', None)  # Get dividend value or None if column doesn't exist
            # Remove unwanted character (single quote) from the data before writing
            cleaned_row = [index.strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], dividend, row.get('Stock Splits', None)]
            cleaned_row = [str(cell).replace("'", "") for cell in cleaned_row]
            writer.writerow(cleaned_row)

    except Exception as e:
        print("Error fetching data:", e)


# Step 1: Read the CSV file
df = pd.read_csv(f'C:\\Users\\{getpass.getuser()}\\Desktop\\pyStockData\\{Company}_data.csv', usecols=['Date', 'Open'])

# Step 2: Prepare Data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 3: Split Data
X = df.index.map(lambda x: x.toordinal()).values.reshape(-1, 1)  # Feature (end_date)
y = df['Open'].values   # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Step 5: Define the time periods
start_date_future = pd.to_datetime("2024-05-01")
end_date_future = pd.to_datetime("2024-11-08")
start_date_past = pd.to_datetime("2023-01-01")
end_date_past = pd.to_datetime("2024-04-30")

# Step 6: Generate future dates within the specified time period
future_dates = pd.date_range(start=start_date_future, end=end_date_future, freq='D')

# Step 7: Generate predictions for future dates
future_dates_numeric = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
future_dates_poly = poly_features.transform(future_dates_numeric)
future_predictions = model.predict(future_dates_poly)

# Step 8: Generate past dates within the specified time period
past_dates = pd.date_range(start=start_date_past, end=end_date_past, freq='D')

# Step 9: Generate predictions for past dates
past_dates_numeric = np.array([date.toordinal() for date in past_dates]).reshape(-1, 1)
past_dates_poly = poly_features.transform(past_dates_numeric)
past_predictions = model.predict(past_dates_poly)

# Step 10: Plot Original Data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Open'], color='blue', label='Original Data')

# Step 11: Plot Future Predictions
plt.plot(future_dates, future_predictions, color='green', label='Future Predictions')

# Step 12: Plot Past Predictions
plt.plot(past_dates, past_predictions, color='red', label='Past Predictions')

plt.title(Company)
plt.xlabel('Date')
plt.ylabel('Open')
plt.legend()
plt.grid(True)
plt.show()
