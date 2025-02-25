import json
import yfinance as yf
from datetime import datetime, timedelta
import csv

# Load stock tickers from the JSON file
with open('companies.json', 'r') as file:
    data = json.load(file)
    companies = data['companies']

# Get date range for the past 30 days
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

# Function to check stock movement
def check_stock_movement(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    if len(hist) == 0:
        return f"{ticker}: No data available", "NO DATA"

    # Get opening price 30 days ago and latest closing price
    start_price = hist['Close'].iloc[0]
    end_price = hist['Close'].iloc[-1]

    # Determine if stock went up or down
    if end_price > start_price:
        return f"{ticker}: UP ({start_price:.2f} → {end_price:.2f})", "UP"
    elif end_price < start_price:
        return f"{ticker}: DOWN ({start_price:.2f} → {end_price:.2f})", "DOWN"
    else:
        return f"{ticker}: NO CHANGE ({start_price:.2f})", "NO CHANGE"

# Prepare the CSV file
output_file = 'stock_movements.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Company", "Outcome"])  # Write the header

    # Check each stock and write the result to the CSV
    for company in companies:
        _, outcome = check_stock_movement(company)
        writer.writerow([company, outcome])
        print(f"{company}: {outcome}")

print(f"\nResults saved in '{output_file}'")
