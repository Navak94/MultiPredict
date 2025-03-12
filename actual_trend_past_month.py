import pandas as pd
import yfinance as yf
import json
import datetime

def load_config_json(path="companies.json"):
    with open(path, 'r') as file:
        config = json.load(file)
    return config['companies']

def get_stock_trend(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        print(f"\nTicker: {ticker}")
        print(data.head())  # Print first few rows
        print("Columns:", data.columns)  # Check if 'Close' exists
        print("NaN count:", data.isna().sum())  # See missing values

        # Fix: Adjust for MultiIndex DataFrame
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data[('Close', ticker)].dropna() if ('Close', ticker) in data.columns else None
        else:
            close_prices = data['Close'].dropna() if 'Close' in data.columns else None

        if close_prices is None or close_prices.empty or len(close_prices) < 2:
            return "NO DATA"

        first_price = close_prices.iloc[0]
        last_price = close_prices.iloc[-1]

        if last_price > first_price:
            trend = "UP"
        elif last_price < first_price:
            trend = "DOWN"
        else:
            trend = "EVEN"

        return trend
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return "ERROR"



def main():
    companies = load_config_json()
    
    today = datetime.date.today()
    one_month_ago = today - datetime.timedelta(days=30)
    
    results = []
    
    for company in companies:
        trend = get_stock_trend(company, one_month_ago, today)
        results.append({"Company": company, "Trend": trend})
    
    df = pd.DataFrame(results)
    output_file = "stock_trends.csv"
    df.to_csv(output_file, index=False)
    print(f"Stock trend analysis saved to {output_file}")

if __name__ == "__main__":
    main()
