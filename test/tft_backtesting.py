import pandas as pd

def clean_and_reformat_csv(raw_csv_path, reformatted_csv_path):
    # Load the CSV while handling header issues
    df = pd.read_csv(raw_csv_path, skiprows=1)  # Skip first row if it contains stock symbol
    
    # Drop rows with NaN values in critical columns
    df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    
    # Ensure 'Date' is properly formatted
    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)  # Handle unnamed column issue
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Select only necessary columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Save cleaned file
    df.to_csv(reformatted_csv_path, index=False)
    print(f'✅ Cleaned and reformatted CSV saved to {reformatted_csv_path}')

# Example usage
raw_csv_path = "tft_results/AAPL_raw.csv"
reformatted_csv_path = "tft_results/AAPL_reformatted.csv"
clean_and_reformat_csv(raw_csv_path, reformatted_csv_path)