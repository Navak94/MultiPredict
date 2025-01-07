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
from datetime import datetime, timedelta
import json

def load_config_json(path="companies.json"):
    with open(path, 'r') as file:
        config = json.load(file)
    return config['companies']

#training_data_days_list = [15,30,90,120,150,200,300]
training_data_days_list = [120]
#companies = ["ADBE","AMD", "NVDA", "INTC", "AAPL", "GOOGL", "HPQ","MSFT","TSLA","AMZN","T","VZ","BA","NOC", "GE", "RTX" ,"GD","LMT","CVS","WMT","UNH","META","JNJ"]
companies = load_config_json()
total_invest=0
total_return=0

for training_data_days in training_data_days_list:
    # Get the current date and time
    now = datetime.now()

    # Format the date as a string in "YYYY-MM-DD" format
    formatted_date = now.strftime("%Y-%m-%d")
    current_date = datetime.strptime(formatted_date, "%Y-%m-%d")
    next_day_date = current_date + timedelta(days=1)
    past_date_beginning_data=current_date - timedelta(days=training_data_days)
    future_date_beginning_data=current_date + timedelta(days=30)
    next_year = current_date + timedelta(days=365)
    #Start and end of real data points
    start_time_past=past_date_beginning_data
    end_time_past=formatted_date

    #Try to predict in this time window
    start_time_future=next_day_date
    end_time_future=future_date_beginning_data
    for Company in companies:

        #Define directory path
        directory = f"C:\\Users\\{getpass.getuser()}\\Desktop\\pyStockData\\"

        #Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        #Define CSV file path
        csv_file = f"{directory}{Company}_data.csv"

        #Open the CSV file for writing with explicit encoding
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"])  # Write header row

            try:
                # Fetch data from yfinance for the specified company
                data = yf.download(Company, start=start_time_past, end=next_year)

                # Write data to CSV file
                for index, row in data.iterrows():
                    dividend = row.get('Dividends', None)  # Get dividend value or None if column doesn't exist
                    # Remove unwanted character (single quote) from the data before writing
                    cleaned_row = [index.strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], dividend, row.get('Stock Splits', None)]
                    cleaned_row = [str(cell).replace("'", "") for cell in cleaned_row]
                    writer.writerow(cleaned_row)

            except Exception as e:
                print("Error fetching data:", e)


        #Read the CSV file
        df = pd.read_csv(f'C:\\Users\\{getpass.getuser()}\\Desktop\\pyStockData\\{Company}_data.csv', usecols=['Date', 'Open'])

        #Prepare Data
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        #Split Data
        X = df.index.map(lambda x: x.toordinal()).values.reshape(-1, 1)  # Feature (end_date)
        y = df['Open'].values   # Target variable
        today_val = float(y[len(y)-1])
        #total_invest = total_invest + today_val
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Model Training
        poly_features = PolynomialFeatures(degree=3)
        X_train_poly = poly_features.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        #Define the time periods
        start_date_future = pd.to_datetime(start_time_future)
        end_date_future = pd.to_datetime(end_time_future)
        start_date_past = pd.to_datetime(start_time_past)
        end_date_past = pd.to_datetime(end_time_past)

        #Generate future dates within the specified time period
        future_dates = pd.date_range(start=start_date_future, end=end_date_future, freq='D')

        #Generate predictions for future dates
        future_dates_numeric = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
        future_dates_poly = poly_features.transform(future_dates_numeric)
        future_predictions = model.predict(future_dates_poly)
        
        #getting trend
        predicted_val = float(future_predictions[len(future_predictions)-1])
        predicted_val_first = float(future_predictions[0])
        #total_return = total_return + predicted_val
        trend_val=predicted_val-predicted_val_first
        term=""
        if training_data_days == 30:
            term= "short term"
        if training_data_days == 120:
            term = "long term"
        if trend_val > 1:
            #print(f'{Company} {term} upward Trend!')
            if trend_val > 1.7:
                print(f'{Company} {term} HIGH upward Trend!')
                total_invest = total_invest + today_val
                total_return = total_return + predicted_val
        if trend_val < 1:
            #print(f'{Company} {term} downward Trend!')
            pass
        
        #Generate past dates within the specified time period
        past_dates = pd.date_range(start=start_date_past, end=end_date_past, freq='D')

        #Generate predictions for past dates
        past_dates_numeric = np.array([date.toordinal() for date in past_dates]).reshape(-1, 1)
        past_dates_poly = poly_features.transform(past_dates_numeric)
        past_predictions = model.predict(past_dates_poly)

        #Plot Original Data
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Open'], color='blue', label='Original Data')

        #Plot Future Predictions
        plt.plot(future_dates, future_predictions, color='green', label='Future Predictions')

        #Plot Past Predictions
        plt.plot(past_dates, past_predictions, color='red', label='Past Predictions')

        plt.title(Company)
        plt.xlabel('Date')
        plt.ylabel('Open')
        plt.legend()
        plt.grid(True)
        plot_folder_save= f'{directory}\\{Company}\\'
        
        if not os.path.exists(plot_folder_save):
            os.makedirs(plot_folder_save)
            
        plot_filename = f'{plot_folder_save}{Company}_{training_data_days}_training_days_data.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

percent_return = round(100*(total_return/total_invest)-100,2)
total_invest = round(total_invest,2)
total_return = round(total_return,2)
print("if you bought everything it'd cost $" , total_invest , " and the value 30 days from now would be $" , total_return , " which is a " ,percent_return , "% return")

