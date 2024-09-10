import os
import pandas as pd

### create prices dataframe ###
# paths to data folders
raw_data_folder = 'Data/raw data'
processed_data_folder = 'Data/processed data'

# create a list of the dataframes for the different years
dataframes = []
for filename in sorted(os.listdir(raw_data_folder)):
    if filename.startswith('Day-ahead_Prices_') and filename.endswith('.csv'):
        file_path = os.path.join(raw_data_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# concatenate dataframe list to one dataframe
day_ahead_prices = pd.concat(dataframes, ignore_index=True)
# delete columns "Currency" en "BZN|NL" from the datadrame
day_ahead_prices = day_ahead_prices.drop(columns=["Currency", "BZN|NL"], errors='ignore')

# create date and hour column
day_ahead_prices['date'] = day_ahead_prices['MTU (CET/CEST)'].str.split(' - ').str[0]
day_ahead_prices['datetime'] = pd.to_datetime(day_ahead_prices['date'], format='%d.%m.%Y %H:%M', errors='coerce')
day_ahead_prices = day_ahead_prices.drop(columns=['MTU (CET/CEST)'])
day_ahead_prices['date'] = day_ahead_prices['datetime'].dt.strftime('%d-%m-%Y')
day_ahead_prices['hour'] = day_ahead_prices['datetime'].dt.hour
day_ahead_prices = day_ahead_prices.drop(columns=["datetime"])

# change name column and rearrange columns
day_ahead_prices = day_ahead_prices.rename(columns={"Day-ahead Price [EUR/MWh]": "price"})
day_ahead_prices = day_ahead_prices[['date', 'hour', 'price']]

# save preprocessed price data
output_path = os.path.join(processed_data_folder, 'prices.csv')
day_ahead_prices.to_csv(output_path, index=False)

### create total load dataframe ###
# read raw data
file_path = os.path.join(raw_data_folder, "Total Load - Day Ahead _ Actual_2015-2024.csv")
total_load = pd.read_csv(file_path)

# split datetime in date and hour column
total_load['start_datetime'] = pd.to_datetime(total_load['start_datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
total_load['date'] = total_load['start_datetime'].dt.strftime('%d-%m-%Y')
total_load['hour'] = total_load['start_datetime'].dt.hour
total_load = total_load.drop(columns=['start_datetime'])

# rename columns and rearrange dataframe
total_load = total_load.rename(columns={'Day-ahead Total Load Forecast [MW] - BZN|NL': 'load_forecast',
                                        'Actual Total Load [MW] - BZN|NL': 'actual_load'})
total_load = total_load[["date", "hour", "load_forecast", "actual_load"]]

# create new df with hourly load
total_load['block'] = (total_load['hour'] != total_load['hour'].shift()).cumsum()
hourly_load_data = total_load.groupby('block').agg({
    'date': 'first',  # Neem de eerste waarde van 'date' van elk blok
    'hour': 'first',  # Neem het eerste uur van elk blok
    'load_forecast': 'sum',
    'actual_load': 'sum'
}).reset_index(drop=True)

# save total load dataframe
output_path = os.path.join(processed_data_folder, 'load.csv')
hourly_load_data.to_csv(output_path, index=False)



