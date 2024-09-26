import os
import pandas as pd
import numpy as np

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

# change name column and rearrange columns
day_ahead_prices = day_ahead_prices.rename(columns={"Day-ahead Price [EUR/MWh]": "price"})
day_ahead_prices = day_ahead_prices[['datetime', 'date', 'hour', 'price']]

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

# rename columns and rearrange dataframe
total_load = total_load.rename(columns={'Day-ahead Total Load Forecast [MW] - BZN|NL': 'load_forecast',
                                        'Actual Total Load [MW] - BZN|NL': 'actual_load'})
total_load = total_load[["start_datetime", "date", "hour", "load_forecast", "actual_load"]]

# create new df with hourly load
total_load['block'] = (total_load['hour'] != total_load['hour'].shift()).cumsum()
hourly_load_data = total_load.groupby('block').agg({
    'date': 'first',  # Neem de eerste waarde van 'date' van elk blok
    'hour': 'first',  # Neem het eerste uur van elk blok
    'load_forecast': 'sum',
    'actual_load': 'sum'
}).reset_index(drop=True)
hourly_load_data['date'] = pd.to_datetime(hourly_load_data['date'], format='%d-%m-%Y')

# Stap 2: Maak een nieuwe 'Datetime' kolom door de 'Hour' kolom toe te voegen aan de 'Date' kolom
hourly_load_data['datetime'] = hourly_load_data['date'] + pd.to_timedelta(hourly_load_data['hour'], unit='h')

# save total load dataframe
output_path = os.path.join(processed_data_folder, 'load.csv')
hourly_load_data.to_csv(output_path, index=False)

### create total production dataframe ###
file_path = os.path.join(raw_data_folder, "Total_production_ned_2016-2024.csv")
total_production = pd.read_csv(file_path)

# create datetime column
total_production = total_production[["ValidFrom", "Volume"]]
total_production['datetime'] = pd.to_datetime(total_production['ValidFrom']).dt.tz_localize(None)
total_production = total_production.drop(columns=['ValidFrom'])

# save total production dataframe
output_path = os.path.join(processed_data_folder, 'production.csv')
total_production.to_csv(output_path, index=False)

### create dataframe TTF gas ###
file_path = os.path.join(raw_data_folder, "TTF Gas (EUR:MWh) 2016-2024.csv")
ttf_gas = pd.read_csv(file_path)

# create datetime column per hour
ttf_gas = ttf_gas[["Date", "Price"]]
ttf_gas['Date'] = pd.to_datetime(ttf_gas['Date'])
# add missing dates
start_date = ttf_gas['Date'].min()
end_date = ttf_gas['Date'].max()
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
full_ttf_gas = pd.DataFrame({'Date': all_dates})
full_ttf_gas = pd.merge(full_ttf_gas, ttf_gas, on='Date', how='left')
# fill al empty dates with last value available
full_ttf_gas['Price'] = full_ttf_gas['Price'].ffill()
# converting dates to 24 hour
full_ttf_gas = full_ttf_gas.loc[full_ttf_gas.index.repeat(24)].reset_index(drop=True)
full_ttf_gas['Hour'] = np.tile(np.arange(24), len(full_ttf_gas) // 24)
full_ttf_gas['Date'] = full_ttf_gas['Date'] + pd.to_timedelta(full_ttf_gas['Hour'], unit='h')
full_ttf_gas = full_ttf_gas.drop(columns=['Hour'])
# sort dates in ascending order
full_ttf_gas = full_ttf_gas.sort_values('Date').reset_index(drop=True)
ttf_gas = full_ttf_gas

# save TTF gas dataframe
output_path = os.path.join(processed_data_folder, 'ttf_gas.csv')
ttf_gas.to_csv(output_path, index=False)

### create dataframe EUA ###
file_path = os.path.join(raw_data_folder, "EUA (EUR:tCO2) 2016-2024.xlsx")
EUA = pd.read_excel(file_path)

# add missing dates and convert to 24 hour
EUA['Date'] = pd.to_datetime(EUA['Date'])
full_date_range = pd.date_range(start=EUA['Date'].min(), end=EUA['Date'].max(), freq='D')
full_dates = pd.DataFrame(full_date_range, columns=['Date'])
EUA_full = pd.merge(full_dates, EUA, on='Date', how='left')
EUA_full['Price'] = EUA_full['Price'].ffill()
EUA_full = EUA_full.loc[EUA_full.index.repeat(24)].reset_index(drop=True)
EUA_full['Hour'] = list(range(24)) * len(full_dates)
EUA_full['Datetime'] = EUA_full['Date'] + pd.to_timedelta(EUA_full['Hour'], unit='h')
EUA_full.drop(columns=['Date', 'Hour'], inplace=True)
EUA = EUA_full

# save EUA gas dataframe
output_path = os.path.join(processed_data_folder, 'EUA.csv')
EUA.to_csv(output_path, index=False)

### create dataframe API2 coal ###
# read the API2 coal data and the exchange data
file_path = os.path.join(raw_data_folder, "API2 Coal (USD:t) 2016-2024.csv")
API2_coal = pd.read_csv(file_path, delimiter=",", quotechar='"')
API2_coal = API2_coal[["Date","Price"]]
API2_coal['Date'] = pd.to_datetime(API2_coal['Date'])
file_path = os.path.join(raw_data_folder, "USD-EUR exchange rates.csv")
exchange = pd.read_csv(file_path)
exchange = exchange[['Date', "Price"]]
exchange['Date'] = pd.to_datetime(exchange['Date'])

# convert prices to EUR/t
merged_df = pd.merge(API2_coal, exchange, on='Date', suffixes=('_USD', '_EUR'))
merged_df['Price_EUR/t'] = merged_df['Price_USD'] * merged_df['Price_EUR']
merged_df.drop(columns=['Price_USD', 'Price_EUR'], inplace=True)
merged_df.sort_values(by='Date', ascending=True, inplace=True)
merged_df.rename(columns={'Price_EUR/t': 'Price'}, inplace=True)

# add missing dates and convert to 24 hour
full_date_range = pd.date_range(start=merged_df['Date'].min(), end=merged_df['Date'].max(), freq='D')
full_dates = pd.DataFrame(full_date_range, columns=['Date'])
merged_full = pd.merge(full_dates, merged_df, on='Date', how='left')
merged_full['Price'] = merged_full['Price'].ffill()
merged_full = merged_full.loc[merged_full.index.repeat(24)].reset_index(drop=True)
merged_full['Hour'] = list(range(24)) * len(full_dates)
merged_full['Datetime'] = merged_full['Date'] + pd.to_timedelta(merged_full['Hour'], unit='h')
merged_full.drop(columns=['Date', 'Hour'], inplace=True)
API2_coal = merged_full

# save API2 Coal dataframe
output_path = os.path.join(processed_data_folder, 'API2_coal.csv')
API2_coal.to_csv(output_path, index=False)

### create dataframe for Brent Oil
# read the brent oil data and the exchange data
file_path = os.path.join(raw_data_folder, "Brent Oil (USD:bbL) 2016-2024.xlsx")
brent_oil = pd.read_excel(file_path)
brent_oil['Date'] = pd.to_datetime(brent_oil['Date'])
file_path = os.path.join(raw_data_folder, "USD-EUR exchange rates.csv")
exchange = pd.read_csv(file_path)
exchange = exchange[['Date', "Price"]]
exchange['Date'] = pd.to_datetime(exchange['Date'])

# convert prices to EUR/t
merged_df = pd.merge(brent_oil, exchange, on='Date', suffixes=('_USD', '_EUR'))
merged_df['Price'] = merged_df['Price_USD'] * merged_df['Price_EUR']
merged_df.drop(columns=['Price_USD', 'Price_EUR'], inplace=True)

# add missing dates and convert to 24 hour
full_date_range = pd.date_range(start=merged_df['Date'].min(), end=merged_df['Date'].max(), freq='D')
full_dates = pd.DataFrame(full_date_range, columns=['Date'])
merged_full = pd.merge(full_dates, merged_df, on='Date', how='left')
merged_full['Price'] = merged_full['Price'].ffill()
merged_full = merged_full.loc[merged_full.index.repeat(24)].reset_index(drop=True)
merged_full['Hour'] = list(range(24)) * len(full_dates)
merged_full['Datetime'] = merged_full['Date'] + pd.to_timedelta(merged_full['Hour'], unit='h')
merged_full.drop(columns=['Date', 'Hour'], inplace=True)
brent_oil = merged_full

# save API2 Coal dataframe
output_path = os.path.join(processed_data_folder, 'brent_oil.csv')
brent_oil.to_csv(output_path, index=False)


test = 5


