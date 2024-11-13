"""
Preprocessing of the data, obtains a data frame with the data ready to be used by the methods
"""

import os
import pandas as pd
import numpy as np


### create prices dataframe ###
# paths to data folders
raw_data_folder = 'Data/raw data'
processed_data_folder = 'Data/processed data'
day_ahead_NL_folder = 'Data/raw data/Day-ahead prices NL'

# create a list of the dataframes for the different years
dataframes = []
for filename in sorted(os.listdir(day_ahead_NL_folder)):
    if filename.startswith('Day-ahead_Prices_') and filename.endswith('.csv'):
        file_path = os.path.join(day_ahead_NL_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# concatenate dataframe list to one dataframe
day_ahead_prices = pd.concat(dataframes, ignore_index=True)
# delete columns "Currency" en "BZN|NL" from the datadrame
day_ahead_prices = day_ahead_prices.drop(columns=["Currency", "BZN|NL"], errors='ignore')

# create date and hour column and interpolate missing prices
day_ahead_prices['date'] = day_ahead_prices['MTU (CET/CEST)'].str.split(' - ').str[0]
day_ahead_prices['Day-ahead Price [EUR/MWh]'] = pd.to_numeric(day_ahead_prices['Day-ahead Price [EUR/MWh]'], errors='coerce')
day_ahead_prices['datetime'] = pd.to_datetime(day_ahead_prices['date'], format='%d.%m.%Y %H:%M', errors='coerce')
day_ahead_prices = day_ahead_prices[day_ahead_prices['datetime'] < '2024-09-28']
missing_prices = day_ahead_prices[day_ahead_prices['Day-ahead Price [EUR/MWh]'].isna()]
missing_datetime_list = missing_prices['datetime'].tolist()
day_ahead_prices['Day-ahead Price [EUR/MWh]'] = day_ahead_prices['Day-ahead Price [EUR/MWh]'].interpolate(method='linear')
day_ahead_prices = day_ahead_prices.drop(columns=['MTU (CET/CEST)'])
day_ahead_prices['date'] = day_ahead_prices['datetime'].dt.strftime('%d-%m-%Y')
day_ahead_prices['hour'] = day_ahead_prices['datetime'].dt.hour

# change name column and rearrange columns
day_ahead_prices = day_ahead_prices.rename(columns={"Day-ahead Price [EUR/MWh]": "price"})
day_ahead_prices = day_ahead_prices[['datetime', 'date', 'hour', 'price']]

# create list with duplicated datetimes (switch from summer time to winter time)
duplicated_datetimes = day_ahead_prices[day_ahead_prices.duplicated(subset='datetime', keep=False)]
duplicated_datetime_list = duplicated_datetimes['datetime'].unique().tolist()

# take mean of the price for switch from summer time to winter time
day_ahead_prices = day_ahead_prices.groupby('datetime', as_index=False).agg({
    'price': 'mean',
    'date': 'first',
    'hour': 'first'
})

# filter data before 28 september 2024
day_ahead_prices['datetime'] = pd.to_datetime(day_ahead_prices['datetime'], errors='coerce')
day_ahead_prices = day_ahead_prices[day_ahead_prices['datetime'] < '2024-09-28']

# save preprocessed price data
output_path = os.path.join(processed_data_folder, 'prices.csv')
day_ahead_prices.to_csv(output_path, index=False)


### create total load dataframe ###
# read raw data
load_NL_folder = os.path.join(raw_data_folder, "Load NL")

# create a list of the dataframes for the different years
dataframes = []
for filename in sorted(os.listdir(load_NL_folder)):
    if filename.startswith('Load forecast') and filename.endswith('.csv'):
        file_path = os.path.join(load_NL_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# concatenate dataframe list to one dataframe and split to date and hour
load = pd.concat(dataframes, ignore_index=True)
load['date'] = load['Time (CET/CEST)'].str.split(' - ').str[0]
load['datetime'] = pd.to_datetime(load['date'], format='%d.%m.%Y %H:%M', errors='coerce')
load['hour'] = load['datetime'].dt.hour
load = load.drop(columns=['Time (CET/CEST)'])
load['date'] = load['datetime'].dt.strftime('%d-%m-%Y')

# drop rows with datetime before January 1, 2016 and after September 28, 2024
load = load[load['datetime'] >= '2016-01-01']
load = load[load['datetime'] < '2024-09-01']

# rename columns and rearrange dataframe
load = load.rename(columns={'Day-ahead Total Load Forecast [MW] - BZN|NL': 'load_forecast',
                                        'Actual Total Load [MW] - BZN|NL': 'actual_load'})
load = load[["datetime", "date", "hour", "load_forecast", "actual_load"]]

# set columns to numeric type
load['load_forecast'] = pd.to_numeric(load['load_forecast'], errors='coerce')
load['actual_load'] = pd.to_numeric(load['actual_load'], errors='coerce')

# create new df with hourly load
load['block'] = (load['hour'] != load['hour'].shift()).cumsum()
hourly_load_data = load.groupby('block').agg({
    'date': 'first',
    'hour': 'first',
    'load_forecast': 'mean',
    'actual_load': 'mean'
}).reset_index(drop=True)
hourly_load_data['date'] = pd.to_datetime(hourly_load_data['date'], format='%d-%m-%Y')

# add datetime column
hourly_load_data['datetime'] = hourly_load_data['date'] + pd.to_timedelta(hourly_load_data['hour'], unit='h')
hourly_load_data['datetime'] = pd.to_datetime(hourly_load_data["datetime"])

# ensure proper transition from winter time to summer time
hourly_load_data['load_forecast'] = pd.to_numeric(hourly_load_data['load_forecast'], errors='coerce')
hourly_load_data['actual_load'] = pd.to_numeric(hourly_load_data['actual_load'], errors='coerce')
hourly_load_data.loc[hourly_load_data['datetime'].isin(missing_datetime_list), 'load_forecast'] = pd.NA
hourly_load_data.loc[hourly_load_data['datetime'].isin(missing_datetime_list), 'actual_load'] = pd.NA
hourly_load_data['load_forecast'] = hourly_load_data['load_forecast'].interpolate(method='linear')
hourly_load_data['actual_load'] = hourly_load_data['actual_load'].interpolate(method='linear')

# ensure proper transition from winter time to summer time
for dt in duplicated_datetime_list:
    hourly_load_data.loc[hourly_load_data['datetime'] == dt, 'load_forecast'] /= 2
    hourly_load_data.loc[hourly_load_data['datetime'] == dt, 'actual_load'] /= 2

# save total load dataframe
output_path = os.path.join(processed_data_folder, 'load.csv')
hourly_load_data.to_csv(output_path, index=False)


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
full_ttf_gas = full_ttf_gas.rename(columns={'Date': 'Datetime'})
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


### create dataframe for wind and solar production ###
wind_and_solar_folder = 'Data/raw data/Wind and solar generation'
# create wind and solar dataframe with day ahead forecasts for solar, wind offshore and wind onshore
dataframes = []
for filename in sorted(os.listdir(wind_and_solar_folder)):
    if filename.startswith('Wind and solar generation forecasts') and filename.endswith('.csv'):
        file_path = os.path.join(wind_and_solar_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)
wind_and_solar = pd.concat(dataframes, ignore_index=True)
columns_to_keep = [
    'MTU (CET/CEST)',
    'Generation - Solar  [MW] Day Ahead/ BZN|NL',
    'Generation - Wind Offshore  [MW] Day Ahead/ BZN|NL',
    'Generation - Wind Onshore  [MW] Day Ahead/ BZN|NL'
]
wind_and_solar = wind_and_solar[columns_to_keep]
# rename columns
wind_and_solar.columns = ['datetime', 'solar', 'wind_offshore', 'wind_onshore']

# create datetime, date and hour
wind_and_solar['datetime'] = wind_and_solar['datetime'].str.split(' - ').str[0]
wind_and_solar['datetime'] = pd.to_datetime(wind_and_solar['datetime'], format='%d.%m.%Y %H:%M')
wind_and_solar['date'] = wind_and_solar['datetime'].dt.strftime('%d-%m-%Y')
wind_and_solar['hour'] = wind_and_solar['datetime'].dt.hour

# set columns to numeric type
wind_and_solar['wind_onshore'] = pd.to_numeric(wind_and_solar['wind_onshore'], errors='coerce')
wind_and_solar['solar'] = pd.to_numeric(wind_and_solar['solar'], errors='coerce')
wind_and_solar['wind_offshore'] = pd.to_numeric(wind_and_solar['wind_offshore'], errors='coerce')

# filter to data before 28 september 2024
wind_and_solar = wind_and_solar[wind_and_solar['datetime'] < '2024-09-28']

# sum up to total generation
wind_and_solar["total_generation"] = wind_and_solar["solar"] + wind_and_solar["wind_onshore"] + wind_and_solar["wind_offshore"]
wind_and_solar["total_wind"] = wind_and_solar["wind_onshore"] + wind_and_solar["wind_offshore"]
wind_and_solar = wind_and_solar.drop(columns=['wind_onshore', 'wind_offshore'])

# create hourly dataframe
wind_and_solar['block'] = (wind_and_solar['hour'] != wind_and_solar['hour'].shift()).cumsum()
hourly_generation_data = wind_and_solar.groupby('block').agg({
    'date': 'first',
    'hour': 'first',
    'total_generation': 'mean',
    'solar': 'mean',
    'total_wind': 'mean',
}).reset_index(drop=True)
hourly_generation_data['date'] = pd.to_datetime(hourly_generation_data['date'], format='%d-%m-%Y')

# add datetime column
hourly_generation_data['datetime'] = hourly_generation_data['date'] + pd.to_timedelta(hourly_generation_data['hour'], unit='h')
hourly_generation_data['datetime'] = pd.to_datetime(hourly_generation_data['datetime'])

# ensure proper transition from winter time to summer time
hourly_generation_data['total_generation'] = pd.to_numeric(hourly_generation_data['total_generation'], errors='coerce')
hourly_generation_data.loc[hourly_generation_data['datetime'].isin(missing_datetime_list), 'total_generation'] = pd.NA
hourly_generation_data['total_generation'] = hourly_generation_data['total_generation'].interpolate(method='linear')
hourly_generation_data['solar'] = pd.to_numeric(hourly_generation_data['solar'], errors='coerce')
hourly_generation_data.loc[hourly_generation_data['datetime'].isin(missing_datetime_list), 'solar'] = pd.NA
hourly_generation_data['solar'] = hourly_generation_data['solar'].interpolate(method='linear')
hourly_generation_data['total_wind'] = pd.to_numeric(hourly_generation_data['total_wind'], errors='coerce')
hourly_generation_data.loc[hourly_generation_data['datetime'].isin(missing_datetime_list), 'total_wind'] = pd.NA
hourly_generation_data['total_wind'] = hourly_generation_data['total_wind'].interpolate(method='linear')

# ensure proper transition from summer time to winter time
for dt in duplicated_datetime_list:
    hourly_generation_data.loc[hourly_generation_data['datetime'] == dt, 'total_generation'] /= 2
    hourly_generation_data.loc[hourly_generation_data['datetime'] == dt, 'solar'] /= 2
    hourly_generation_data.loc[hourly_generation_data['datetime'] == dt, 'total_wind'] /= 2


### create dataframe for actual wind and solar generation ###
actual_generation_folder = 'Data/raw data/Actual generation'
# create wind and solar dataframe with day ahead forecasts for solar, wind offshore and wind onshore
dataframes = []
for filename in sorted(os.listdir(actual_generation_folder)):
    if filename.startswith('Actual') and filename.endswith('.csv'):
        file_path = os.path.join(actual_generation_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)
actual_generation = pd.concat(dataframes, ignore_index=True)
columns_to_keep = [
    'MTU',
    'Solar  - Actual Aggregated [MW]',
    'Wind Offshore  - Actual Aggregated [MW]',
    'Wind Onshore  - Actual Aggregated [MW]'
]
actual_generation = actual_generation[columns_to_keep]
actual_generation.columns = ['datetime', 'actual_solar', 'actual_wind_offshore', 'actual_wind_onshore']

# create datetime, date and hour
actual_generation['datetime'] = actual_generation['datetime'].str.split(' - ').str[0]
actual_generation['datetime'] = pd.to_datetime(actual_generation['datetime'], format='%d.%m.%Y %H:%M')
actual_generation['date'] = actual_generation['datetime'].dt.strftime('%d-%m-%Y')
actual_generation['hour'] = actual_generation['datetime'].dt.hour

# set columns to numeric type
actual_generation['actual_wind_onshore'] = pd.to_numeric(actual_generation['actual_wind_onshore'], errors='coerce')
actual_generation['actual_solar'] = pd.to_numeric(actual_generation['actual_solar'], errors='coerce')
actual_generation['actual_wind_offshore'] = pd.to_numeric(actual_generation['actual_wind_offshore'], errors='coerce')

# filter to data before 28 september 2024
actual_generation = actual_generation[actual_generation['datetime'] < '2024-09-28']

# sum up to total generation
actual_generation["actual_total_generation"] = actual_generation["actual_solar"] + actual_generation["actual_wind_onshore"] + actual_generation["actual_wind_offshore"]
actual_generation["actual_total_wind"] = actual_generation["actual_wind_onshore"] + actual_generation["actual_wind_offshore"]
actual_generation = actual_generation.drop(columns=['actual_wind_onshore', 'actual_wind_offshore'])

# create hourly dataframe
actual_generation['block'] = (actual_generation['hour'] != actual_generation['hour'].shift()).cumsum()
hourly_actual_generation_data = actual_generation.groupby('block').agg({
    'date': 'first',
    'hour': 'first',
    'actual_total_generation': 'mean',
    'actual_solar': 'mean',
    'actual_total_wind': 'mean',
}).reset_index(drop=True)
hourly_actual_generation_data['date'] = pd.to_datetime(hourly_actual_generation_data['date'], format='%d-%m-%Y')

# add datetime column
hourly_actual_generation_data['datetime'] = hourly_actual_generation_data['date'] + pd.to_timedelta(hourly_actual_generation_data['hour'], unit='h')
hourly_actual_generation_data['datetime'] = pd.to_datetime(hourly_actual_generation_data['datetime'])

# ensure proper transition from winter time to summer time
hourly_actual_generation_data['actual_total_generation'] = pd.to_numeric(hourly_actual_generation_data['actual_total_generation'], errors='coerce')
hourly_actual_generation_data.loc[hourly_actual_generation_data['datetime'].isin(missing_datetime_list), 'actual_total_generation'] = pd.NA
hourly_actual_generation_data['actual_total_generation'] = hourly_actual_generation_data['actual_total_generation'].interpolate(method='linear')
hourly_actual_generation_data['actual_solar'] = pd.to_numeric(hourly_actual_generation_data['actual_solar'], errors='coerce')
hourly_actual_generation_data.loc[hourly_actual_generation_data['datetime'].isin(missing_datetime_list), 'actual_solar'] = pd.NA
hourly_actual_generation_data['actual_solar'] = hourly_actual_generation_data['actual_solar'].interpolate(method='linear')
hourly_actual_generation_data['actual_total_wind'] = pd.to_numeric(hourly_actual_generation_data['actual_total_wind'], errors='coerce')
hourly_actual_generation_data.loc[hourly_actual_generation_data['datetime'].isin(missing_datetime_list), 'actual_total_wind'] = pd.NA
hourly_actual_generation_data['actual_total_wind'] = hourly_actual_generation_data['actual_total_wind'].interpolate(method='linear')

# ensure proper transition from summer time to winter time
for dt in duplicated_datetime_list:
    hourly_actual_generation_data.loc[hourly_actual_generation_data['datetime'] == dt, 'actual_total_generation'] /= 2
    hourly_actual_generation_data.loc[hourly_actual_generation_data['datetime'] == dt, 'actual_solar'] /= 2
    hourly_actual_generation_data.loc[hourly_actual_generation_data['datetime'] == dt, 'actual_total_wind'] /= 2

# merge generation dataframes
cols_hourly_generation = ['datetime', 'total_generation', 'solar', 'total_wind']
cols_hourly_actual_generation = ['datetime', 'actual_total_generation', 'actual_solar', 'actual_total_wind']
generation = pd.merge(hourly_generation_data[cols_hourly_generation], hourly_actual_generation_data[cols_hourly_actual_generation], on='datetime', how='outer')

# save total generation dataframe
output_path = os.path.join(processed_data_folder, 'generation.csv')
generation.to_csv(output_path, index=False)


### create solar generation dataframe NED data ###
wind_and_solar_folder = 'Data/raw data/Wind and solar generation'
# create wind and solar dataframe with day ahead forecasts for solar, wind offshore and wind onshore
dataframes = []
for filename in sorted(os.listdir(wind_and_solar_folder)):
    if filename.startswith('zon') and filename.endswith('.csv'):
        file_path = os.path.join(wind_and_solar_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)
solar = pd.concat(dataframes, ignore_index=True)
columns_to_keep = ['validfrom (UTC)', 'volume (kWh)']
solar = solar[columns_to_keep]
# rename columns
solar.columns = ['datetime', 'solar_ned']

# convert to MWh
solar['solar_ned'] = solar['solar_ned'] / 1000

# create datetime, date and hour
solar['datetime'] = pd.to_datetime(solar['datetime'])
solar['date'] = solar['datetime'].dt.strftime('%d-%m-%Y')
solar['hour'] = solar['datetime'].dt.hour
solar = solar.iloc[1:].reset_index(drop=True) # delete the first column

# save solar dataframe
output_path = os.path.join(processed_data_folder, 'solar_NED.csv')
solar.to_csv(output_path, index=False)


### create wind generation dataframe NED data ###
wind_and_solar_folder = 'Data/raw data/Wind and solar generation'
# create wind and solar dataframe with day ahead forecasts for solar, wind offshore and wind onshore
dataframes = []
for filename in sorted(os.listdir(wind_and_solar_folder)):
    if filename.startswith('wind') and filename.endswith('.csv'):
        file_path = os.path.join(wind_and_solar_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)
wind_onshore = pd.concat(dataframes, ignore_index=True)
columns_to_keep = ['validfrom (UTC)', 'volume (kWh)']
wind_onshore = wind_onshore[columns_to_keep]
# rename columns
wind_onshore.columns = ['datetime', 'wind_onshore_ned']

# convert to MWh
wind_onshore['wind_onshore_ned'] = wind_onshore['wind_onshore_ned'] / 1000

# create datetime, date and hour
wind_onshore['datetime'] = pd.to_datetime(wind_onshore['datetime'])
wind_onshore = wind_onshore.iloc[1:].reset_index(drop=True) # delete the first column

# create wind offshore dataframe
dataframes = []
for filename in sorted(os.listdir(wind_and_solar_folder)):
    if filename.startswith('zeewind') and filename.endswith('.csv'):
        file_path = os.path.join(wind_and_solar_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)
wind_offshore = pd.concat(dataframes, ignore_index=True)
columns_to_keep = ['validfrom (UTC)', 'volume (kWh)']
wind_offshore = wind_offshore[columns_to_keep]
# rename columns
wind_offshore.columns = ['datetime', 'wind_offshore_ned']

# convert to MWh
wind_offshore['wind_offshore_ned'] = wind_offshore['wind_offshore_ned'] / 1000

# create datetime, date and hour
wind_offshore['datetime'] = pd.to_datetime(wind_offshore['datetime'])
wind_offshore = wind_offshore.iloc[1:].reset_index(drop=True) # delete the first column

wind = pd.merge(wind_onshore, wind_offshore, on='datetime', how='outer')
wind['total_wind_ned'] = wind['wind_onshore_ned'] + wind['wind_offshore_ned']

# save wind dataframe
output_path = os.path.join(processed_data_folder, 'wind_NED.csv')
wind.to_csv(output_path, index=False)


### create dataframe with all data ###
# change datetime name
API2_coal.rename(columns={'Datetime': 'datetime'}, inplace=True)
ttf_gas.rename(columns={'Datetime': 'datetime'}, inplace=True)
EUA.rename(columns={'Datetime': 'datetime'}, inplace=True)

# choose relevant columns
day_ahead_prices = day_ahead_prices[['datetime', 'price']]
hourly_load_data = hourly_load_data[['datetime', 'load_forecast', 'actual_load']]
generation = generation[['datetime', 'total_generation','solar', 'total_wind', 'actual_total_generation', 'actual_solar', 'actual_total_wind']]
API2_coal = API2_coal[['datetime', 'Price']].rename(columns={'Price': 'API2_coal_price'})
ttf_gas = ttf_gas[['datetime', 'Price']].rename(columns={'Price': 'ttf_gas_price'})
EUA = EUA[['datetime', 'Price']].rename(columns={'Price': 'EUA_price'})
solar = solar[['datetime', 'solar_ned']]
wind = wind[['datetime', 'total_wind_ned']]

# merge dfs
merged_df = day_ahead_prices.merge(hourly_load_data, on='datetime', how='outer')
merged_df = merged_df.merge(generation, on='datetime', how='outer')
merged_df = merged_df.merge(API2_coal, on='datetime', how='outer')
merged_df = merged_df.merge(ttf_gas, on='datetime', how='outer')
merged_df = merged_df.merge(EUA, on='datetime', how='outer')
merged_df = merged_df.merge(solar, on='datetime', how='outer')
merged_df = merged_df.merge(wind, on='datetime', how='outer')

# filter on the correct data
merged_df = merged_df[(merged_df['datetime'] >= '2016-01-01') & (merged_df['datetime'] < '2024-09-01')]

# save merged dataframe
output_path = os.path.join(processed_data_folder, 'data.csv')
merged_df.to_csv(output_path, index=False)


















