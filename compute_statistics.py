"""
Computation of the summary statistics
"""

import pandas as pd
import os

# import the data
processed_data_folder = 'Data/processed data'
data = pd.read_csv(os.path.join(processed_data_folder, "data.csv"))

# set to datetime
data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
data['date'] = data['datetime'].dt.date
data['date'] = pd.to_datetime(data['date'])

# convert load forecast to GWh
#data['load_forecast'] = data['load_forecast'] / 1000

# set date boundaries for the two periods
start_date_1 = pd.to_datetime('2016-01-01')
end_date_1 = pd.to_datetime('2020-12-31')

start_date_2 = pd.to_datetime('2020-01-01')
end_date_2 = pd.to_datetime('2024-08-31')

# filter for first time period (01-01-2016 t/m 31-12-2020)
data_period_1 = data[(data['date'] >= start_date_1) & (data['date'] <= end_date_1)]
data_period_2 = data[(data['date'] >= start_date_2) & (data['date'] <= end_date_2)]

# create dictionairy of dataframes
dataframes = {'period_1': data_period_1, 'period_2': data_period_2}

# select the columns
columns = ['price', 'total_wind', 'solar', 'load_forecast', 'API2_coal_price', 'ttf_gas_price', 'EUA_price']

# create dictionairy to store the statistics
statistics = {}

# calculate summary statistics
for period, df in dataframes.items():
    selected_data = df[columns]
    stats_df = selected_data.agg(['mean', 'std', 'min', 'median', 'max']).T
    stats_df['Q25'] = selected_data.quantile(0.25)
    stats_df['Q75'] = selected_data.quantile(0.75)
    # add statistics dataframe to the dictionairy
    statistics[period] = stats_df

# select the statistics per period
statistics_period_1 = statistics['period_1']
statistics_period_2 = statistics['period_2']
