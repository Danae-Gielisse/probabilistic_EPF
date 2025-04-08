"""
Creates point forecasts of the naive benchmark
"""

import pandas as pd

# choose time span
time_span = 1

# import the data
data = pd.read_csv('../Data/processed data/data.csv')

# add weekday column to the data
data['datetime'] = pd.to_datetime(data['datetime'])
data['weekday'] = data['datetime'].dt.weekday + 1

# select test data
if time_span == 1:
    data_df = data.loc[26040:43847, ['datetime', 'price']]
    forecast = data.loc[26208:43847, ['datetime', 'price', 'weekday']]
else:
    data_df = data.loc[61104:75985, ['datetime', 'price']]
    forecast = data.loc[61272:75985, ['datetime', 'price', 'weekday']]

# create list of how much the price must be shifted
shift_24 = [2, 3, 4, 5]
shift_168 = [1, 6, 7]

# make forecast with naive model
forecast.loc[forecast['weekday'].isin(shift_24), 'forecasted_price'] = \
    data_df['price'].shift(24).reindex(forecast.index)
forecast.loc[forecast['weekday'].isin(shift_168), 'forecasted_price'] = \
    data_df['price'].shift(24 * 7).reindex(forecast.index)


# reset index, drop weekday column and change name of price column
forecast = forecast.reset_index(drop=True)
forecast = forecast.drop(columns=['weekday'])
forecast = forecast.rename(columns={'price': 'price_real'})

# save forecast
forecast.to_csv(f'../Results/point_forecasting_time_span_{time_span}/naive_benchmark.csv')
