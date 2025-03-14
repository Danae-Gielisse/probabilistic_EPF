"""
Creates plots of the preprocessed data
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

# import the data
processed_data_folder = 'Data/processed data'
data = pd.read_csv(os.path.join(processed_data_folder, "data.csv"))
# set datetime column to datetime type
data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')

# set colors
color_1 = '#A0522D'
color_2 = '#4682B4'
color_3 = '#8B0000'
color_4 = '#6A5ACD'

### create figure with prices for API2 coal, TTF gas and EUA ###
fig, axs = plt.subplots(3, 1, figsize=(10, 16), sharex=True)

# plot the API2 coal data
axs[0].plot(data['datetime'], data['API2_coal_price'], label='API2 Coal', color=color_1)
axs[0].set_title('API2 coal price over time')
axs[0].set_ylabel('Price [EUR/t]', labelpad=20)
axs[0].grid(True)

# plot the TTF gas data
axs[1].plot(data['datetime'], data['ttf_gas_price'], label='TTF Gas', color=color_2)
axs[1].set_title('TTF gas price over time')
axs[1].set_ylabel('Price [EUR/MWh]', labelpad=20)
axs[1].grid(True)

# plot the EUA data
axs[2].plot(data['datetime'], data['EUA_price'], label='EUA', color=color_3)
axs[2].set_title('EUA price over time')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Price [EUR/tCO2]', labelpad=20)
axs[2].grid(True)

# create plots for the day-ahead prices on winter and summer days in time span 1 and time span 2
dates_2019 = ["2019-01-25", "2019-07-29"]
dates_2024 = ["2024-01-25", "2024-07-29"]
dates = dates_2019 + dates_2024
colors = [color_1, color_2, color_3, color_4]
filtered_data_2019 = data[data["datetime"].dt.date.isin(pd.to_datetime(dates_2019).date)]
filtered_data_2024 = data[data["datetime"].dt.date.isin(pd.to_datetime(dates_2024).date)]
y_min_2019, y_max_2019 = filtered_data_2019["price"].min(), filtered_data_2019["price"].max()
y_min_2024, y_max_2024 = filtered_data_2024["price"].min(), filtered_data_2024["price"].max()
y_min_all = min(y_min_2019, y_min_2024)
y_max_all = max(y_max_2019, y_max_2024)

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8.27, 11.69))  # A4 size in inches (21x29.7 cm)

for ax, date, color in zip(axes, dates, colors):
    df_filtered = data[data["datetime"].dt.date == pd.to_datetime(date).date()]
    df_filtered["hour"] = df_filtered["datetime"].dt.hour
    '''
    if date in dates_2019:
        ax.set_ylim(y_min_2019, y_max_2019)
    else:
        ax.set_ylim(y_min_2024, y_max_2024)
    '''
    ax.set_ylim(y_min_all, y_max_all)
    ax.plot(df_filtered["hour"], df_filtered["price"], linestyle='-', color=color)
    ax.set_title(f"Day-ahead electricity price for the EPEX-NL market on {date}")
    ax.set_xlabel("Hour")
    ax.set_ylabel("day-ahead electricity price", fontsize=9)
    ax.set_xlim(0, 23)
    ax.grid(True)

plt.tight_layout()
plt.savefig('day-ahead price plots winter and summer.png', dpi=300)
plt.show()

# ensure the x-axis labels do not overlap and everything fits well
plt.xticks(rotation=45)
plt.tight_layout()
fig.tight_layout(pad=2.0)
plt.subplots_adjust(top=0.95, bottom=0.1)

# show the graphs
plt.show()

# create a new column 'date' to obtain the date without time component
data['date'] = data['datetime'].dt.date

# create df's with daily data
daily_data_df = data.groupby('date').agg({'price': 'mean',
                                            'total_generation': 'sum',
                                            'actual_total_generation': 'sum',
                                            'actual_total_wind': 'sum',
                                            'total_wind': 'sum',
                                            'actual_solar': 'sum',
                                            'solar': 'sum',
                                            'load_forecast': 'sum',
                                            'solar_ned': 'sum',
                                            'total_wind_ned': 'sum',
                                            'actual_load': 'sum'}).reset_index()

# convert to GWh
daily_data_df['solar'] = daily_data_df['solar'] / 1000
daily_data_df['solar_ned'] = daily_data_df['solar_ned'] / 1000
daily_data_df['total_wind'] = daily_data_df['total_wind'] / 1000
daily_data_df['load_forecast'] = daily_data_df['load_forecast'] / 1000
daily_data_df['actual_load'] = daily_data_df['actual_load'] / 1000
daily_data_df['total_wind_ned'] = daily_data_df['total_wind_ned'] / 1000


#### create the figure with the day-ahead electricity prices, generation forecast and load forecast ###
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# plot daily prices
axs[0].plot(daily_data_df['date'], daily_data_df['price'], label='Price', color=color_1)
axs[0].set_title('Daily average day-ahead electricity prices EPEX-NL over time')
axs[0].set_ylabel('Daily average day-ahead electricity price [EUR/MWh]', fontsize=6, labelpad=20)
axs[0].grid(True)

# plot daily wind generation
axs[1].plot(daily_data_df['date'], daily_data_df['total_wind'], label='Generation Forecast Wind', color=color_2)
#axs[1].plot(daily_data_df['date'], daily_data_df['total_wind_ned'], label='Generation Forecast Wind', color=color_3)
axs[1].set_title('Daily wind generation forecast over time')
axs[1].set_ylabel('Daily wind generation forecast [GWh]', fontsize=6, labelpad=20)
axs[1].grid(True)

# plot daily solar generation
axs[2].plot(daily_data_df['date'], daily_data_df['solar'], label='Generation Forecast Solar', color=color_3)
#axs[2].plot(daily_data_df['date'], daily_data_df['solar_ned'], label='Generation  Solar', color=color_4)
axs[2].set_title('Daily solar generation forecast over time')
axs[2].set_ylabel('Daily solar generation forecast [GWh]', fontsize=6, labelpad=20)
axs[2].grid(True)

# plot daily load forecast
axs[3].plot(daily_data_df['date'], daily_data_df['load_forecast'], label='Load Forecast', color=color_4)
axs[3].set_title('Daily load forecast over time')
axs[3].set_xlabel('Date')
axs[3].set_ylabel('Daily load forecast [GWh]', fontsize=8, labelpad=20)
axs[3].grid(True)

# ensure the x-axis labels do not overlap and everything fits well
plt.xticks(rotation=45)
plt.tight_layout()
fig.tight_layout(pad=2.0)
plt.subplots_adjust(top=0.95, bottom=0.1)

# show the graph
plt.show()

daily_data_df['date'] = pd.to_datetime(daily_data_df['date'])

# set date boundaries for the two periods
start_date_1 = pd.to_datetime('2016-01-01')
end_date_1 = pd.to_datetime('2020-12-31')
start_date_2 = pd.to_datetime('2020-01-01')
end_date_2 = pd.to_datetime('2024-08-31')

# filter for first time period (01-01-2016 t/m 31-12-2020)
daily_data_df_period_1 = daily_data_df[(daily_data_df['date'] >= start_date_1) & (daily_data_df['date'] <= end_date_1)]
# filter for second time period (01-01-2020 t/m 31-08-2024)
daily_data_df_period_2 = daily_data_df[(daily_data_df['date'] >= start_date_2) & (daily_data_df['date'] <= end_date_2)]


### create plot for the first time period ###
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# plot daily prices for the first time peeriod
max_price = daily_data_df_period_1['price'].max()
y_margin = max_price * 0.35
axs[0].plot(daily_data_df_period_1['date'], daily_data_df_period_1['price'], label='Price', color=color_1)
axs[0].set_title('Daily Average Day-Ahead Electricity Prices EPEX-NL Over Time')
axs[0].set_ylim(0, max_price + y_margin)
axs[0].set_ylabel('Daily average day-ahead electricity price [EUR/MWh]', fontsize=6, labelpad=20)
axs[0].grid(True)

# plot daily wind generation of the first time period
axs[1].plot(daily_data_df_period_1['date'], daily_data_df_period_1['total_wind'], label='Total wind', color=color_2)
axs[1].set_title('Daily wind generation forecast over time')
axs[1].set_ylabel('Daily wind generation forecast [GWh]', fontsize=6, labelpad=20)
axs[1].grid(True)

# plot daily solar generation of the first time period
axs[2].plot(daily_data_df_period_1['date'], daily_data_df_period_1['solar'], label='Total solar', color=color_3)
axs[2].set_title('Daily solar generation forecast over time')
axs[2].set_ylabel('Daily solar generation forecast [GWh]', fontsize=6, labelpad=20)
axs[2].grid(True)

# plot daily load forecast of the first time period
axs[3].plot(daily_data_df_period_1['date'], daily_data_df_period_1['load_forecast'], label='Total Load Forecast', color=color_4)
axs[3].set_title('Daily load forecast over time')
axs[3].set_xlabel('Date')
axs[3].set_ylabel('Daily load forecast [GWh]', fontsize=8, labelpad=20)
axs[3].grid(True)

# ensure the x-axis labels do not overlap
plt.xticks(rotation=45)
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.1)
plt.show()


### create plot for the second time period ###
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# plot the daily price of the second period
max_price = daily_data_df_period_2['price'].max()
y_margin = max_price * 0.35
axs[0].plot(daily_data_df_period_2['date'], daily_data_df_period_2['price'], label='Price', color=color_1)
axs[0].set_title('Daily Average Day-Ahead Electricity Prices EPEX-NL Over Time')
axs[0].set_ylim(0, max_price + y_margin)
axs[0].set_ylabel('Daily average day-ahead electricity price [EUR/MWh]', fontsize=6, labelpad=20)
axs[0].grid(True)

# plot daily wind generation of the second time period
axs[1].plot(daily_data_df_period_2['date'], daily_data_df_period_2['total_wind'], label='Total wind', color=color_2)
axs[1].set_title('Daily wind generation forecast over time')
axs[1].set_ylabel('Daily wind generation forecast [GWh]', fontsize=6, labelpad=20)
axs[1].grid(True)

# plot daily solar generation of the second time period
axs[2].plot(daily_data_df_period_2['date'], daily_data_df_period_2['solar'], label='Total solar', color=color_3)
axs[2].set_title('Daily solar generation forecast over time')
axs[2].set_ylabel('Daily solar generation forecast [GWh]', fontsize=6, labelpad=20)
axs[2].grid(True)

# plot daily load forecasting of the second time period
axs[3].plot(daily_data_df_period_2['date'], daily_data_df_period_2['load_forecast'], label='Total Load Forecast', color=color_4)
axs[3].set_title('Daily load forecast over time')
axs[3].set_xlabel('Date')
axs[3].set_ylabel('Daily load forecast [GWh]', fontsize=8, labelpad=20)
axs[3].grid(True)

# ensure the x-axis labels do not overlap
plt.xticks(rotation=45)
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.1)
plt.show()
