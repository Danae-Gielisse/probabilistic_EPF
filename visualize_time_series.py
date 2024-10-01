import pandas as pd
import os
import matplotlib.pyplot as plt

# import the data
processed_data_folder = 'Data/processed data'
prices_df = pd.read_csv(os.path.join(processed_data_folder, "prices.csv"))
generation_df = pd.read_csv(os.path.join(processed_data_folder, "generation.csv"))
load_df = pd.read_csv(os.path.join(processed_data_folder, "load.csv"))
api2_coal = pd.read_csv(os.path.join(processed_data_folder, "API2_coal.csv"))
brent_oil = pd.read_csv(os.path.join(processed_data_folder, "brent_oil.csv"))
EUA = pd.read_csv(os.path.join(processed_data_folder, "EUA.csv"))
ttf_gas = pd.read_csv(os.path.join(processed_data_folder, "ttf_gas.csv"))

# set datetime columns to datetime type
prices_df['datetime'] = pd.to_datetime(prices_df['datetime'], errors='coerce')
generation_df['datetime'] = pd.to_datetime(generation_df['datetime'], errors='coerce')
load_df['datetime'] = pd.to_datetime(load_df['datetime'], errors='coerce')
api2_coal['Datetime'] = pd.to_datetime(api2_coal['Datetime'], errors='coerce')
brent_oil['Datetime'] = pd.to_datetime(brent_oil['Datetime'], errors='coerce')
EUA['Datetime'] = pd.to_datetime(EUA['Datetime'], errors='coerce')
ttf_gas['Datetime'] = pd.to_datetime(ttf_gas['Datetime'], errors='coerce')

# set colors
color_1 = '#A0522D'
color_2 = '#4682B4'
color_3 = '#8B0000'
color_4 = '#6A5ACD'

# create figure with prices for API2 coal, TTF gas, Brent oil en EUA
fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

# plot the API2 coal data
axs[0].plot(api2_coal['Datetime'], api2_coal['Price'], label='API2 Coal', color=color_1)
axs[0].set_title('API2 coal price over time')
axs[0].set_ylabel('Price [EUR/t]', labelpad=20)
axs[0].grid(True)
axs[0].legend()

# plot the Brent oil data
axs[1].plot(brent_oil['Datetime'], brent_oil['Price'], label='Brent Oil', color=color_2)
axs[1].set_title('Brent oil price over time')
axs[1].set_ylabel('Price [EUR/bbL]', labelpad=20)
axs[1].grid(True)
axs[1].legend()

# plot the TTF gas data
axs[2].plot(ttf_gas['Datetime'], ttf_gas['Price'], label='TTF Gas', color=color_3)
axs[2].set_title('TTF gas price over time')
axs[2].set_ylabel('Price [EUR/MWh]', labelpad=20)
axs[2].grid(True)
axs[2].legend()

# plot the EUA data
axs[3].plot(EUA['Datetime'], EUA['Price'], label='EUA', color=color_4)
axs[3].set_title('EUA price over time')
axs[3].set_xlabel('Date')
axs[3].set_ylabel('Price [EUR/tCO2]', labelpad=20)
axs[3].grid(True)
axs[3].legend()

# ensure the x-axis labels do not overlap and everything fits well
plt.xticks(rotation=45)
plt.tight_layout()
fig.tight_layout(pad=2.0)
plt.subplots_adjust(top=0.95, bottom=0.1)

# show the graphs
plt.show()

# create a new column 'date' to obtain the date without time component
prices_df['date'] = prices_df['datetime'].dt.date
generation_df['date'] = generation_df['datetime'].dt.date
load_df['date'] = load_df['datetime'].dt.date

# create df's with daily data
daily_prices_df = prices_df.groupby('date').agg({'price': 'sum'}).reset_index()
daily_generation_df = generation_df.groupby('date').agg({'total_generation': 'sum'}).reset_index()
daily_load_df = load_df.groupby('date').agg({'load_forecast': 'sum'}).reset_index()

# rename the columns
daily_prices_df.rename(columns={'price': 'daily_price'}, inplace=True)
daily_generation_df.rename(columns={'total_generation': 'daily_generation'}, inplace=True)
daily_load_df.rename(columns={'load_forecast': 'daily_load_forecast'}, inplace=True)

# converting from MWh to GWh
daily_prices_df['daily_price_gwh'] = daily_prices_df['daily_price'] / 1000
daily_generation_df['daily_generation_gwh'] = daily_generation_df['daily_generation'] / 1000
daily_load_df['daily_load_forecast_gwh'] = daily_load_df['daily_load_forecast'] / 1000

# create the figure with the day-ahead electricity prices, generation forecast and load forecast
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# plot de dagelijkse prijzen
axs[0].plot(daily_prices_df['date'], daily_prices_df['daily_price_gwh'], label='Total Price', color=color_1)
axs[0].set_title('Day-ahead electricity prices EPEX-NL over time')
axs[0].set_ylabel('Day-ahead electricity price [EUR/GWh]', fontsize=8, labelpad=20)
axs[0].grid(True)
axs[0].legend()

# plot de dagelijkse generatie
axs[1].plot(daily_generation_df['date'], daily_generation_df['daily_generation_gwh'], label='Total Generation', color=color_2)
axs[1].set_title('Day-ahead wind and solar generation forecast over time')
axs[1].set_ylabel('Day-ahead wind and solar generation forecast [GW]', fontsize=8, labelpad=20)
axs[1].grid(True)
axs[1].legend()

# plot de dagelijkse load forecast
axs[2].plot(daily_load_df['date'], daily_load_df['daily_load_forecast_gwh'], label='Total Load Forecast', color=color_3)
axs[2].set_title('Day-ahead load forecast over time')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Day-ahead load forecast [GW]', fontsize=8, labelpad=20)
axs[2].grid(True)
axs[2].legend()

# ensure the x-axis labels do not overlap and everything fits well
plt.xticks(rotation=45)
plt.tight_layout()
fig.tight_layout(pad=2.0)
plt.subplots_adjust(top=0.95, bottom=0.1)

# show the graph
plt.show()