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

# Definieer het pad waar het gecombineerde bestand moet worden opgeslagen in de Data map
output_path = os.path.join(processed_data_folder, 'prices.csv')

# Sla het gecombineerde dataframe op als een nieuwe CSV in de Data map
day_ahead_prices.to_csv(output_path, index=False)



