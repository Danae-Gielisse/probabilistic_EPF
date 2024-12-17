import pandas as pd
import numpy as np
from scipy.stats import johnsonsu

# choose run, time span and regularization
time_span = 2
run = 'pEns' # choose 1, 2, 3, 4 or 'pEns'
regularization = 'enet' # choose for lasso or enet
distribution = 'jsu'

# define input and output folder
distribution_df_folder = f'../Results/distparams_NN_time_span_{time_span}/parameter_df_{distribution}{run}_{regularization}.csv'
output_folder = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_{distribution}{run}_{regularization}.csv'

# import the data
df_distribution = pd.read_csv(distribution_df_folder)
data = pd.read_csv('../Data/processed data/data.csv')

if time_span == 1:
    subset = data.loc[29376:43848, ['datetime', 'price']]
else:
    subset = data.loc[62472: , ['datetime', 'price']]
df_distribution = pd.concat([subset.reset_index(drop=True), df_distribution.reset_index(drop=True)], axis=1)

percentiles = np.arange(1, 100)
result_list = []

for (day, hour), group in df_distribution.groupby(['day', 'hour']):
    loc = group['loc'].values[0]
    scale = group['scale'].values[0]
    tailweight = group['tailweight'].values[0]
    skew = group['skewness'].values[0]

    dist = johnsonsu(skew, tailweight, loc=loc, scale=scale)
    percentiles_values = dist.ppf(percentiles / 100)
    datetime = group['datetime'].values[0]
    price = group['price'].values[0]
    result_list.append([datetime, price] + list(percentiles_values))

columns = ['Datetime', 'Price'] + [f'Percentile_{p}' for p in percentiles]
df_percentile = pd.DataFrame(result_list, columns=columns)

# save the dataframe
df_percentile.to_csv(output_folder)
