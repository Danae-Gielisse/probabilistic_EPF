"""
Performs the horizontal and vertical aggregration of the four different runs
"""

import numpy as np
from scipy.stats import johnsonsu
import pandas as pd
from scipy.optimize import minimize

# choose time span and regularization
time_span = 2
regularization = 'enet'
# define distribution
distribution = 'jsu'

# import the testdata
data = pd.read_csv(f'../Data/processed data/data.csv')
if time_span == 1:
    subset = data.loc[29376:43847, ['datetime', 'price']]
else:
    subset = data.loc[62472:75985, ['datetime', 'price']]


def fit_johnsonsu(x_range, pdf_values, df1_row, df2_row, df3_row, df4_row, bound):
    initial_skewness = np.median([df1_row['skewness'], df2_row['skewness'],
                                df3_row['skewness'], df4_row['skewness']])
    initial_tailweight = np.median([df1_row['tailweight'], df2_row['tailweight'],
                                  df3_row['tailweight'], df4_row['tailweight']])
    initial_loc = np.median([df1_row['loc'], df2_row['loc'],
                           df3_row['loc'], df4_row['loc']])
    initial_scale = np.median([df1_row['scale'], df2_row['scale'],
                             df3_row['scale'], df4_row['scale']])


    def objective(params):
        a, b, loc, scale = params
        try:
            fitted_pdf = johnsonsu.pdf(x_range, a, b, loc, scale)
            return np.sum((fitted_pdf - pdf_values) ** 2)
        except:
            return np.inf

    result = minimize(objective,
                      x0=[initial_skewness, initial_tailweight, initial_loc, initial_scale],
                      method='L-BFGS-B', bounds=bound)

    return result.x


def create_johnson_distribution(row):
    return johnsonsu(
        a=row['skewness'],
        b=row['tailweight'],
        loc=row['loc'],
        scale=row['scale']
    )


def aggregate_vertical(df1, df2, df3, df4, bound, num_points):
    results = []
    min_price = min([df1['price'].min(), df2['price'].min(), df3['price'].min(), df4['price'].min()])
    max_price = max([df1['price'].max(), df2['price'].max(), df3['price'].max(), df4['price'].max()])
    margin = (max_price - min_price) * 0.1  # 10% marge
    x_range = np.linspace(min_price - margin, max_price + margin, num_points)

    # Voor elke timestamp
    for idx in range(len(df1)):
        if idx % 24 == 0:
            print('starting day ' + str(idx / 24))
        dist1 = create_johnson_distribution(df1.iloc[idx])
        dist2 = create_johnson_distribution(df2.iloc[idx])
        dist3 = create_johnson_distribution(df3.iloc[idx])
        dist4 = create_johnson_distribution(df4.iloc[idx])

        avg_pdf = (dist1.pdf(x_range) + dist2.pdf(x_range) +
                   dist3.pdf(x_range) + dist4.pdf(x_range)) / 4

        skewness, tailweight, location, scale = fit_johnsonsu(
            x_range, avg_pdf,
            df1.iloc[idx], df2.iloc[idx], df3.iloc[idx], df4.iloc[idx], bound
        )

        results.append({
            'day': df1.iloc[idx]['day'],
            'hour': df1.iloc[idx]['hour'],
            'loc': location,
            'scale': scale,
            'tailweight': tailweight,
            'skewness': skewness
        })

    return pd.DataFrame(results)


def aggregate_horizontal(df_list_per):
    percentile_columns = [df.filter(regex='Percentile_') for df in df_list_per]
    average_df = pd.concat(percentile_columns, axis=1).T.groupby(level=0).mean().T

    # Voeg 'Datetime' en 'Price' kolommen toe aan het gemiddelde dataframe
    average_df['Datetime'] = df_list_per[0]['Datetime']
    average_df['Price'] = df_list_per[0]['Price']

    # Herordenen van kolommen
    average_df = average_df[
        ['Datetime', 'Price'] + [col for col in average_df.columns if col.startswith('Percentile_')]]

    return average_df


# create list of the 4 parameter dataframes and 4 percentiles dataframes
df_list = []
df_list_percentiles = []
for run in range(1, 5):
    folder_parameter = f'../Results/distparams_NN_time_span_{time_span}/parameter_df_{distribution}{run}_{regularization}.csv'
    folder_percentiles = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_{distribution}{run}_{regularization}.csv'
    df_parameter = pd.read_csv(folder_parameter)
    df_percentile = pd.read_csv(folder_percentiles, index_col=0)
    df_parameter = pd.concat([subset.reset_index(drop=True), df_parameter.reset_index(drop=True)], axis=1)
    df_list.append(df_parameter)
    df_list_percentiles.append(df_percentile)

combined_df = pd.concat([df_list[0], df_list[1], df_list[2], df_list[3]])

# calculate minima and maxima for all parameters
min_skewness, max_skewness = combined_df['skewness'].min(), combined_df['skewness'].max()
min_tailweight, max_tailweight = combined_df['tailweight'].min(), combined_df['tailweight'].max()
min_loc, max_loc = combined_df['loc'].min(), combined_df['loc'].max()
min_scale, max_scale = combined_df['scale'].min(), combined_df['scale'].max()

# define bounds
bounds = [
    (min_skewness, max_skewness),       # for skewness (a)
    (min_tailweight, max_tailweight),   # for tailweight (b)
    (min_loc, max_loc),                 # for loc
    (min_scale, max_scale)              # for scale
]

# perform the vertical aggregation and save the ensemble
df_vertical_aggregated = aggregate_vertical(df_list[0], df_list[1], df_list[2], df_list[3], bounds, 50000)
output_folder_param_ens = f'../Results/distparams_NN_time_span_{time_span}/parameter_df_{distribution}pEns_{regularization}.csv'
df_vertical_aggregated.to_csv(output_folder_param_ens)

# perform the horizontal aggregation and save the ensemble
df_horizontal_aggregated = aggregate_horizontal(df_list_percentiles)
output_folder_percentiles_ens = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_{distribution}qEns_{regularization}.csv'
df_horizontal_aggregated.to_csv(output_folder_percentiles_ens)
