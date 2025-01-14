import pandas as pd

time_span = 2
data = pd.read_csv(f'../Data/processed data/data.csv')
if time_span == 1:
    subset = data.loc[29376:43847, ['datetime', 'price']]
else:
    subset = data.loc[62472:75985, ['datetime', 'price']]


def aggregate_horizontal(df_list_per):
    percentile_columns = [df.filter(regex='Percentile_') for df in df_list_per]
    average_df = pd.concat(percentile_columns, axis=1).T.groupby(level=0).mean().T

    # Add datetime en price columns
    average_df['Datetime'] = df_list_per[0]['Datetime']
    average_df['Price'] = df_list_per[0]['Price']

    average_df = average_df[
        ['Datetime', 'Price'] + [col for col in average_df.columns if col.startswith('Percentile_')]]

    return average_df


# create list of dataframes with percentiles forecasts
forecast_list_lasso = []
forecast_list_enet = []
forecast_list_lasso_weighted = []
# add jsu runs to forecast list
for run in range(1, 5):
    folder_percentiles = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_jsu{run}_lasso.csv'
    df_percentile = pd.read_csv(folder_percentiles, index_col=0)
    forecast_list_lasso.append(df_percentile)
    forecast_list_enet.append(df_percentile)
    forecast_list_lasso_weighted.append(df_percentile)
# add LQRA(BIC) forecast to forecast list
folder_lasso = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_BIC.csv'
if time_span == 1:
    index_col = [0, 1]
    date = '2019-05-09'
else:
    index_col = 0
    date = '2023-02-16'

df_LQRA_BIC = pd.read_csv(folder_lasso, index_col=index_col)
df_LQRA_BIC = df_LQRA_BIC[df_LQRA_BIC['Date'] >= date]
df_LQRA_BIC = df_LQRA_BIC.reset_index(drop=True)
forecast_list_lasso.append(df_LQRA_BIC)
for i in range(0, 4):
    forecast_list_lasso_weighted.append(df_LQRA_BIC)

# add EQRA(BIC-0.25), EQRA(BIC-0.5) and EQRA(BIC-0.75)
alpha_list = [0.25, 0.5, 0.75]
for alpha in alpha_list:
    folder_enet = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_BIC_alpha_{alpha}.csv'
    df_EQRA_BIC = pd.read_csv(folder_enet, index_col=0)
    df_EQRA_BIC = df_EQRA_BIC[df_EQRA_BIC['Date'] >= date]
    df_EQRA_BIC = df_EQRA_BIC.reset_index(drop=True)
    forecast_list_enet.append(df_EQRA_BIC)

# perform horizontal aggregation
stat_nn_ens_forecast_lasso = aggregate_horizontal(forecast_list_lasso)
stat_nn_ens_forecast_enet = aggregate_horizontal(forecast_list_enet)
stat_nn_ens_forecast_weighted = aggregate_horizontal(forecast_list_lasso_weighted)
output_folder_lasso = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_stat_nn_ens_lasso.csv'
output_folder_enet = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_stat_nn_ens_enet.csv'
output_folder_lasso_weighted = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_stat_nn_ens_lasso_weighted.csv'
stat_nn_ens_forecast_lasso.to_csv(output_folder_lasso)
stat_nn_ens_forecast_enet.to_csv(output_folder_enet)
stat_nn_ens_forecast_weighted.to_csv(output_folder_lasso_weighted)

