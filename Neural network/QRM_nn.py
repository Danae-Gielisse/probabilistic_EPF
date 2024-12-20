import pandas as pd
from sklearn.linear_model import QuantileRegressor

# choose time span
time_span = 1


def perform_qrm(df_list):
    # combine all forecasts for the same datetime
    combined_df = pd.DataFrame()
    for df in df_list:
        if combined_df.empty:
            combined_df = df[['datetime', 'price']].copy()
        temp_df = df[['datetime', 'forecast']].copy()
        temp_df = temp_df.rename(columns={'forecast': f'forecast_{len(combined_df.columns) - 2}'})
        combined_df = combined_df.merge(temp_df, on='datetime', how='outer')

    # calculate the mean for all forecasts
    forecast_columns = [col for col in combined_df.columns if 'forecast' in col]
    combined_df['mean_forecast'] = combined_df[forecast_columns].mean(axis=1)

    # calculate quantiles for each datetime
    result_df = combined_df[['datetime', 'price']].copy()
    percentiles = list(range(1, 100))

    for percentile in percentiles:
        print('Percentile ' + str(percentile) + ' started')
        qr = QuantileRegressor(quantile=percentile / 100, alpha=0)

        mean_forecasts = combined_df['mean_forecast'].values.reshape(-1, 1)
        actual_prices = combined_df['price'].values

        qr.fit(mean_forecasts, actual_prices)
        predictions = qr.predict(mean_forecasts)

        result_df[f'Percentile_{percentile}'] = predictions

    return result_df


# read the data
data = pd.read_csv('../Data/processed data/data.csv')
if time_span == 1:
    subset = data.loc[29376:43847, ['datetime']]
else:
    subset = data.loc[62472:75985, ['datetime']]

run_list = [1, 2, 3, 4]
df_list = []
for run in run_list:
    df = pd.read_csv(f"../Results/point_forecasts_NN_time_span_{time_span}/point{run}_lasso.csv")
    df = pd.concat([subset.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    df.columns = ['datetime', 'price', 'forecast']
    df_list.append(df)

forecast = perform_qrm(df_list)

# change column names
forecast.columns = ['Datetime', 'Price'] + [f'Percentile_{i}' for i in range(1, 100)]

# save result
forecast.to_csv(f'../Results/percentiles_NN_time_span_{time_span}/percentiles_point_qrm.csv')

