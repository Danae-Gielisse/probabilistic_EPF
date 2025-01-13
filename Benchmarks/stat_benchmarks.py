import pandas as pd
from sklearn.linear_model import QuantileRegressor

# choose time span
time_span = 1

WIN = range(56, 729, 28)


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


def perform_qr(point_forecast_df):
    result_df = point_forecast_df[['datetime', 'price']].copy()
    percentiles = list(range(1, 100))

    for percentile in percentiles:
        print('Percentile ' + str(percentile) + ' started')
        qr = QuantileRegressor(quantile=percentile / 100, alpha=0)

        actual_prices = point_forecast_df['price'].values
        forecasted_prices = point_forecast_df['forecast'].values.reshape(-1, 1)

        qr.fit(forecasted_prices, actual_prices)
        predictions = qr.predict(forecasted_prices)

        result_df[f'Percentile_{percentile}'] = predictions

    return result_df


def aggregate_horizontal(df_list_per):
    percentile_columns = [df.filter(regex='Percentile_') for df in df_list_per]
    average_df = pd.concat(percentile_columns, axis=1).T.groupby(level=0).mean().T

    # Add datetime en price columns
    average_df['Datetime'] = df_list_per[0]['Datetime']
    average_df['Price'] = df_list_per[0]['Price']

    average_df = average_df[
        ['Datetime', 'Price'] + [col for col in average_df.columns if col.startswith('Percentile_')]]

    return average_df


# Perform Stat-q_ens
forecast_list = []
qr_list = []
for win in WIN:
    data = pd.read_csv(f'../Results/point_forecasting_time_span_{time_span}/window_{win}.csv')
    if time_span == 1:
        data = data[data['datetime'] >= '2018-12-28']
    else:
        data = data[data['datetime'] >= '2022-12-28']
    data.columns = ['datetime', 'price', 'forecast']
    forecast_list.append(data)

    qr_win = perform_qr(data)
    qr_win.columns = ['Datetime', 'Price'] + [f'Percentile_{i}' for i in range(1, 100)]
    qr_list.append(qr_win)
    qr_win.to_csv(f'../Results/probabilistic_forecasts_time_span_{time_span}/qr_win{win}.csv')

q_ens = aggregate_horizontal(qr_list)
q_ens.to_csv(f'../Results/probabilistic_forecasts_time_span_{time_span}/q_Ens_forecast.csv')






