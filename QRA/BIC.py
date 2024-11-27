"""
Calculates the minimum BIC value per day, hour and tau and stores the results in a dataframe. Also creates the
forecast dataframe for the BIC.
"""

import numpy as np
import pandas as pd

LAMBDA = np.concatenate(([0], np.logspace(-1, 3, 19)))
TAU = np.arange(1, 100) / 100
calib = 364
time_span = 1 # Choose the time span

# folders where the csv files are located
folder_BIC_df = f'../Results/probabilistic_forecasts_time_span_{time_span}/BIC_df.csv'
folder_forecast_BIC = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_BIC.csv'

# get number of days in test period
days_mapping = {1: 735, 2: 613}
number_of_days = days_mapping.get(time_span)

def preload_forecasts(lambda_values, time_span):
    forecast_data = {}
    beta_data = {}
    for l in lambda_values:
        forecast_path = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_lambda_{l}.csv'
        beta_path = f'../Results/probabilistic_forecasts_time_span_{time_span}/beta_lambda_{l}.csv'
        forecast_data[l] = pd.read_csv(forecast_path)
        beta_data[l] = pd.read_csv(beta_path)
    return forecast_data, beta_data

def create_df_BIC(lambda_values, tau_values, total_days, calib, time_span):
    forecast_data, beta_data = preload_forecasts(lambda_values, time_span)
    results = []

    for day in range(total_days):
        print(f'Starting day {day}')
        for hour in range(24):
            print(f'Starting hour {hour}')
            for q in tau_values:
                min_BIC, min_lambda = calculate_min_BIC(lambda_values, q, day, hour, calib, time_span, forecast_data, beta_data)
                results.append({
                    'day': day,
                    'hour': hour,
                    'tau': q,
                    'min_lambda': min_lambda,
                    'min_BIC': min_BIC
                })

    return pd.DataFrame(results)

def calculate_min_BIC(lambda_values, tau, day, hour, calib, time_span, forecast_data, beta_data):
    min_BIC = np.inf
    min_lambda = None
    for l in lambda_values:
        BIC_lambda = BIC(tau, day, hour, l, calib, forecast_data, beta_data)
        if BIC_lambda < min_BIC:
            min_BIC = BIC_lambda
            min_lambda = l

    return min_BIC, min_lambda

def BIC(tau, day, hour, lambda_value, calib, forecast_data, beta_data):
    forecast = forecast_data[lambda_value]
    beta_df = beta_data[lambda_value]

    percentile = int(round(tau * 100))
    column_name = f'Percentile_{percentile}'
    price_real = forecast['Price']
    predicted_price_tau = forecast[column_name]

    beta_index = day * 2376 + 99 * hour + percentile - 1
    beta = beta_df.loc[beta_index, 'beta']
    m = np.count_nonzero(beta)
    p = len(beta)

    d_star_range = range(max(0, day - calib), day + 1)
    price_real_dh = price_real[24 * np.array(d_star_range) + hour].values
    predicted_price_tdh = predicted_price_tau[24 * np.array(d_star_range) + hour].values
    indicators = (price_real_dh < predicted_price_tdh).astype(int)

    sum_term = np.sum((tau - indicators) * (price_real_dh - predicted_price_tdh))
    first_term = np.log(sum_term) if sum_term != 0 else np.inf
    second_term = m * (np.log(calib) / (2 * calib)) * np.log(p)

    return first_term + second_term

# create dataframe BIC
df_BIC = create_df_BIC(LAMBDA, TAU, number_of_days, calib, time_span)
# save dataframe BIC
df_BIC.to_csv(folder_BIC_df, index=False)

### create forecast BIC dataframe ###
df_BIC = pd.read_csv(folder_BIC_df) # for case run seperate from the code above

# create dictionairy for lambda
lambda_dict = {lmbda: i for i, lmbda in enumerate(LAMBDA)}

# create list of al forecast dataframes
forecast_list = []
for i in range(0, len(LAMBDA)):
    probabilistic_forecast_folder = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_lambda_' + str(LAMBDA[i]) + '.csv'
    forecast = pd.read_csv(probabilistic_forecast_folder)
    forecast_list.append(forecast)

# build forecast_df
forecast_df = forecast_list[0].iloc[:, :2]
percentiles = ['Percentile_' + str(i) for i in range(1, 100)]
forecast_df[percentiles] = None

# fill all rows of the forecast_df with the correct values
for index, row in df_BIC.iterrows():
    if index % 1000 == 0:
        print(index)
    day = row['day']
    hour = row['hour']
    tau = row['tau']
    min_lambda = row['min_lambda']

    # find the index of the forecast dataframe that corresponds to the min_lambda
    if min_lambda in lambda_dict:
        forecast_index = lambda_dict[min_lambda]
    else:
        # if min_lambda is not found, use the closest match
        closest_lambda = min(LAMBDA, key=lambda x: abs(x - min_lambda))
        forecast_index = lambda_dict[closest_lambda]

    # get the value from the correct forecast dataframe
    percentile_column = f'Percentile_{int(round(tau * 100))}'
    forecast_df.loc[24 * day + hour, percentile_column] = forecast_list[forecast_index].loc[24 * day + hour, percentile_column]


# sort the percentiles in ascending order
percentile_columns = [f'Percentile_{i}' for i in range(1, 100)]
sorted_percentiles = np.sort(forecast_df[percentile_columns].values)
forecast_df[percentile_columns] = sorted_percentiles

# save forecast_df
forecast_df.to_csv(folder_forecast_BIC)
