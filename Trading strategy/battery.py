"""
This code creates the profits dataframes of the trading strategies
"""

import pandas as pd
import numpy as np

# choose time span
time_span = 1

# create lists of methodnames
method_names = ['lambda_0.0', 'q_Ens', 'qrm', 'point_qra', 'point_qrm', 'BIC', 'BIC_alpha_0.25',
                'BIC_alpha_0.5', 'BIC_alpha_0.75', 'jsu1_lasso', 'jsu2_lasso', 'jsu3_lasso', 'jsu4_lasso',
                'jsupEns_lasso', 'jsuqEns_lasso', 'jsu1_enet', 'jsu2_enet', 'jsu3_enet', 'jsu4_enet', 'jsupEns_enet',
                'jsuqEns_enet', 'stat_nn_ens_lasso', 'stat_nn_ens_enet', 'stat_nn_ens_lasso_weighted']

methods = ['QRA', 'Stat-qEns', 'Stat-QRM', 'DNN-QRA', 'DNN-QRM', 'LQRA(BIC)', 'EQRA(BIC-0.25)', 'EQRA(BIC-0.5)',
           'EQRA(BIC-0.75)', 'DDNN-L-1', 'DDNN-L-2', 'DDNN-L-3', 'DDNN-L-4', 'DDNN-L-pEns', 'DDNN-L-qEns',
           'DDNN-E-1', 'DDNN-E-2', 'DDNN-E-3', 'DDNN-E-4', 'DDNN-E-pEns', 'DDNN-E-qEns', 'DDNN-LQRA(BIC)-qEns',
           'DDNN-EQRA(BIC)-qEns', 'DDNN-LQRA(BIC)-wqEns']

stat_methods = ['lambda_0.0', 'qrm', 'BIC', 'BIC_alpha_0.25', 'BIC_alpha_0.5', 'BIC_alpha_0.75', 'stat_nn_ens_enet',
                'stat_nn_ens_lasso', 'stat_nn_ens_lasso_weighted']
nn_methods = ['point_qra', 'point_qrm', 'jsu1_lasso', 'jsu2_lasso', 'jsu3_lasso', 'jsu4_lasso', 'jsupEns_lasso',
              'jsuqEns_lasso', 'jsu1_enet', 'jsu2_enet', 'jsu3_enet', 'jsu4_enet', 'jsupEns_enet', 'jsuqEns_enet']
date_methods = ['lambda_0.0', 'BIC', 'BIC_alpha_0.25','BIC_alpha_0.5', 'BIC_alpha_0.75']
no_index_col_methods = ['lambda_0.0', 'point_qra']

# define risk values
alpha_values = np.array([0.5, 0.76, 0.8, 0.86, 0.9, 0.96])


# method to load the forecast in
def load_forecast(method):
    method_index = methods.index(method)
    method_name = method_names[method_index]
    datetime_name = 'Datetime'
    if method_name in date_methods:
        datetime_name = 'Date'
    if method_name == 'q_Ens':
        path = f'../Results/probabilistic_forecasts_time_span_{time_span}/{method_name}_forecast.csv'
    elif method_name in stat_methods:
        path = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_{method_name}.csv'
    elif method_name in nn_methods:
        path = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_{method_name}.csv'
    if method_name in no_index_col_methods:
        forecast = pd.read_csv(path, parse_dates=[datetime_name])
    else:
        forecast = pd.read_csv(path, index_col=0, parse_dates=[datetime_name])

    if method_name in date_methods:
        datetime_column = forecast['Date']
    else:
        datetime_column = forecast['Datetime']
    forecast['Date'] = pd.to_datetime(datetime_column)
    forecast['Day'] = datetime_column.dt.date
    forecast['Hour'] = datetime_column.dt.hour
    if time_span == 1:
        cutoff_date = pd.to_datetime('2019-11-07')
    else:
        cutoff_date = pd.to_datetime('2023-08-17')
    forecast = forecast[forecast['Date'] >= cutoff_date]
    forecast.reset_index(drop=True, inplace=True)

    return forecast


# create result dataframe
profit_df = pd.DataFrame(index=methods, columns=alpha_values)
profit_df['unlimited'] = 0
profit_df['naive'] = 0


# method to find optimal h1, h2 and h*
def optimize_hours(prices, mode):
    best_score = -np.inf
    best_combination = (0, 0, 0)
    n = len(prices)

    if mode == 'buy':
        for h_star in range(n - 2):
            for h1 in range(h_star + 1, n - 1):
                for h2 in range(h1 + 1, n):
                    score = 0.9 * prices[h2] - (1 / 0.9) * prices[h1] - (1 / 0.9) * prices[h_star]
                    if score > best_score:
                        best_score = score
                        best_combination = (h_star, h1, h2)
    elif mode == 'sell':
        for h_star in range(n - 2):
            for h1 in range(h_star + 1, n - 1):
                for h2 in range(h1 + 1, n):
                    score = 0.9 * prices[h2] - (1 / 0.9) * prices[h1] + 0.9 * prices[h_star]
                    if score > best_score:
                        best_score = score
                        best_combination = (h_star, h1, h2)

    return best_combination

# calculate the profits for each method
for method in methods:
    print(f'Begin method {method}')
    for alpha in alpha_values:
        method_profits = []

        # load the forecast
        forecast_df = load_forecast(method)

        battery_state = 1
        for day, day_data in forecast_df.groupby('Day'):
            day_prices = day_data['Price'].values
            p50 = day_data['Percentile_50'].values

            if battery_state == 0:
                h_star, h1, h2 = optimize_hours(p50, 'buy')
            elif battery_state == 2:
                h_star, h1, h2 = optimize_hours(p50, 'sell')
            else:
                h1, h2 = np.argmin(p50), np.argmax(p50)
                h_star = None

            h1 = int(h1)
            h2 = int(h2)
            if h_star is not None:
                h_star = int(h_star)

            # Determine bid and offer price
            bid_price = day_data[f'Percentile_{int((1 + alpha) / 2 * 100)}'].iloc[h1]
            offer_price = day_data[f'Percentile_{int((1 - alpha) / 2 * 100)}'].iloc[h2]

            # determine if bid and offer are accepted
            market_bid_accepted = day_prices[h1] <= bid_price
            market_offer_accepted = day_prices[h2] >= offer_price
            profit = 0

            # don't make bid and offer if expected to be not profitable
            if 0.9 * p50[h2] - (1 / 0.9) * p50[h1] < 0:
                profit = 0
            else:  # make bid and offer
                if market_bid_accepted and market_offer_accepted:
                    profit = 0.9 * day_prices[h2] - (1 / 0.9) * day_prices[h1]
                elif market_bid_accepted:
                    profit = -(1 / 0.9) * day_prices[h1]
                    battery_state = min(battery_state + 1, 2)
                elif market_offer_accepted:
                    profit = 0.9 * day_prices[h2]
                    battery_state = max(battery_state - 1, 0)

            # Extra bid and offers if battery state is not equal to 1
            if h_star is not None:
                if battery_state == 0:
                    profit -= (1 / 0.9) * day_prices[h_star]
                    battery_state += 1
                elif battery_state == 2:
                    profit += 0.9 * day_prices[h_star]
                    battery_state -= 1

            method_profits.append(profit)

        # add profit to result dataframe
        profit_df.loc[method, alpha] = round(np.sum(method_profits), 2)

# add the profits of the unlimited strategy
for method in methods:
    naive_profits = []
    unlimited_profits = []
    forecast_df = load_forecast(method)
    for day, day_data in forecast_df.groupby('Day'):
        day_prices = day_data['Price'].values
        p50 = day_data['Percentile_50'].values

        h1_unlimited, h2_unlimited = np.argmin(p50), np.argmax(p50)
        profit_unlimited = 0.9 * day_prices[h2_unlimited] - (1 / 0.9) * day_prices[h1_unlimited]
        if time_span == 1:
            h1, h2 = 4, 19
        else:
            h1, h2 = 13, 20
        profit_naive = 0.9 * day_prices[h2] - (1 / 0.9) * day_prices[h1]

        unlimited_profits.append(profit_unlimited)
        naive_profits.append(profit_naive)
    profit_df.loc[method, 'unlimited'] = round(np.sum(unlimited_profits), 2)
    profit_df.loc[method, 'naive'] = round(np.sum(naive_profits), 2)


# create profit per day dataframe
if time_span == 1:
    profit_per_day_df = (profit_df / 421).astype(float).round(2)
else:
    profit_per_day_df = (profit_df / 381).astype(float).round(2)
# save dataframes
profit_per_day_df.to_excel(f'profit_per_day_ts{time_span}.xlsx', index=True)
profit_df.to_excel(f'profit_ts{time_span}.xlsx', index=True)
