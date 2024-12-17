"""
Code based on matlab code from Uniejewski, B., & Weron, R. (2021).
Regularized quantile regression averaging for probabilistic electricity price forecasting. Energy Economics, 95, 105121.

Computation of the probabilistic forecasts for the QRA methods
"""


import numpy as np
import pandas as pd
from scipy.optimize import linprog
import os
import pickle
from joblib import Parallel, delayed

# choose time span and regularization
time_span = 2
regularization = 'lasso'
run_list = [1, 2, 3, 4]
# define folders
data_folder = '../Data/processed data/data.csv'
results_point_forecasts_time_span_folder = f'../Results/point_forecasts_NN_time_span_{time_span}'
output_folder = f'../Results/percentiles_NN_time_span_{time_span}'

data = pd.read_csv(data_folder)
if time_span == 1:
    subset = data.loc[29376:43847, ['datetime']]
else:
    subset = data.loc[62472:75985, ['datetime']]

# load point forecast data
X_list = []
for run in run_list:
    point_df = pd.read_csv(os.path.join(results_point_forecasts_time_span_folder, f'point{run}_{regularization}') + ".csv")
    point_df = pd.concat([subset.reset_index(drop=True), point_df.reset_index(drop=True)], axis=1).to_numpy()
    X_list.append(point_df)
X = np.stack(X_list, axis=2)

# extract date, Y, and X arrays
date = X[:, 0, 0]
Y = X[:, 1, 0]
X = X[:, 2, :]

def probabilistic_single_day(Date, Price, X, calib, tau, day):
    print(f"Processing day {day}")
    forecast = np.full((24, 2 + len(tau)), np.nan, dtype=object)

    idx = np.arange((day - calib) * 24, day * 24)
    cur_idx = np.arange(day * 24, (day + 1) * 24)

    price = Price[idx]
    x = X[np.r_[idx, cur_idx], :]

    forecast[:, 0] = Date[cur_idx]
    forecast[:, 1] = Price[cur_idx]

    prediction = QRA(price, x, tau)

    forecast[:, 2:] = prediction

    return forecast


def QRA(price, x, tau):
    prediction = np.full((24, len(tau)), np.nan)

    for hour in range(24):
        Y = price[hour::24]
        X = x[hour::24, :]
        X_fut = X[-1, :]

        prediction[hour, :] = qra(Y, X[:-1, :], X_fut, tau)

    return prediction

def qra(y, x, x_fut, tau):
    n, m = x.shape
    forecast = np.full(len(tau), np.nan)

    Aeq = np.hstack((np.eye(n), -np.eye(n), x, -x))
    beq = y
    lb = np.zeros(2 * (n + m))
    ub = np.inf * np.ones(2 * (n + m))
    options = {'disp': False}

    for i, q in enumerate(tau):
        f = np.concatenate([q * np.ones(n), (1 - q) * np.ones(n), 0 + np.zeros(2 * m)])
        res = linprog(f, A_eq=Aeq, b_eq=beq, bounds=list(zip(lb, ub)), options=options)

        beta = res.x[-2 * m: -m] - res.x[-m:]
        forecast[i] = np.dot(x_fut, beta)

    forecast = np.sort(forecast)
    return forecast


def combine_results(results):
    forecasts = list(zip(*results))
    combined_forecast = np.vstack(forecasts)

    return combined_forecast


def compute(folder):
    N = len(Y)
    D = N // 24
    calib = 182
    tau = np.arange(1, 100) / 100

    results = []

    for day in range(calib, D):
        result = probabilistic_single_day(date, Y, X, calib, tau, day)
        results.append(result)

    forecast = combine_results(results)

    column_names = ['Datetime', 'Price'] + [f'Percentile_{i}' for i in range(1, 100)]
    forecast_df = pd.DataFrame(forecast, columns=column_names)

    if not os.path.exists(folder):
        os.makedirs(folder)

    forecast_file = os.path.join(folder, 'percentiles_point_qra.csv')
    forecast_df.to_csv(forecast_file, index=False)

# perform the QRA
compute(output_folder)
