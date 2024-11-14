"""
Computation of the probabilistic forecasts
"""


import numpy as np
import pandas as pd
from scipy.optimize import linprog
import os
from joblib import Parallel, delayed
# from qpsolvers import solve_qp

# parameters
WIN = range(56, 729, 28)
LAMBDA = np.concatenate(([0], np.logspace(-1, 3, 19)))
time_span = 2

# define folders
results_point_forecasts_time_span_folder = f'Results/point_forecasting_time_span_{time_span}'
output_folder = f'Results/probabilistic_forecasts_time_span_{time_span}'

# load point forecast data
X_list = []
for win in WIN:
    data = pd.read_csv(os.path.join(results_point_forecasts_time_span_folder, "window_" + str(win)) + ".csv").to_numpy()
    X_list.append(data)
X = np.stack(X_list, axis=2)

# extract date, Y, and X arrays
date = X[:, 0, 0]
Y = X[:, 1, 0]
X = X[:, 2, :]


def probabilistic_single_day(Date, Price, X, calib, tau, lambda_val, day):
    forecast = np.full((24, 2 + len(tau)), np.nan, dtype=object)
    beta_results = []

    print(f"Processing day {day} for lambda {lambda_val}")
    idx = np.arange((day - calib) * 24, day * 24)
    cur_idx = np.arange(day * 24, (day + 1) * 24)

    price = Price[idx]
    x = X[np.r_[idx, cur_idx], :]

    forecast[:, 0] = Date[cur_idx]
    forecast[:, 1] = Price[cur_idx]

    prediction, beta_day_hour = QRA(price, x, tau, lambda_val)

    forecast[:, 2:] = prediction

    for hour in range(24):
        for j, t in enumerate(tau):
            beta_results.append({
                'day': day,
                'hour': hour,
                'tau': t,
                'lambda': lambda_val,
                'beta': beta_day_hour[hour][j]
            })

    return forecast, beta_results

def QRA(price, x, tau, lambda_val):
    prediction = np.full((24, len(tau)), np.nan)
    beta_matrix = [[None] * len(tau) for _ in range(24)]

    for hour in range(24):
        Y = price[hour::24]
        X = x[hour::24, :]
        X_fut = X[-1, :]

        prediction[hour, :], beta_matrix[hour] = QRA_LASSO(Y, X[:-1, :], X_fut, tau, lambda_val)

    return prediction, beta_matrix

def QRA_LASSO(y, x, x_fut, tau, lambda_val):
    n, m = x.shape
    forecast = np.full(len(tau), np.nan)
    beta_list = []

    Aeq = np.hstack((np.eye(n), -np.eye(n), x, -x))
    beq = y
    lb = np.zeros(2 * (n + m))
    ub = np.inf * np.ones(2 * (n + m))
    options = {'disp': False}

    for i, q in enumerate(tau):
        f = np.concatenate([q * np.ones(n), (1 - q) * np.ones(n), lambda_val + np.zeros(2 * m)])
        res = linprog(f, A_eq=Aeq, b_eq=beq, bounds=list(zip(lb, ub)), options=options)

        beta = res.x[-2 * m: -m] - res.x[-m:]
        forecast[i] = np.dot(x_fut, beta)
        beta_list.append(beta)

    forecast = np.sort(forecast)
    return forecast, beta_list


def combine_results(results):
    forecasts, beta_results = zip(*results)
    combined_forecast = np.vstack(forecasts)
    combined_beta = [item for sublist in beta_results for item in sublist]
    return combined_forecast, pd.DataFrame(combined_beta)


def compute_for_lambda(lambda_val, folder):
    N = len(Y)
    D = N // 24
    calib = 364
    tau = np.arange(1, 100) / 100

    results = Parallel(n_jobs=-1)(
        delayed(probabilistic_single_day)(date, Y, X, calib, tau, lambda_val, day)
        for day in range(calib, D)
    )

    forecast, beta_df = combine_results(results)

    column_names = ['Date', 'Price'] + [f'Percentile_{i}' for i in range(1, 100)]
    forecast_df = pd.DataFrame(forecast, columns=column_names)

    if not os.path.exists(folder):
        os.makedirs(folder)

    forecast_file = os.path.join(folder, f'forecast_lambda_{lambda_val}.csv')
    forecast_df.to_csv(forecast_file, index=False)
    beta_file = os.path.join(folder, f'beta_lambda_{lambda_val}.csv')
    beta_df.to_csv(beta_file, index=False)

if __name__ == "__main__":
    lambda_values = LAMBDA[[13]]
    for l in lambda_values:
        l_val = l
        compute_for_lambda(l_val, output_folder)