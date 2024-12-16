"""
Code based on matlab code from Uniejewski, B., & Weron, R. (2021).
Regularized quantile regression averaging for probabilistic electricity price forecasting. Energy Economics, 95, 105121.

Computation of the probabilistic forecasts for the QRA methods
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import os
from joblib import Parallel, delayed
import cvxpy as cp

# choose to use LQRA or EQRA
LQRA = True # if False, then perform EQRA
# choose alpha value for EQRA
alph = 0.25 # choose 0.25, 0.5 or 0.75
# choose time span
time_span = 1

# define parameters
WIN = range(56, 729, 28)
LAMBDA = np.concatenate(([0], np.logspace(-1, 3, 19)))

# define folders
results_point_forecasts_time_span_folder = f'../Results/point_forecasting_time_span_{time_span}'
output_folder = f'../Results/probabilistic_forecasts_time_span_{time_span}'

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

        if LQRA:
            prediction[hour, :], beta_matrix[hour] = QRA_LASSO(Y, X[:-1, :], X_fut, tau, lambda_val)
        else:
            prediction[hour, :], beta_matrix[hour] = QRA_elastic_net(Y, X[:-1, :], X_fut, tau, lambda_val, alph)

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


def QRA_elastic_net(y, x, x_fut, tau, lambda_val, gamma):
    n, m = x.shape
    forecast = np.zeros(len(tau))
    beta_list = []

    # convert input to array
    x = np.asarray(x)
    y = np.asarray(y)
    x_fut = np.asarray(x_fut)

    # CVXPY variabeles
    beta = cp.Variable(m)
    u = cp.Variable(n)  # positive deviations
    v = cp.Variable(n)  # negative deviations

    for i, q in enumerate(tau):
        # construct objective function
        l1_term = lambda_val * (1 - gamma) * cp.norm1(beta)
        l2_term = lambda_val * gamma * cp.sum_squares(beta)
        quantile_term = q * cp.sum(u) + (1 - q) * cp.sum(v)
        objective = cp.Minimize(quantile_term + l1_term + l2_term)

        # construct constraints
        constraints = [
            y - x @ beta == u - v,
            u >= 0,
            v >= 0
        ]

        # solve minimalization problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, abstol=1e-5, reltol=1e-5, max_iters=5000, verbose=False)

        # save results
        beta_vals = beta.value
        forecast[i] = np.dot(x_fut, beta_vals)
        beta_list.append(beta_vals)

    # sort the quantiles
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

    test = probabilistic_single_day(date, Y, X, calib, tau, lambda_val, calib)

    results = Parallel(n_jobs=-1)(
        delayed(probabilistic_single_day)(date, Y, X, calib, tau, lambda_val, day)
        for day in range(calib, D)
    )

    forecast, beta_df = combine_results(results)

    column_names = ['Date', 'Price'] + [f'Percentile_{i}' for i in range(1, 100)]
    forecast_df = pd.DataFrame(forecast, columns=column_names)

    if not os.path.exists(folder):
        os.makedirs(folder)

    if LQRA:
        forecast_file = os.path.join(folder, f'forecast_lambda_{lambda_val}.csv')
        beta_file = os.path.join(folder, f'beta_lambda_{lambda_val}.csv')
    else:
        forecast_file = os.path.join(folder, f'forecast_lambda_{lambda_val}_alpha_{alph}.csv')
        beta_file = os.path.join(folder, f'beta_lambda_{lambda_val}_alpha_{alph}.csv')

    forecast_df.to_csv(forecast_file, index=False)
    beta_df.to_csv(beta_file, index=False)

if __name__ == "__main__":
    lambda_values = LAMBDA[[19]]
    for l in lambda_values:
        l_val = l
        compute_for_lambda(l_val, output_folder)
