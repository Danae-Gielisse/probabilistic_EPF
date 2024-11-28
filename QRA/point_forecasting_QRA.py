"""
Code based on matlab code from Uniejewski, B., & Weron, R. (2021).
Regularized quantile regression averaging for probabilistic electricity price forecasting. Energy Economics, 95, 105121.

Performs the point forecasting of het QRA based methods
"""

import pandas as pd
import os
import numpy as np
from scipy.stats import median_abs_deviation
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor

# import the data
processed_data_folder = '../Data/processed data'
data = pd.read_csv(os.path.join(processed_data_folder, "data.csv"))

# choose time span
time_span = 2

# set correct values for the different time spans
if time_span == 1:
    start_date = '2016-01-01'
    end_date = '2021-01-01'
    results_point_forecasts_time_span_folder = '../Results/point_forecasting_time_span_1'
else:
    start_date = '2020-01-01'
    end_date = '2024-09-01'
    results_point_forecasts_time_span_folder = '../Results/point_forecasting_time_span_2'

# filter on start date and end date (not included)
data = data[(data['datetime'] >= start_date) & (data['datetime'] < end_date)]
if time_span == 2:
    data = data.reset_index(drop=True)

# select data columns needed for LQRA
selected_columns = ["datetime", "price", "load_forecast", "solar", "total_wind", "total_generation"]
data = data[selected_columns]

# add an hour column and a day of the week column
data['datetime'] = pd.to_datetime(data['datetime'])
data['hour'] = data['datetime'].dt.hour + 1
data['day_of_week'] = (data['datetime'].dt.dayofweek + 1) % 7
data['day_of_week'] = data['day_of_week'].replace(0, 7)

# create dataframes apart
Price = data["price"]
Exg = data[["load_forecast", "total_generation"]]
Dummies = data['day_of_week'].to_frame()
Date = data['datetime'].to_frame()


'''
Function that saves the results of the point forecasting
'''
def save_forecast(win):
    print('started win ' + str(win))
    forecast = point(Date, Price, Exg, Dummies, win, 728)
    name = "window_" + str(win)
    output_path = os.path.join(results_point_forecasts_time_span_folder, name + ".csv")
    forecast.to_csv(output_path, index=False)

'''
Function to perform the asinh transformation
'''
def transformation(X):
    a = np.median(X)
    b = median_abs_deviation(X, scale=1) / norm.ppf(0.75)
    Y = np.arcsinh((X - a) / b)
    return Y, a, b

'''
Function to undo the transformation of the price
'''
def inv(Y, a, b):
    X = np.mean(np.sinh(Y) * b + a)
    return X

'''
Predicts the price for a certain day and calibration window
'''
def ARX(price, exg, dummies, calib):
    # transform the price data and the exogenous variables
    price, a, b = transformation(price)
    load_forecast, _, _= transformation(exg["load_forecast"])
    total_generation, _, _ = transformation(exg["total_generation"])
    exg = pd.concat([load_forecast, total_generation], axis=1)
    # create empty vector to store the predictions
    prediction = np.full(24, np.nan)

    # change price data to matrix with rows indicating the 24 hours and columns indicating the calibraqtion window days
    reshaped_price = np.reshape(price, (24, calib), order='F')
    # determine for every day in the calibration window the lowest price and take the index
    idx_min = np.argmin(reshaped_price, axis=0)
    # determine te index of the lowest price per day for the column vector price
    offsets = np.arange(0, calib * 24, 24)
    idx_min = idx_min + offsets
    # remove the indexes for the first 7 days in the calibration window (WHY!!)
    idx_min = idx_min[6:]
    # determine for every day in the calibration window the highest price and take the index
    idx_max = np.argmax(reshaped_price, axis=0)
    # determine te index of the lowest price per day for the column vector price
    idx_max = idx_max + offsets
    # remove the indexes for the first 7 days in the calibration window (WHY!!)
    idx_max = idx_max[6:]
    # indexes of beginning of a day after first 7 days to the end of the calibration window
    idx24 = np.arange(7 * 24 - 1, 24 * calib, 24)

    # predict the price for every hour
    for hour in range(1, 25):
        # calculate index of d-1,h
        idx = np.arange(hour + 6 * 24 - 1, 24 * calib, 24)
        # calculate index of d,h
        idxEx = np.arange(hour + 7 * 24 - 1, 24 * calib + 24, 24)
        Y = price[idx[1:]]
        X = np.column_stack([
            price[idx],  # Y_{d-1,h}
            price[idx - 24],  # Y{d-2,h}
            price[idx - 144],  # Y{d-7,h}
            price[idx24],  # Y_{d-1,24}
            price[idx_min],  # Y_{d-1, min}
            price[idx_max],  # Y_{d-1, max}
            (dummies[idxEx] == 7),  # D_{Sun}
            (dummies[idxEx] == 1),  # D_{Mon}
            (dummies[idxEx] == 6),  # D_{Sat}
        ])
        # add exogenous variables to X (Z^1_{d,h}, Z^2_{d,h}
        for i in range(exg.shape[1]):
            X = np.column_stack([X, exg.iloc[idxEx, i]])
        # delete regressor if hour equals 24 (otherwise two times the same regressor)
        if hour == 24:
            X = np.delete(X, 3, axis=1)
        # predict the price for every hour
        X_fut = X[-1, :]
        beta, _, _, _ = np.linalg.lstsq(X[:-1, :], Y, rcond=None)
        err = Y - np.dot(X[:-1, :], beta)
        prediction[hour - 1] = inv(np.dot(X_fut, beta) + err, a, b)
    return prediction

'''
Performs point forecasting for all days in the test period
'''
def point(Date, Price, Exg, Dummies, calib, max_calib):
    N = len(Price) # number of observations
    D = N // 24 # number of days
    # create matrix to store all forecasts for test period
    forecast = np.full(((D-max_calib)*24, 3), np.nan, dtype=object)
    # for loop over test period
    for day in range(max_calib, D):
        idx = np.arange((day - calib) * 24, day * 24)
        cur_idx = day * 24 - 1 + np.arange(1, 25)
        price = Price[idx].reset_index(drop=True)
        exg = Exg.loc[np.concatenate((idx, cur_idx)), :].reset_index(drop=True)
        dummies = Dummies.loc[np.concatenate((idx, cur_idx)), :].reset_index(drop=True)
        start_idx = (day - max_calib) * 24
        end_idx = start_idx + 24
        forecast[start_idx:end_idx, 0] = Date.iloc[cur_idx, :].astype(str).to_numpy().flatten()
        forecast[start_idx:end_idx, 1] = Price.iloc[cur_idx].to_numpy()
        forecast[start_idx:end_idx, 2] = ARX(price, exg, dummies.iloc[:, 0], calib)
        print("Day " + str(day) + " from calibration window " + str(calib) + " is done")

    # change forecast to a df
    forecast_df = pd.DataFrame(forecast, columns=['datetime', 'price_real', 'forecasted_price'])
    forecast_df['datetime'] = pd.to_datetime(forecast_df['datetime'], errors='coerce')
    forecast_df['price_real'] = forecast_df['price_real'].astype(float)
    forecast_df['forecasted_price'] = forecast_df['forecasted_price'].astype(float)
    return forecast_df

# perform parallel computation of the point forecasting
if __name__ == '__main__':
    # execute forecasts in parallel
    with ProcessPoolExecutor() as executor:
        executor.map(save_forecast, range(28, 729))


