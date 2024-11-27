"""
Creates the graphs for in the results section
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np

# set colors
color_1 = '#A0522D'
color_2 = '#4682B4'
color_3 = '#8B0000'
color_4 = '#6A5ACD'

# choose time span
time_span = 1
folder_point = f'Results/point_forecasting_time_span_{time_span}'
folder_coverage_90 = f'Results/Evaluation_metrics/emperical_coverage_ts{time_span}_nc0.9.csv'
folder_coverage_50 = f'Results/Evaluation_metrics/emperical_coverage_ts{time_span}_nc0.5.csv'
folder_CPRS = f'Results/Evaluation_metrics/CPRS_ts{time_span}.csv'
folder_BIC = f'Results/probabilistic_forecasts_time_span_{time_span}/forecast_BIC.csv'

### create plot prediction intervals ###
forecast_BIC = pd.read_csv(folder_BIC, index_col=0).reset_index(drop=True)
forecast_BIC['Date'] = pd.to_datetime(forecast_BIC['Date'])

# Extract unique dates from the dataset (ignoring the time)
forecast_BIC['Day'] = forecast_BIC['Date'].dt.date  # Add a new column for just the day
unique_days = forecast_BIC['Day'].unique()  # Get all unique days

# Iterate through each unique day
for day in unique_days:
    # Filter data for the current day
    daily_data = forecast_BIC[forecast_BIC['Day'] == day]

    # Check if all prices are within the 50% prediction interval
    if (daily_data['Price'] >= daily_data['Percentile_5']).all() and \
            (daily_data['Price'] <= daily_data['Percentile_95']).all():
        print(f"The first day where all prices are within the 90% prediction interval is: {day}")
        break
else:
    print("No day found where all prices are within the 50% prediction interval.")

# Filter data for the second day (24-48 hours)
start_time = forecast_BIC['Date'].min() + pd.Timedelta(days=1)  # Start of the second day
end_time = start_time + pd.Timedelta(hours=24)  # End of the second day
forecast_BIC = forecast_BIC[(forecast_BIC['Date'] >= start_time) & (forecast_BIC['Date'] < end_time)]


# Create a figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the actual prices
ax.plot(forecast_BIC['Date'], forecast_BIC['Price'], label='Actual Price', color='black', linewidth=2)

# 90% prediction interval (5-95)
ax.fill_between(
    forecast_BIC['Date'],
    forecast_BIC['Percentile_5'],
    forecast_BIC['Percentile_95'],
    color='blue', alpha=0.2, label='90% Prediction Interval'
)

# 50% prediction interval (25-75)
ax.fill_between(
    forecast_BIC['Date'],
    forecast_BIC['Percentile_25'],
    forecast_BIC['Percentile_75'],
    color='orange', alpha=0.4, label='50% Prediction Interval'
)

# Add labels, title, and legend
ax.set_title('Predicted Price and Prediction Intervals', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price', fontsize=12)
ax.legend()
ax.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
test=5


### create point forecasting graphs of mae and rmse ###
# list to store loaded DataFrames along with their filenames
point_results = []

# loop through all files in the folder and load them as DataFrames
for filename in os.listdir(folder_point):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_point, filename)
        df = pd.read_csv(file_path)
        point_results.append((filename, df))  # Store filename and DataFrame as a tuple

# list to store the error metrics (MAE and RMSE)
errors_dic = []

# loop through the list of filenames and dataframes
for filename, df in point_results:
    # extract the window number from the filename
    match = re.search(r'window_(\d+)', filename)
    if match:
        window = int(match.group(1))  # Extract the window number from the filename

        # calculate MAE and RMSE if the necessary columns are present
        if 'price_real' in df.columns and 'forecasted_price' in df.columns:
            mae = (df['price_real'] - df['forecasted_price']).abs().mean()
            rmse = np.sqrt(((df['price_real'] - df['forecasted_price']) ** 2).mean())

            # append the results to the errors_dic list
            errors_dic.append({
                'window': window,  # Use the window number
                'MAE': mae,
                'RMSE': rmse
            })
        else:
            print(f"DataFrame for window {window} is missing required columns.")
    else:
        print(f"Filename {filename} does not match the expected format.")

# extract the data from the dictionairy for plotting
windows = [error['window'] for error in errors_dic]
maes = [error['MAE'] for error in errors_dic]
rmses = [error['RMSE'] for error in errors_dic]

# define the windows to be highlighted in black
highlight_windows = list(range(56, 729, 28))

# create scatter plot of MAE vs Window
plt.figure(figsize=(10, 5))
# First, plot all the points
for window, mae in zip(windows, maes):
    plt.scatter(window, mae, color=color_4, s=40)  # Regular points with blue color and smaller size

# plot the highlighted windows (black) on top
for window, mae in zip(windows, maes):
    if window in highlight_windows:
        plt.scatter(window, mae, color='black', s=60)  # Highlighted points with black color and slightly larger size

# show the graph
plt.xlabel('Window')
plt.ylabel('MAE')
plt.title('MAE vs. Window')
plt.grid(True)
plt.show()

# create scatter plot of RMSE vs window
plt.figure(figsize=(10, 5))
# plot all the points
for window, rmse in zip(windows, rmses):
    plt.scatter(window, rmse, color=color_2, s=40)  # Regular points with red color and smaller size

# plot the highlighted windows black on top
for window, rmse in zip(windows, rmses):
    if window in highlight_windows:
        plt.scatter(window, rmse, color='black', s=60)  # Highlighted points with black color and slightly larger size

# show the plot
plt.xlabel('Window')
plt.ylabel('RMSE')
plt.title('RMSE vs. Window')
plt.grid(True)
plt.show()


### create plot for emperical coverage ###
# plot coverage for nnominal coverage equal to 0.9
# get data for lambda and PICP_values
coverage_df = pd.read_csv(folder_coverage_90, index_col=0)
coverage_BIC = coverage_df.loc['Empirical_coverage', 'BIC']
# drop BIC column from dataframe
coverage_df = coverage_df.drop(coverage_df.columns[-1], axis=1)
lambda_values = coverage_df.columns.astype(float)
picp_values = coverage_df.loc['Empirical_coverage'].values

# scatter plot for PICP vs lambda values
plt.figure(figsize=(10, 6))
plt.scatter(lambda_values, picp_values, color='b', edgecolors='b', facecolors='none', label='LQRA(\u03BB)')
plt.xlabel('\u03BB')
plt.ylabel('Empirical Coverage (PICP)')
plt.title(f'Empirical coverage LQRA time span {time_span}')
plt.xscale('log')
plt.ylim(0.7, 0.92)
plt.axhline(y=0.9, color='black', linestyle='--', label='Target PICP = 0.9')
plt.axhline(y=coverage_BIC, color=color_1, linestyle='-', label='LQRA(BIC)')
plt.xticks([1, 10, 100, 1000], ['10^0', '10^1', '10^2', '10^3'])
plt.legend()
plt.show()

# plot coverage for nominal coverage equal to 0.5
# get data for lambda and PICP_values
coverage_df = pd.read_csv(folder_coverage_50, index_col=0)
coverage_BIC = coverage_df.loc['Empirical_coverage', 'BIC']
# drop BIC column from dataframe
coverage_df = coverage_df.drop(coverage_df.columns[-1], axis=1)
lambda_values = coverage_df.columns.astype(float)
picp_values = coverage_df.loc['Empirical_coverage'].values

# scatter plot for PICP vs lambda values
plt.figure(figsize=(10, 6))
plt.scatter(lambda_values, picp_values, color='b', edgecolors='b', facecolors='none', label='LQRA(\u03BB)')
plt.xlabel('\u03BB')
plt.ylabel('Empirical Coverage (PICP)')
plt.title(f'Empirical coverage LQRA time span {time_span}')
plt.xscale('log')
plt.ylim(0.35, 0.52)
plt.axhline(y=0.5, color='black', linestyle='--', label='Target PICP = 0.5')
plt.axhline(y=coverage_BIC, color=color_1, linestyle='-', label='LQRA(BIC)')
plt.xticks([1, 10, 100, 1000], ['10^0', '10^1', '10^2', '10^3'])
plt.legend()
plt.show()


### create plot for CPRS ###
# get data for lambda and APS values
CRPS_df = pd.read_csv(folder_CPRS, index_col=0)
CRPS_BIC = CRPS_df.loc['CRPS', 'BIC']
# drop BIC column from dataframe
CRPS_df = CRPS_df.drop(CRPS_df.columns[-1], axis=1)
lambda_values = CRPS_df.columns.astype(float)
APS = CRPS_df.loc['CRPS'].values

# scatter plot for APS vs lambda values
plt.figure(figsize=(10, 6))
plt.scatter(lambda_values, APS, color='b', edgecolors='b', facecolors='none', label='LQRA(\u03BB)')
plt.xlabel('\u03BB')
plt.ylabel('Average Pinball Score (APS)')
plt.title(f'Average pinball score LQRA time span {time_span}')
plt.xscale('log')
plt.xticks([1, 10, 100, 1000], ['10^0', '10^1', '10^2', '10^3'])
plt.axhline(y=CRPS_BIC, color=color_1, linestyle='--', label='LQRA(BIC)')
plt.legend()
plt.show()



