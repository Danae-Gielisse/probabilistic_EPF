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
folder_point = f'../Results/point_forecasting_time_span_{time_span}'
folder_BIC = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_BIC.csv'

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


### create point forecasting graphs of mae and rmse ###
# list to store loaded DataFrames along with their filenames
point_results = []

# loop through all files in the folder and load them as DataFrames
for filename in os.listdir(folder_point):
    if filename.startswith('window'):
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

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# First subplot: MAE vs Window
for window, mae in zip(windows, maes):
    axes[0].scatter(window, mae, color=color_4, s=40)  # Regular points

for window, mae in zip(windows, maes):
    if window in highlight_windows:
        axes[0].scatter(window, mae, color='black', s=60)  # Highlighted points

axes[0].set_xlabel('Window length')
axes[0].set_ylabel('MAE')
axes[0].grid(True)
axes[0].set_title(f'MAE for each window length time span {time_span}')

# Second subplot: RMSE vs Window
for window, rmse in zip(windows, rmses):
    axes[1].scatter(window, rmse, color=color_2, s=40)  # Regular points

for window, rmse in zip(windows, rmses):
    if window in highlight_windows:
        axes[1].scatter(window, rmse, color='black', s=60)  # Highlighted points

axes[1].set_xlabel('Window length')
axes[1].set_ylabel('RMSE')
axes[1].grid(True)
axes[1].set_title(f'RMSE for each window length time span {time_span}')

# Show the figure
plt.tight_layout()
plt.savefig(f"mae_rmse_figure_ts{time_span}.png", dpi=300, bbox_inches="tight")
plt.show()

### create emperical coverage graphs ###
alpha_list = [0.25, 0.5, 0.75]
coverage_list = [0.5, 0.9]

# create figure
fig, axes = plt.subplots(4, 2, figsize=(8.27, 11.69))
axes = axes.flatten()
plot_idx = 0

for coverage in coverage_list:
    # Load LQRA data
    folder_coverage_LQRA = f'../Results/Evaluation_metrics/emperical_coverage_ts{time_span}_nc{coverage}.csv'
    coverage_df_LQRA = pd.read_csv(folder_coverage_LQRA, index_col=0)
    coverage_BIC = coverage_df_LQRA.loc['Empirical_coverage', 'BIC']
    coverage_df_LQRA = coverage_df_LQRA.drop(coverage_df_LQRA.columns[-1], axis=1)
    lambda_values_LQRA = coverage_df_LQRA.columns.astype(float)
    picp_values_LQRA = coverage_df_LQRA.loc['Empirical_coverage'].values

    ax = axes[plot_idx]
    ax.scatter(lambda_values_LQRA, picp_values_LQRA, color=color_4, edgecolors=color_4, facecolors='none', label='LQRA(\u03BB)')
    ax.set_xlabel('\u03BB')
    ax.set_ylabel('PICP')
    ax.set_title(f'LQRA, PINC={coverage}')
    ax.set_xscale('log')
    if coverage == 0.9:
        ax.set_ylim(0.7, 0.92)
    else:
        ax.set_ylim(0.3, 0.52)
    ax.axhline(y=coverage, color='black', linestyle='--', label=f'Target PICP = {coverage}')
    ax.axhline(y=coverage_BIC, color=color_3, linestyle='-', label='LQRA(BIC)')
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['10^0', '10^1', '10^2', '10^3'])
    ax.legend(fontsize=6)
    xlim_min, xlim_max = ax.get_xlim()
    plot_idx += 1

# create EQRA graphs
for alpha in alpha_list:
    for coverage in coverage_list:
        # load EQRA data
        folder_coverage_EQRA = f'../Results/Evaluation_metrics/empirical_coverage_alpha_{alpha}_nc{coverage}_ts{time_span}.csv'
        coverage_df_EQRA = pd.read_csv(folder_coverage_EQRA, index_col=0)
        coverage_BIC_EQRA = coverage_df_EQRA.iloc[0, -1]
        coverage_df_EQRA = coverage_df_EQRA.drop(coverage_df_EQRA.columns[-1], axis=1)
        lambda_values_EQRA = coverage_df_EQRA.columns.astype(float)
        picp_values_EQRA = coverage_df_EQRA.values[0]

        # plot EQRA
        ax = axes[plot_idx]
        ax.scatter(lambda_values_EQRA, picp_values_EQRA, color=color_2, edgecolors=color_2, facecolors='none', label=f'EQRA(\u03BB, α={alpha})')
        ax.set_xlabel('\u03BB')
        ax.set_ylabel('PICP')
        ax.set_title(f'EQRA, α={alpha}, PINC={coverage}')
        ax.set_xscale('log')
        if coverage == 0.9:
            ax.set_ylim(0.7, 0.92)
        else:
            ax.set_ylim(0.3, 0.52)
        ax.axhline(y=coverage, color='black', linestyle='--', label=f'Target PICP = {coverage}')
        ax.axhline(y=coverage_BIC_EQRA, color=color_3, linestyle='-', label=f'EQRA(BIC-{alpha})')
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(['10^0', '10^1', '10^2', '10^3'])
        ax.set_xlim(xlim_min, xlim_max)
        ax.legend(fontsize=6)
        plot_idx += 1

# plot the figure
plt.tight_layout()
plt.savefig(f"empirical_coverage_plots_ts{time_span}.png", dpi=300, bbox_inches="tight")  # save to png
plt.show()

### create plots for CRPS ###
alpha_list = [0, 0.25, 0.5, 0.75]

# create plot
fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.845))
axes = axes.flatten()

# determine x and y limits
folder_CRPS = f'../Results/Evaluation_metrics/CRPS_ts{time_span}.csv'
CRPS_df = pd.read_csv(folder_CRPS, index_col=0)
CRPS_LQRA = CRPS_df.loc['CRPS'].values
y_min, y_max = CRPS_LQRA.min(), CRPS_LQRA.max()
margin = (y_max - y_min) * 0.10
y_min -= margin
y_max += margin
CRPS_df = CRPS_df.drop(CRPS_df.columns[-1], axis=1)
lambda_values_common = CRPS_df.columns.astype(float)
lambda_values_common = lambda_values_common[lambda_values_common > 0]  # Remove zero if present
x_min, x_max = lambda_values_common.min(), lambda_values_common.max()
margin = (x_max - x_min) * 0.5
x_min -= 1
x_max += (x_max - x_min) * 0.5


# plot CRPS for every alpha
for i, alpha in enumerate(alpha_list):
    if alpha == 0:
        folder_CRPS = f'../Results/Evaluation_metrics/CRPS_ts{time_span}.csv'
    else:
        folder_CRPS = f'../Results/Evaluation_metrics/CRPS_ts{time_span}_alpha_{alpha}.csv'

    CRPS_df = pd.read_csv(folder_CRPS, index_col=0)
    CRPS_BIC = CRPS_df.loc['CRPS', 'BIC']
    CRPS_df = CRPS_df.drop(columns=[col for col in CRPS_df.columns if 'BIC' in col])
    lambda_values = CRPS_df.columns.astype(float)
    CRPS = CRPS_df.loc['CRPS'].values

    ax = axes[i]
    if alpha == 0:
        ax.scatter(lambda_values, CRPS, color=color_4, edgecolors=color_4, facecolors='none', label='LQRA(\u03BB)')
        ax.set_title(f'CRPS LQRA')
        ax.axhline(y=CRPS_BIC, color=color_3, linestyle='--', label='LQRA(BIC)')
    else:
        ax.scatter(lambda_values, CRPS, color=color_2, edgecolors=color_2, facecolors='none', label='EQRA(\u03BB)')
        ax.set_title(f'CRPS EQRA, α={alpha}')
        ax.axhline(y=CRPS_BIC, color=color_3, linestyle='--', label=f'EQRA(BIC-{alpha})')
    ax.set_xlabel('\u03BB')
    ax.set_ylabel('CRPS')

    ax.set_xscale('log')
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['10^0', '10^1', '10^2', '10^3'])
    if alpha == 0:
        xlim_min, xlim_max = ax.get_xlim()
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=8)

plt.tight_layout()  # Ensure subplots are properly arranged
plt.savefig(f"CRPS_plots_ts{time_span}.png", dpi=300, bbox_inches="tight")  # Save the figure as PNG
plt.show()


### create kupiec plots ###
methods = ['QRA', 'Stat-qEns', 'Stat-QRM', 'DNN-QRA', 'DNN-QRM', 'LQRA(BIC)', 'EQRA(BIC-0.25)', 'EQRA(BIC-0.5)',
             'EQRA(BIC-0.75)', 'DDNN-L-1', 'DDNN-L-2', 'DDNN-L-3', 'DDNN-L-4', 'DDNN-L-pEns', 'DDNN-L-qEns',
             'DDNN-E-1', 'DDNN-E-2', 'DDNN-E-3', 'DDNN-E-4', 'DDNN-E-pEns', 'DDNN-E-qEns', 'DDNN-LQRA(BIC)-qEns',
             'DDNN-EQRA(BIC)-qEns', 'DDNN-LQRA(BIC)-wqEns']

method_names = ['LQRA(0)', 'Stat-q_Ens', 'Stat-qrm', 'qra', 'qrm', 'LQRA(BIC)', 'EQRA(BIC-0.25)',
                'EQRA(BIC-0.5)', 'EQRA(BIC-0.75)', 'jsu1_lasso', 'jsu2_lasso', 'jsu3_lasso', 'jsu4_lasso',
                'jsupEns_lasso', 'jsuqEns_lasso', 'jsu1_enet', 'jsu2_enet', 'jsu3_enet', 'jsu4_enet', 'jsupEns_enet',
                'jsuqEns_enet', 'ens_lasso', 'ens_enet', 'ens_lasso_weighted']

hours = np.arange(1, 25)  # Uren van de dag (1 t/m 24)
num_methods = len(methods)

ACE_50, ACE_90, p_50, p_90 = None, None, None, None
nc_list = [0.5, 0.9]
for nc in nc_list:
    for method in methods:
        method_index = methods.index(method)
        method_name = method_names[method_index]
        folder = f'../Results/Evaluation_metrics/ec_kupiec_dfs/{method_name}_nc{nc}_ts{time_span}.csv'
        ec_kupiec_df = pd.read_csv(folder, index_col=0)

        if nc == 0.5:
            if ACE_50 is None:
                ACE_50 = ec_kupiec_df.iloc[0:1]
                p_50 = ec_kupiec_df.iloc[1:2]
            else:
                ACE_50 = pd.concat([ACE_50, ec_kupiec_df.iloc[0:1]], ignore_index=True)
                p_50 = pd.concat([p_50, ec_kupiec_df.iloc[1:2]], ignore_index=True)

        elif nc == 0.9:
            if ACE_90 is None:
                ACE_90 = ec_kupiec_df.iloc[0:1]
                p_90 = ec_kupiec_df.iloc[1:2]
            else:
                ACE_90 = pd.concat([ACE_90, ec_kupiec_df.iloc[0:1]], ignore_index=True)
                p_90 = pd.concat([p_90, ec_kupiec_df.iloc[1:2]], ignore_index=True)

# Adjust ACE values
ACE_50 = ACE_50 - 0.5
ACE_90 = ACE_90 - 0.9

# Define method groups
method_groups = [
    ['QRA', 'Stat-qEns', 'Stat-QRM', 'DNN-QRA', 'DNN-QRM'],
    ['LQRA(BIC)', 'EQRA(BIC-0.25)', 'EQRA(BIC-0.5)', 'EQRA(BIC-0.75)'],
    ['DDNN-L-1', 'DDNN-L-2', 'DDNN-L-3', 'DDNN-L-4', 'DDNN-L-pEns', 'DDNN-L-qEns'],
    ['DDNN-E-1', 'DDNN-E-2', 'DDNN-E-3', 'DDNN-E-4', 'DDNN-E-pEns', 'DDNN-E-qEns'],
    ['DDNN-LQRA(BIC)-qEns', 'DDNN-EQRA(BIC)-qEns', 'DDNN-LQRA(BIC)-wqEns']
]

# Split methods into 5 groups
method_groups = [
    ['QRA', 'Stat-qEns', 'Stat-QRM', 'DNN-QRA', 'DNN-QRM'],
    ['LQRA(BIC)', 'EQRA(BIC-0.25)', 'EQRA(BIC-0.5)', 'EQRA(BIC-0.75)'],
    ['DDNN-L-1', 'DDNN-L-2', 'DDNN-L-3', 'DDNN-L-4', 'DDNN-L-pEns', 'DDNN-L-qEns'],
    ['DDNN-E-1', 'DDNN-E-2', 'DDNN-E-3', 'DDNN-E-4', 'DDNN-E-pEns', 'DDNN-E-qEns'],
    ['DDNN-LQRA(BIC)-qEns', 'DDNN-EQRA(BIC)-qEns', 'DDNN-LQRA(BIC)-wqEns']
]

hours = np.arange(1, 25)

# Function to create subplot grid based on number of methods
def get_subplot_dimensions(n_methods):
    n_rows = (n_methods + 2) // 3  # Round up to nearest multiple of 3
    n_cols = min(3, n_methods)
    return n_rows, n_cols


# Create separate figure for each group
for group_idx, group_methods in enumerate(method_groups):
    n_methods = len(group_methods)
    n_rows, n_cols = get_subplot_dimensions(n_methods)

    # Create figure with appropriate size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True)
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each method in the group
    for i, method in enumerate(group_methods):
        method_idx = methods.index(method)
        ax = axes[i]

        # Determine fill status based on p-value threshold
        filled_50 = p_50.iloc[method_idx, :].values > 0.01
        filled_90 = p_90.iloc[method_idx, :].values > 0.01

        # First draw the error bars
        ax.vlines(hours, ACE_50.iloc[method_idx, :].values, 0, color=color_3, linewidth=0.8, alpha=0.5)
        ax.vlines(hours, ACE_90.iloc[method_idx, :].values, 0, color=color_2, linewidth=0.8, alpha=0.5)

        # Then draw the markers on top
        # 90%: Blue circles
        ax.scatter(hours, ACE_90.iloc[method_idx, :].values,
                   facecolors=np.where(filled_90, color_2, 'none'),
                   edgecolors=color_2, marker='o', s=10, zorder=3)

        # 50%: Red squares
        ax.scatter(hours, ACE_50.iloc[method_idx, :].values,
                   facecolors=np.where(filled_50, color_3, 'none'),
                   edgecolors=color_3, marker='s', s=10, zorder=3)

        # Layout settings
        ax.axhline(0, color='black', linewidth=0.8, zorder=1)
        ax.set_ylim(min(ACE_50.min().min(), ACE_90.min().min()) - 0.04,
                    max(ACE_50.max().max(), ACE_90.max().max()) + 0.04)
        ax.set_xticks([2, 6, 10, 14, 18, 22])
        ax.set_title(method, fontsize=10, fontweight="bold")

        # Add legend with empty markers to each subplot
        ax.scatter([], [], facecolors='none', edgecolors=color_2, marker='o', s=10, label='90%')
        ax.scatter([], [], facecolors='none', edgecolors=color_3, marker='s', s=10, label='50%')
        ax.legend(loc="upper right", fontsize=8, frameon=False)

        # Labels
        if i % n_cols == 0:
            ax.set_ylabel("ACE")
        if i >= n_methods - n_cols:
            ax.set_xlabel("Hour")

    # Hide empty subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.savefig(f"kupiec_plots_group_{group_idx}_ts{time_span}.png", dpi=300, bbox_inches="tight")
    plt.show()

