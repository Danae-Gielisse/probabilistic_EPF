"""
Creates the graphs for in the results section
"""

import pandas as pd
import matplotlib.pyplot as plt

# choose time span
time_span = 1
if time_span == 1:
    folder_coverage = 'Results/Evaluation_metrics/emperical_coverage_ts1.csv'
    folder_CPRS = 'Results/Evaluation_metrics/CPRS_ts1.csv'
else:
    folder_coverage = 'Results/Evaluation_metrics/emperical_coverage_ts2.csv'
    folder_CPRS = 'Results/Evaluation_metrics/CPRS_ts2.csv'

# get data for lambda and PICP_values
coverage_df = pd.read_csv(folder_coverage, index_col=0)
lambda_values = coverage_df.columns.astype(float)  # Lambda waarden
picp_values = coverage_df.loc['Empirical_coverage'].values

# scatter plot for PICP vs Lambda values
plt.figure(figsize=(10, 6))
plt.scatter(lambda_values, picp_values, color='b', edgecolors='b', facecolors='none', label='Empirical Coverage (PICP)')
plt.xlabel('Lambda Value')
plt.ylabel('Empirical Coverage (PICP)')
plt.title('Scatter Plot of Lambda vs. Empirical Coverage (PICP)')
plt.xscale('log')
plt.ylim(0, 1)
plt.axhline(y=0.9, color='r', linestyle='--', label='Target PICP = 0.9')
plt.xticks([1, 10, 100, 1000], ['10^0', '10^1', '10^2', '10^3'])
plt.legend()
plt.show()

# get data for lambda and APS values
CRPS_df = pd.read_csv(folder_CPRS, index_col=0)
lambda_values = CRPS_df.columns.astype(float)
APS = CRPS_df.loc['CRPS'].values

# scatter plot for APS vs Lambda values
plt.figure(figsize=(10, 6))
plt.scatter(lambda_values, APS, color='b', edgecolors='b', facecolors='none', label='Average Pinball Score (APS)')
plt.xlabel('Lambda Value')
plt.ylabel('Average Pinball Score (APS)')
plt.title('Scatter Plot of Lambda vs. Average Pinball Score (APS)')
plt.xscale('log')
plt.xticks([1, 10, 100, 1000], ['10^0', '10^1', '10^2', '10^3'])
plt.legend()
plt.show()



