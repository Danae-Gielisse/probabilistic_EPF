"""
Creates battery plots for illustration trading strategy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Loop through both time spans
for idx, time_span in enumerate([1, 2]):
    # Get the current axis
    ax = axes[idx]

    # Load data for current time_span
    forecast_df = pd.read_csv(
        f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_stat_nn_ens_lasso_weighted.csv',
        parse_dates=['Datetime']
    )

    # Define colors
    color_1 = '#A0522D'
    color_2 = '#4682B4'
    color_3 = '#8B0000'
    color_4 = '#6A5ACD'

    # Filter the data for only July 25th 2019/2024
    if time_span == 1:
        selected_date = '2019-07-25'
        subtitle = f"{selected_date} (time span 1)"
    else:
        selected_date = '2024-07-25'
        subtitle = f"{selected_date} (time span 2)"

    filtered_df = forecast_df[forecast_df['Datetime'].dt.date == pd.to_datetime(selected_date).date()]

    # Create a new column for the hours (0 to 23)
    filtered_df['Hour'] = range(24)

    # Get the date (for the title)
    plot_date = filtered_df['Datetime'].dt.date.iloc[0]

    # Determine h1 and h2 and the corresponding prices
    h1, h2 = np.argmin(filtered_df['Percentile_50']), np.argmax(filtered_df['Percentile_50'])
    bid_price = filtered_df['Percentile_90'].iloc[h1]
    offer_price = filtered_df['Percentile_10'].iloc[h2]

    # Plot the median (Percentile_50)
    ax.plot(filtered_df['Hour'], filtered_df['Percentile_50'], label='Median forecast', color=color_3)

    # Add the 90% prediction interval (gray area)
    ax.fill_between(
        filtered_df['Hour'],
        filtered_df['Percentile_10'],
        filtered_df['Percentile_90'],
        color='gray', alpha=0.2, label='80% Prediction Interval'
    )

    # Add markers
    ax.scatter(h1, bid_price, color='black', marker='o', s=50, label='Selected price limits')
    ax.scatter(h2, offer_price, color='black', marker='o', s=50)
    ax.scatter(h1, filtered_df['Percentile_50'].iloc[h1], color=color_3, marker='s', s=50,
               label=r'Selected $h_1$,$h_2$')
    ax.scatter(h2, filtered_df['Percentile_50'].iloc[h2], color=color_3, marker='s', s=50)

    if time_span == 2:
        to_add_or_subtract = 3
    else:
        to_add_or_subtract = 0.75
    # Add text labels
    ax.text(h1, bid_price + to_add_or_subtract, r'$\hat{P}_{d,h_1}^{0.9}$', fontsize=12, color='black', ha='center', va='bottom',
            fontweight='bold')
    ax.text(h2, offer_price - to_add_or_subtract, r'$\hat{P}_{d,h_2}^{0.1}$', fontsize=12, color='black', ha='center', va='top',
            fontweight='bold')
    ax.text(h1, filtered_df['Percentile_50'].iloc[h1] - to_add_or_subtract, r'$\hat{P}_{d,h_1}^{0.5}$', fontsize=12, color='black',
            ha='center', va='top', fontweight='bold')
    ax.text(h2, filtered_df['Percentile_50'].iloc[h2] + to_add_or_subtract, r'$\hat{P}_{d,h_2}^{0.5}$', fontsize=12, color='black',
            ha='center', va='bottom', fontweight='bold')

    # Calculations for the vertical line and arrows
    mid_x = (h1 + h2) / 2
    y1 = filtered_df['Percentile_50'].iloc[h1]
    y2 = filtered_df['Percentile_50'].iloc[h2]
    mid_y = (y1 + y2) / 2

    # Add vertical double arrow
    arrow = FancyArrowPatch((mid_x, y1), (mid_x, y2),
                            arrowstyle='<->',
                            color='black',
                            linewidth=1.5,
                            mutation_scale=15)
    ax.add_patch(arrow)

    # Add dashed lines
    ax.plot([h1, mid_x], [y1, y1], 'k--', linewidth=1)
    ax.plot([h2, mid_x], [y2, y2], 'k--', linewidth=1)

    # Add "max median difference" text in the middle of the arrow
    # Position depends on time_span value
    if time_span == 1:
        # Right side of the arrow
        text_x = mid_x + 0.5
        ha_align = 'left'
    else:
        # Left side of the arrow
        text_x = mid_x - 0.5
        ha_align = 'right'

    ax.text(text_x, mid_y, "max median\ndifference",
            color='black',
            fontsize=9,
            ha=ha_align,
            va='center',
            fontweight='bold')

    # Add subplot title
    ax.set_title(subtitle)

    # Labels for axes
    ax.set_xlabel('Hour')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.set_xticks(range(24))
    ax.set_xlim([0, 23])

    # Only add legend to the first subplot to avoid redundancy
    if idx == 0:
        ax.legend(loc='upper left')

# Add a main title for the entire figure
#fig.suptitle('DDNN-LQRA(BIC)-wqEns forecast with 80% Prediction Interval', fontsize=16)

# Improve layout and show plot
plt.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.1)  # Make room for the main title
plt.show()