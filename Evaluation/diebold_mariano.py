import pandas as pd
import numpy as np
import statistics
from scipy import stats
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import LinearSegmentedColormap


def create_CRPS_matrix(forecast):
    number_of_days = int(round(len(forecast) / 24))
    crps_df = pd.DataFrame(index=range(0, number_of_days), columns=range(0, 24))

    # fill CPRS dataframe with average pinball score for every day and hour
    for day in range(0, number_of_days):
        if day == 400:
            print('day 400 is done')
        for hour in range(0, 24):
            crps_df.loc[day, hour] = CRPS(day, hour, forecast)

    return crps_df


def CRPS(day, hour, forecast):
    # average over all tau's, to calculate the average pinball score
    tau = np.arange(1, 100) / 100
    pinball_list = []
    for q in tau:
        pinball_list.append(pinball_score(q, day, hour, forecast))
    return statistics.mean(pinball_list)


def pinball_score(tau, day, hour, forecast):
    # calculate the pinball score for every day, hour and tau combination
    idx = 24 * day + hour
    real_price = forecast.loc[idx, 'Price']
    quantile = int(round(tau * 100))
    column_quantile = 'Percentile_' + str(quantile)
    price_quantile_predicted = forecast.loc[idx, column_quantile]
    if real_price < price_quantile_predicted:
        pinball = (1-tau) * (price_quantile_predicted - real_price)
    else:
        pinball = tau * (real_price - price_quantile_predicted)

    return pinball


def compute_vector_dt(CRPS_matrix_A, CRPS_matrix_B):
    # only take tail rows if length of test data is not equal
    min_rows = min(len(CRPS_matrix_A), len(CRPS_matrix_B))
    CRPS_matrix_A = CRPS_matrix_A.tail(min_rows).reset_index(drop=True)
    CRPS_matrix_B = CRPS_matrix_B.tail(min_rows).reset_index(drop=True)

    # calculate L1 norm for both dataframes
    l1_norm_A = CRPS_matrix_A.abs().sum(axis=1)
    l1_norm_B = CRPS_matrix_B.abs().sum(axis=1)

    # calculate the differences for each day in the test data
    dt_vector = l1_norm_A - l1_norm_B

    return dt_vector


def dm_test(dt_vec):
    dm = np.sqrt(len(dt_vec)) * (dt_vec.mean() / dt_vec.std())
    p_val = 1 - stats.norm.cdf(dm)

    return dm, p_val


def load_forecast(method):
    method_index = methods.index(method)
    method_name = method_names[method_index]

    if method_name == 'q_Ens':
        path = f'../Results/probabilistic_forecasts_time_span_{time_span}/{method_name}_forecast.csv'
    elif method_name in stat_methods:
        path = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_{method_name}.csv'
    elif method_name in nn_methods:
        path = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_{method_name}.csv'
    if method_name in no_index_col_methods:
        forecast = pd.read_csv(path)
    else:
        forecast = pd.read_csv(path, index_col=0)
    return forecast


# choose time span
time_span = 1
no_index_col_methods = ['lambda_0.0', 'point_qra']

methods = ['QRA', 'Stat-qEns', 'Stat-QRM', 'DNN-QRA', 'DNN-QRM', 'LQRA(BIC)', 'EQRA(BIC-0.25)', 'EQRA(BIC-0.5)',
             'EQRA(BIC-0.75)', 'DDNN-L-1', 'DDNN-L-2', 'DDNN-L-3', 'DDNN-L-4', 'DDNN-L-pEns', 'DDNN-L-qEns',
             'DDNN-E-1', 'DDNN-E-2', 'DDNN-E-3', 'DDNN-E-4', 'DDNN-E-pEns', 'DDNN-E-qEns', 'DDNN-LQRA(BIC)-qEns',
             'DDNN-EQRA(BIC)-qEns', 'DDNN-LQRA(BIC)-wqEns']

method_names = ['lambda_0.0', 'q_Ens', 'qrm', 'point_qra', 'point_qrm', 'BIC', 'BIC_alpha_0.25',
                'BIC_alpha_0.5', 'BIC_alpha_0.75', 'jsu1_lasso', 'jsu2_lasso', 'jsu3_lasso', 'jsu4_lasso',
                'jsupEns_lasso', 'jsuqEns_lasso', 'jsu1_enet', 'jsu2_enet', 'jsu3_enet', 'jsu4_enet', 'jsupEns_enet',
                'jsuqEns_enet', 'stat_nn_ens_enet', 'stat_nn_ens_lasso', 'stat_nn_ens_lasso_weighted']

stat_methods = ['lambda_0.0', 'qrm', 'BIC', 'BIC_alpha_0.25', 'BIC_alpha_0.5', 'BIC_alpha_0.75', 'stat_nn_ens_enet',
                'stat_nn_ens_lasso', 'stat_nn_ens_lasso_weighted']
nn_methods = ['point_qra', 'point_qrm', 'jsu1_lasso', 'jsu2_lasso', 'jsu3_lasso', 'jsu4_lasso', 'jsupEns_lasso',
              'jsuqEns_lasso', 'jsu1_enet', 'jsu2_enet', 'jsu3_enet', 'jsu4_enet', 'jsupEns_enet', 'jsuqEns_enet' ]
'''
crps_matrices = {method: create_CRPS_matrix(load_forecast(method)) for method in methods}

with open(f"crps_matrices_ts{time_span}.pkl", "wb") as f:
    pickle.dump(crps_matrices, f)
'''
with open(f"crps_matrices_ts{time_span}.pkl", "rb") as f:
    crps_matrices = pickle.load(f)

# Initialise p-value matrix and insert zero's
p_value_matrix = pd.DataFrame(np.zeros((len(methods), len(methods))), index=methods, columns=methods)

# compute p-values and fill p_value matrix
for method_A, method_B in itertools.combinations(methods, 2):
    dt_vector = compute_vector_dt(crps_matrices[method_A], crps_matrices[method_B])
    _, p_value = dm_test(dt_vector)
    p_value_2 = 1- p_value

    p_value_matrix.loc[method_A, method_B] = p_value
    p_value_matrix.loc[method_B, method_A] = p_value_2

# fill the diagonals with ones
np.fill_diagonal(p_value_matrix.values, 1)

# define colors
colors = ["green", "yellow", "red"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# create a copy of the p-value matrix with nan for p-values greater or equal to 10%
p_value_matrix_clipped = p_value_matrix.copy()
p_value_matrix_clipped[p_value_matrix > 0.1] = np.nan

# create heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(p_value_matrix_clipped, cmap=cmap, annot=False, cbar=True, linewidths=0.5, linecolor='gray',
                 vmin=0, vmax=0.1, cbar_kws={"label": "p-value"})
ax.set_facecolor("black")

# fill diagonals grey
for i in range(len(p_value_matrix)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, color="gray", lw=2))
    plt.plot(i + 0.5, i + 0.5, marker="x", color="black", markersize=10, mew=2)

# define labels
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.title(f"Heatmap Diebold-Mariano test time span {time_span}", fontsize=14)

# give the labels space
plt.tight_layout()

# plot the figure
plt.show()
