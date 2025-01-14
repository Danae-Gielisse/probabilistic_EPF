import pandas as pd
import numpy as np

results = []

def calculate_mae(real_col, forecast_col):
    return np.mean(np.abs(real_col - forecast_col))


def calculate_rmse(real_col, forecast_col):
    return np.sqrt(np.mean((real_col - forecast_col) ** 2))


def forecast_to_point_error(forecast_df):
    real_prices = forecast_df['Price']
    forecasted_prices = forecast_df['Percentile_50']
    mae = calculate_mae(real_prices, forecasted_prices)
    rmse = calculate_rmse(real_prices, forecasted_prices)

    return mae, rmse


def append_to_results(ts, dis, run, reg, mae, rmse):
    results.append({
        'time_span': ts,
        'distribution': dis,
        'run': run,
        'regularization': reg,
        'mae': mae,
        'rmse': rmse
    })




# define time span and distribution list
time_span_list = [1, 2]
distribution_list = ['point', 'jsu']
regularization_list = ['lasso', 'enet']


for time_span in time_span_list:
    # calculate naive errors
    forecast = pd.read_csv(f'../Results/point_forecasting_time_span_{time_span}/naive_benchmark.csv')
    mae = calculate_mae(forecast['price_real'], forecast['forecasted_price'])
    rmse = calculate_rmse(forecast['price_real'], forecast['forecasted_price'])
    append_to_results(time_span, None, 'naive', None, mae, rmse)

    # calculate DDNN-Ens errors
    DDNN_run_list = [1, 2, 3, 4]
    forecast_list = []
    for run in DDNN_run_list:
        forecast = pd.read_csv(f'../Results/point_forecasts_NN_time_span_{time_span}/point{run}_lasso.csv')
        forecast_list.append(forecast['forecast'])
        real_price = forecast['real']
    average_forecast = pd.concat(forecast_list, axis=1).mean(axis=1)
    mae = calculate_mae(real_price, average_forecast)
    rmse = calculate_rmse(real_price, average_forecast)
    append_to_results(time_span, None, 'DNN_Ens', 'lasso', mae, rmse)

    for distribution in distribution_list:
        if distribution == 'jsu':
            run_list = [1, 2, 3, 4, 'pEns', 'qEns']
        else:
            run_list = ['qra', 'qrm']
        for run in run_list:
            if distribution == 'jsu':
                for reg in regularization_list:
                    # define forecast path
                    forecast_path = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_jsu{run}_{reg}.csv'
                    forecast = pd.read_csv(forecast_path, index_col=0)

                    # calculate mean absolute error and root mean squared error
                    mae, rmse = forecast_to_point_error(forecast)

                    # Append the results to the list
                    append_to_results(time_span, distribution, run, reg, mae, rmse)
            else:
                # define forecast path
                forecast_path = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_point_{run}.csv'
                forecast = pd.read_csv(forecast_path)

                # calculate mean absolute error and root mean squared error
                mae, rmse = forecast_to_point_error(forecast)

                append_to_results(time_span, distribution, run, 'lasso', mae, rmse)

    # calculate mae and rmse for LQRA(0)
    forecast = pd.read_csv(f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_lambda_0.0.csv')
    mae, rmse = forecast_to_point_error(forecast)
    append_to_results(time_span, 'LQRA', 'lambda_0', 'lasso', mae, rmse)
    # calculate mae and rmse for LQRA(BIC)
    forecast = pd.read_csv(f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_BIC.csv')
    mae, rmse = forecast_to_point_error(forecast)
    append_to_results(time_span, 'LQRA', 'BIC', 'lasso', mae, rmse)

    # calculate mae and rmse for EQRA(BIC) for different alphas
    alpha_list = [0.25, 0.5, 0.75]
    for alpha in alpha_list:
        forecast = pd.read_csv(f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_BIC_alpha_{alpha}.csv')
        mae, rmse = forecast_to_point_error(forecast)
        append_to_results(time_span, 'EQRA', f'BIC_alpha_{alpha}', 'enet', mae, rmse)

    # calculate mae and rmse for q-ens-stat
    forecast = pd.read_csv(f'../Results/probabilistic_forecasts_time_span_{time_span}/q_Ens_forecast.csv')
    mae, rmse = forecast_to_point_error(forecast)
    append_to_results(time_span, None, 'q-Ens', None, mae, rmse)
    # calculate mae and rmse for QRM
    forecast = pd.read_csv(f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_qrm.csv')
    mae, rmse = forecast_to_point_error(forecast)
    append_to_results(time_span, None, 'QRM', None, mae, rmse)

    # calculate mae and rmse for ensembles
    ensemble_list = ['lasso', 'enet', 'lasso_weighted']
    for ens in ensemble_list:
        forecast = pd.read_csv(f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_stat_nn_ens_{ens}.csv')
        mae, rmse = forecast_to_point_error(forecast)
        append_to_results(time_span, 'ens', ens, None, mae, rmse)

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

results_df.to_csv('../Results/Evaluation_metrics/point_forecasting_errors.csv', index=False)



