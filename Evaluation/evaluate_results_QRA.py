import pandas as pd
import numpy as np
import statistics
from scipy.stats import chi2
import os

# choose time span, alpha and nominal coverage
time_span = 1
alpha = 0 # choose 0, 0.25, 0.5 or 0.75
percentage = 0.5

significance_levels = [0.01, 0.05, 0.1]

# define paths
forecast_path = f'../Results/probabilistic_forecasts_time_span_{time_span}'
if alpha != 0:
    cprs_path = f'../Results/Evaluation_metrics/CRPS_ts{time_span}_alpha_{alpha}.csv'
    forecast_BIC = pd.read_csv(os.path.join(forecast_path, f'forecast_BIC_alpha_{alpha}.csv'),
                               index_col=[0, 1]).reset_index(drop=True)
    kupiec_output_path = f'../Results/Evaluation_metrics/kupiec_passes_alpha_{alpha}_nc{percentage}_ts{time_span}'
    ec_path_output = f'../Results/Evaluation_metrics/empirical_coverage_alpha_{alpha}_nc{percentage}_ts{time_span}.csv'

else:
    cprs_path = f'../Results/Evaluation_metrics/CRPS_ts{time_span}.csv'
    forecast_BIC = pd.read_csv(os.path.join(forecast_path, f'forecast_BIC.csv'),
                               index_col=[0, 1]).reset_index(drop=True)
    kupiec_output_path = f'../Results/Evaluation_metrics/kupiec_passes_nc{percentage}_ts{time_span}'
    ec_path_output = f'../Results/Evaluation_metrics/empirical_coverage_nc{percentage}_ts{time_span}.csv'

# define correct lambda array
LAMBDA = np.concatenate(([0], np.logspace(-1, 3, 19)))
if alpha != 0:
    if time_span == 1:
        LAMBDA = LAMBDA[[8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
    else:
        LAMBDA = LAMBDA[[9, 10, 11, 12, 13, 14, 15, 16, 18, 19]]

forecast_list = []
for i in range(0, len(LAMBDA)):
    if alpha != 0:
        probabilistic_forecast_folder = os.path.join(forecast_path, 'forecast_lambda_' + str(LAMBDA[i]) + '_alpha_' + str(alpha) + '.csv')
    else:
        probabilistic_forecast_folder = os.path.join(forecast_path, 'forecast_lambda_' + str(LAMBDA[i]) + '.csv')
    forecast = pd.read_csv(probabilistic_forecast_folder)
    forecast_list.append(forecast)

def create_CRPS_matrix(forecast):
    number_of_days = int(round(len(forecast) / 24))
    crps_df = pd.DataFrame(index=range(0, number_of_days), columns=range(0, 24))

    # fill CPRS dataframe with average pinball score for every day and hour
    for day in range(0, number_of_days):
        if day == 400:
            print('day 400 is done')
        for hour in range(0, 24):
            crps_df.loc[day, hour] = CRPS(day, hour, forecast)

    # add column with the average pinball score per day
    crps_df['average_pinball_score_day'] = crps_df.mean(axis=1)

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

"""
# compute CPRS for BIC
CRPS_BIC_df = create_CRPS_matrix(forecast_BIC)
average_CRPS_BIC = CRPS_BIC_df['average_pinball_score_day'].mean()

# create df for mean CPRS for every lambda
df_mean_CRPS = pd.DataFrame(index=['CRPS'])
for i in range(0, len(forecast_list)):
    CRPS_df = create_CRPS_matrix(forecast_list[i])
    average_CPRS = CRPS_df['average_pinball_score_day'].mean()
    lambda_value = LAMBDA[i]
    print('lambda value ' + str(lambda_value) + ' done')
    df_mean_CRPS[str(lambda_value)] = average_CPRS
# add BIC result to CRPS_df
df_mean_CRPS['BIC'] = average_CRPS_BIC

# save CPRS dataframe
df_mean_CRPS.to_csv(cprs_path)
"""

### methods for computing emperical coverage ###
'''
computes the emperical coverage for a forecast dataframe for a certain percentage
'''
def empirical_coverage(forecast, percentage):
    lowerbound_quantile = int(round(((1-percentage) / 2) * 100))
    upperbound_quantile = int(round(100 - lowerbound_quantile))
    quantile_low = 'Percentile_' + str(lowerbound_quantile)
    quantile_high = 'Percentile_' + str(upperbound_quantile)
    empirical_coverage_list = []
    for i in range(0, len(forecast)):
        lowerbound = forecast.loc[i, quantile_low]
        upperbound = forecast.loc[i, quantile_high]
        real_price = forecast.loc[i, 'Price']
        if lowerbound <= real_price <= upperbound:
            empirical_coverage_list.append(1)
        else:
            empirical_coverage_list.append(0)

    return statistics.mean(empirical_coverage_list), empirical_coverage_list


'''
computes the emperical coverage for a specific hour
'''
def empirical_coverage_hour(forecast, percentage, hour):
    idx = np.arange(hour, len(forecast), 24)
    hour_forecast = forecast.loc[idx].reset_index(drop=True)
    lowerbound_quantile = int(round(((1-percentage) / 2) * 100))
    upperbound_quantile = int(round(100 - lowerbound_quantile))
    quantile_low = 'Percentile_' + str(lowerbound_quantile)
    quantile_high = 'Percentile_' + str(upperbound_quantile)
    empirical_coverage_list_hour = []
    for i in range(0, len(hour_forecast)):
        lowerbound_hour = hour_forecast.loc[i, quantile_low]
        upperbound_hour = hour_forecast.loc[i, quantile_high]
        real_price = hour_forecast.loc[i, 'Price']
        if lowerbound_hour <= real_price <= upperbound_hour:
            empirical_coverage_list_hour.append(1)
        else:
            empirical_coverage_list_hour.append(0)

    return statistics.mean(empirical_coverage_list_hour), empirical_coverage_list_hour

'''
Performs the kupiec test
'''
def kupiec_test(emperical_coverage_list, confidence_level):
    # calculate the number of times that the real price is in the interval
    n1 = sum(emperical_coverage_list)
    # calculate the number of times that the real price is outside the interval
    n0 = len(emperical_coverage_list) - n1
    # define the total number of observations
    total_observations = len(emperical_coverage_list)

    # calculate nominal coverage (c) and observed failure rate (π)
    c = 1 - confidence_level
    pi = n1 / total_observations

    # compute the likelihood ratio (LR) statistic
    lr_uc = -2 * (
            n0 * np.log(1 - c) + n1 * np.log(c) -
            n0 * np.log(1 - pi) - n1 * np.log(pi)
    )

    # compute the p-value
    p_value = 1 - chi2.cdf(lr_uc, df=1)

    return p_value


kupiec_results = []
LAMBDA = np.append(LAMBDA, -1)# for BIC
for significance_level in significance_levels:
    level_results = []
    for lambda_value in LAMBDA:
        if lambda_value == -1: # then BIC forecast needed
            if alpha != 0:
                forecast_path = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_BIC_alpha_{alpha}.csv'
            else:
                forecast_path = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_BIC.csv'
        else:
            if alpha != 0:
                forecast_path = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_lambda_{lambda_value}_alpha_{alpha}.csv'
            else:
                forecast_path = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_lambda_{lambda_value}.csv'
        forecast = pd.read_csv(forecast_path)

        # perform kupiec test
        number_of_passes_dict = {}
        # perform the kupiec test for all 24 hours
        kupiec_list = []
        for hour in range(0, 24):
            ec_hour, coverage_list_hour = empirical_coverage_hour(forecast, percentage, hour)
            kupiec = kupiec_test(coverage_list_hour, 1-percentage)
            kupiec_list.append(kupiec)

        # count number of passes
        number_of_passes = 0
        for j in kupiec_list:
            if j <= significance_level:
                number_of_passes += 1
        level_results.append(number_of_passes)

    # append the level results to the kupiec_list
    kupiec_results.append(level_results)

# create df with number of passes for each run
kupiec_df = pd.DataFrame(kupiec_results, columns=LAMBDA, index=[f"sig_{sl}" for sl in significance_levels])
kupiec_df = kupiec_df.rename(columns={'-1': 'BIC'})

# save dataframe
kupiec_df.to_csv(kupiec_output_path)

coverage_results = []
for lambda_value in LAMBDA:
    if lambda_value == -1:  # then BIC forecast needed
        if alpha != 0:
            forecast_path = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_BIC_alpha_{alpha}.csv'
        else:
            forecast_path = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_BIC.csv'
    else:
        if alpha != 0:
            forecast_path = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_lambda_{lambda_value}_alpha_{alpha}.csv'
        else:
            forecast_path = f'../Results/probabilistic_forecasts_time_span_{time_span}/forecast_lambda_{lambda_value}.csv'
    forecast = pd.read_csv(forecast_path)
    # calculate empirical coverage
    coverage, _ = empirical_coverage(forecast, percentage)
    # add coverage to results list
    coverage_results.append(coverage)

# create results dataframe
ec_df = pd.DataFrame([coverage_results], columns=LAMBDA)
ec_df = ec_df.rename(columns={'-1': 'BIC'})

# save CRPS dataframe
ec_df.to_csv(ec_path_output)
