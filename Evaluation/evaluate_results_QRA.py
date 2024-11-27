import pandas as pd
import numpy as np
import statistics
from scipy.stats import chi2
import os

# choose time span
time_span = 2
# choose nominal coverage and significance level
percentage = 0.5
significance_level = 0.01

forecast_path = f'Results/probabilistic_forecasts_time_span_{time_span}'
cprs_path = f'Results/Evaluation_metrics/CPRS_ts{time_span}.csv'
PICP_path = f'Results/Evaluation_metrics/emperical_coverage_ts{time_span}_nc{percentage}.csv'

# create list of all forecasts for all lambdas
LAMBDA = np.concatenate(([0], np.logspace(-1, 3, 19)))

forecast_list = []
for i in range(0, len(LAMBDA)):
    probabilistic_forecast_folder = os.path.join(forecast_path, 'forecast_lambda_' + str(LAMBDA[i]) + '.csv')
    forecast = pd.read_csv(probabilistic_forecast_folder)
    forecast_list.append(forecast)
"""
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


# compute CPRS for BIC
forecast = pd.read_csv(os.path.join(forecast_path, 'forecast_BIC.csv'), index_col=[0, 1]).reset_index(drop=True)
CRPS_BIC_df = create_CRPS_matrix(forecast)
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


# perform kupiec test for LQRA(BIC) and LQRA(46)
forecast_list_2 = []
forecast_BIC = pd.read_csv(os.path.join(forecast_path, 'forecast_BIC.csv'), index_col=[0, 1]).reset_index(drop=True)
forecast_list_2.append(forecast_BIC)
forecast_list_2.append(pd.read_csv(os.path.join(forecast_path, 'forecast_lambda_46.41588833612777.csv')))
number_of_passes_dict = {}
for i, forecast in enumerate(forecast_list_2):
    # perform the kupiec test for all 24 hours
    kupiec_list = []
    for hour in range(0, 24):
        ec_hour, coverage_list_hour = empirical_coverage_hour(forecast, percentage, hour)
        kupiec = kupiec_test(coverage_list_hour, 1-percentage)
        kupiec_list.append(kupiec)

    number_of_passes = 0
    for j in kupiec_list:
        if j <= significance_level:
            number_of_passes += 1

    if i == 0:
        number_of_passes_dict['LQRA(BIC)'] = number_of_passes
    elif i == 1:
        number_of_passes_dict['LQRA(46)'] = number_of_passes

# create dataframe for empirical coverage in general
df_empirical_coverage = pd.DataFrame(index=['Empirical_coverage'], columns=LAMBDA)
for i in range(0, len(forecast_list)):
    coverage, _ = empirical_coverage(forecast_list[i], percentage)
    df_empirical_coverage.loc['Empirical_coverage', LAMBDA[i]] = coverage
# add coverage LQRA(BIC)
coverage_BIC, _ = empirical_coverage(forecast_BIC, percentage)
df_empirical_coverage.loc['Empirical_coverage', 'BIC'] = coverage_BIC

# save PICP dataframe
df_empirical_coverage.to_csv(PICP_path)
